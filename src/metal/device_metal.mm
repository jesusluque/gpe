// Copyright (c) 2026 gpe contributors.
//
// The Metal backend. Nothing here knows the pool exists.
//
// A .mm, and the only one. metal-cpp is a header-only C++ wrapper over the
// Objective-C runtime and needs no Objective-C source at all -- but it has to
// be compiled as Objective-C++ so that the runtime is linked and its ARC rules
// apply, and one file is the whole cost of that.
//
// WHAT SLANG'S OUTPUT ASKS FOR
//
// Read off the generated .metal rather than assumed:
//
//     [[kernel]] void scaleMain(uint3 tid [[thread_position_in_grid]],
//                               ScaleParams constant* params [[buffer(2)]],
//                               packed_float4 device* dst    [[buffer(1)]],
//                               packed_float4 device* src    [[buffer(0)]])
//
// Buffers take slots in the order the .slang declares them and the uniforms
// take the slot after the last one -- which is exactly the shape of gpe::Args,
// so the translation is a loop and not a table. The CUDA side had to build a
// struct of pointers and counts and copy it into __constant__; this binds.
//
// `thread_position_in_grid` rather than a threadgroup index also means the
// dispatch can be in threads, with no rounding up to whole groups. The kernel
// still bounds-checks, because the CUDA side does round up and one kernel
// serves both.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "completion.h"
#include "hostring.h"
#include "gpe/args.h"
#include "gpe/device.h"
#include "gpe_kernels.h"

namespace gpe {
namespace {

class MetalDevice final : public Device,
                         public CompletionReporting,
                         public HostStaging {
public:
    ~MetalDevice() override {
        for (MTL::Buffer* buffer : buffers_) {
            if (buffer != nullptr) {
                buffer->release();
            }
        }
        for (const Kernel& kernel : kernels_) {
            kernel.pipeline->release();
        }
        for (const HostBuffer& host : hostBuffers_) {
            host.buffer->release();
        }
        for (const Staging& slot : uploads_) {
            if (slot.buffer != nullptr) {
                slot.buffer->release();
            }
        }
        if (uploadEvent_ != nullptr) {
            uploadEvent_->release();
        }
        if (blitQueue_ != nullptr) {
            blitQueue_->release();
        }
        if (queue_ != nullptr) {
            queue_->release();
        }
        if (device_ != nullptr) {
            device_->release();
        }
    }

    [[nodiscard]] bool open() {
        device_ = MTL::CreateSystemDefaultDevice();
        if (device_ == nullptr) {
            return false;
        }
        queue_ = device_->newCommandQueue();
        // A second queue for transfers, and an event to order them against the
        // first. Same reason as the CUDA copy stream: a blit and a kernel on
        // two queues run at once because they are different hardware, not
        // because two threads issued them, so one thread issuing to two queues
        // gets the whole of the benefit.
        blitQueue_ = device_->newCommandQueue();
        uploadEvent_ = device_->newEvent();
        return queue_ != nullptr && blitQueue_ != nullptr &&
               uploadEvent_ != nullptr;
    }

    [[nodiscard]] Backend backend() const override { return Backend::Metal; }

    [[nodiscard]] BufferId alloc(size_t bytes) override {
        // Private storage: the GPU's own memory, not visible to the CPU.
        //
        // On a unified-memory machine `Shared` would let upload be a memcpy and
        // skip a copy entirely, and it is worth measuring -- but it is not free
        // either, because a shared buffer the GPU is reading is a buffer the CPU
        // must not touch, and the engine has no way to say when that is. Private
        // plus an explicit blit is the behaviour that matches CUDA, which is
        // what makes one client work on both.
        MTL::Buffer* buffer =
            device_->newBuffer(bytes, MTL::ResourceStorageModePrivate);
        if (buffer == nullptr) {
            return kInvalidBuffer;
        }
        buffers_.push_back(buffer);
        return static_cast<BufferId>(buffers_.size());   // 1-based; 0 is invalid
    }

    void release(BufferId id) override {
        if (MTL::Buffer** slot = find(id); slot != nullptr && *slot != nullptr) {
            (*slot)->release();
            *slot = nullptr;
        }
    }

    void upload(BufferId id, const void* src, size_t bytes) override {
        MTL::Buffer** slot = find(id);
        if (slot == nullptr || src == nullptr || bytes == 0) {
            return;
        }
        // Already one of ours? Then blit straight from it.
        //
        // The decode ring hands out the contents of a shared MTLBuffer this
        // backend made, so the bytes are already where the GPU can read them.
        // Copying them into a second shared buffer first would be 33 MB of
        // memcpy per frame to arrive where they already were.
        if (MTL::Buffer* source = hostBufferFor(src, bytes); source != nullptr) {
            MTL::CommandBuffer* direct = blitQueue_->commandBuffer();
            MTL::BlitCommandEncoder* encoder = direct->blitCommandEncoder();
            const size_t offset =
                static_cast<const unsigned char*>(src) -
                static_cast<const unsigned char*>(source->contents());
            encoder->copyFromBuffer(source, offset, *slot, 0, bytes);
            encoder->endEncoding();
            direct->encodeSignalEvent(uploadEvent_, ++uploadValue_);
            direct->commit();
            if (lastBlit_ != nullptr) {
                lastBlit_->release();
            }
            direct->retain();
            lastBlit_ = direct;
            uploadPending_ = true;
            return;
        }

        // Through shared memory the engine owns, on the blit queue.
        //
        // `src` belongs to the caller, who may destroy it the moment this
        // returns, so the bytes are taken now with a memcpy and the blit out of
        // the staging buffer is what overlaps. A staging buffer allocated per
        // call and released here would also have to be waited for, which is
        // what made this synchronous before.
        MTL::Buffer* staging = uploadStaging(bytes);
        if (staging == nullptr) {
            return;
        }
        std::memcpy(staging->contents(), src, bytes);

        MTL::CommandBuffer* commands = blitQueue_->commandBuffer();
        MTL::BlitCommandEncoder* blit = commands->blitCommandEncoder();
        blit->copyFromBuffer(staging, 0, *slot, 0, bytes);
        blit->endEncoding();
        // Signalled here, waited for by the next dispatch and by nothing else.
        commands->encodeSignalEvent(uploadEvent_, ++uploadValue_);
        commands->commit();
        if (lastBlit_ != nullptr) {
            lastBlit_->release();
        }
        commands->retain();
        lastBlit_ = commands;
        uploadPending_ = true;
    }

    void download(void* dst, BufferId id, size_t bytes) override {
        MTL::Buffer** slot = find(id);
        if (slot == nullptr || dst == nullptr || bytes == 0) {
            return;
        }
        MTL::Buffer* staging =
            device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (staging == nullptr) {
            return;
        }
        MTL::CommandBuffer* commands = queue_->commandBuffer();
        MTL::BlitCommandEncoder* blit = commands->blitCommandEncoder();
        blit->copyFromBuffer(*slot, 0, staging, 0, bytes);
        blit->endEncoding();
        commands->commit();
        // The one call in the interface that has to wait: the bytes must be
        // there when it returns.
        commands->waitUntilCompleted();
        std::memcpy(dst, staging->contents(), bytes);
        staging->release();
    }

    [[nodiscard]] KernelId load(std::string_view name) override {
        for (size_t i = 0; i < kernels_.size(); ++i) {
            if (kernels_[i].name == name) {
                return static_cast<KernelId>(i + 1);
            }
        }
        const kernels::Blob* blob = kernels::find(name);
        if (blob == nullptr) {
            std::fprintf(stderr, "gpe/metal: no kernel '%.*s' in this binary\n",
                         static_cast<int>(name.size()), name.data());
            return kInvalidKernel;
        }

        // dispatch_data_t over the embedded bytes, with a destructor that frees
        // nothing: the blob is in the binary's rodata and outlives everything.
        dispatch_data_t data = dispatch_data_create(
            blob->data, blob->size, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NS::Error*   error = nullptr;
        MTL::Library* library = device_->newLibrary(data, &error);
        dispatch_release(data);
        if (library == nullptr) {
            std::fprintf(stderr, "gpe/metal: newLibrary failed: %s\n",
                         describe(error).c_str());
            return kInvalidKernel;
        }

        const std::string entry(blob->entry);
        NS::String* entryName = NS::String::string(entry.c_str(),
                                                   NS::UTF8StringEncoding);
        MTL::Function* function = library->newFunction(entryName);
        if (function == nullptr) {
            std::fprintf(stderr, "gpe/metal: no function '%s' in the library\n",
                         entry.c_str());
            library->release();
            return kInvalidKernel;
        }
        MTL::ComputePipelineState* pipeline =
            device_->newComputePipelineState(function, &error);
        function->release();
        library->release();
        if (pipeline == nullptr) {
            std::fprintf(stderr, "gpe/metal: pipeline failed: %s\n",
                         describe(error).c_str());
            return kInvalidKernel;
        }
        kernels_.push_back(Kernel{std::string(name), pipeline});
        return static_cast<KernelId>(kernels_.size());
    }

    void dispatch(KernelId id, Grid grid, const void* args,
                  size_t bytes) override {
        if (id == kInvalidKernel || id > kernels_.size()) {
            return;
        }
        const Args::View view = Args::read(args, bytes);
        if (!view.valid) {
            std::fprintf(stderr, "gpe/metal: malformed dispatch args\n");
            return;
        }

        MTL::CommandBuffer* commands = queue_->commandBuffer();
        // Wait for whatever was uploaded since the last dispatch, and for
        // nothing else. Without it a kernel could read a buffer whose blit is
        // still running; with a full wait instead there would be no overlap
        // left to have.
        if (uploadPending_) {
            commands->encodeWait(uploadEvent_, uploadValue_);
            uploadPending_ = false;
        }
        MTL::ComputeCommandEncoder* encoder = commands->computeCommandEncoder();
        encoder->setComputePipelineState(kernels_[id - 1].pipeline);

        for (uint32_t i = 0; i < view.bufferCount; ++i) {
            MTL::Buffer** slot = find(view.buffers[i]);
            encoder->setBuffer(slot != nullptr ? *slot : nullptr, 0, i);
        }
        if (view.uniformBytes > 0) {
            // setBytes, not a buffer: Metal copies small blocks into the
            // command buffer itself, so there is nothing to allocate, nothing
            // to keep alive and no staging ring -- which is the whole of what
            // the CUDA side needed a pinned host ring for.
            encoder->setBytes(view.uniforms, view.uniformBytes,
                              view.bufferCount);
        }

        // Threads, not threadgroups. `thread_position_in_grid` means Metal will
        // launch a partial group at the edge rather than rounding up, so the
        // grid is exactly what was asked for.
        const MTL::Size threads = MTL::Size::Make(grid.x, grid.y, grid.z);
        const MTL::Size perGroup = MTL::Size::Make(kGroupX, kGroupY, 1);
        encoder->dispatchThreads(threads, perGroup);
        encoder->endEncoding();

        // Numbered before it is committed, so the handler reports the right
        // one. Metal runs handlers on its own thread; reportCompleted stores
        // and returns, which is all a handler is allowed to do.
        const uint64_t submission = ++submitted_;
        // The std::function overload, named explicitly: metal-cpp also takes an
        // Objective-C block and a lambda converts to either.
        const MTL::HandlerFunction handler =
            [this, submission](MTL::CommandBuffer* done) {
                // The device's own clock, read off the command buffer. Seconds
                // since an arbitrary epoch, so only the difference means
                // anything -- which is all that is wanted.
                const double seconds = done->GPUEndTime() - done->GPUStartTime();
                reportCompleted(submission, seconds * 1000.0);
            };
        commands->addCompletedHandler(handler);

        // Returns without waiting. `sync` is what waits, and the counter above
        // is what says which submissions have retired without waiting at all.
        commands->commit();
        commands->retain();
        if (previous_ != nullptr) {
            previous_->release();
        }
        previous_ = commands;
    }

    // --- HostStaging -------------------------------------------------------
    //
    // A shared MTLBuffer, handed out as its contents pointer. On unified memory
    // there is no separate host allocation to pin: a shared buffer already is
    // memory both sides can reach, which is the same thing this interface is
    // for on CUDA and arrived at from the other direction.

    [[nodiscard]] void* allocHost(size_t bytes) override {
        MTL::Buffer* buffer =
            device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (buffer == nullptr) {
            return nullptr;
        }
        void* contents = buffer->contents();
        hostBuffers_.push_back(HostBuffer{contents, buffer});
        return contents;
    }
    void freeHost(void* memory) override {
        for (auto it = hostBuffers_.begin(); it != hostBuffers_.end(); ++it) {
            if (it->contents == memory) {
                it->buffer->release();
                hostBuffers_.erase(it);
                return;
            }
        }
    }

    void sync() override {
        // Both queues: a caller asking for everything to be finished means the
        // transfers as well as the work.
        if (lastBlit_ != nullptr) {
            lastBlit_->waitUntilCompleted();
        }
        if (previous_ != nullptr) {
            previous_->waitUntilCompleted();
        }
    }

private:
    struct Staging {
        MTL::Buffer* buffer = nullptr;
    };

    struct HostBuffer {
        void*        contents = nullptr;
        MTL::Buffer* buffer = nullptr;
    };

    struct Kernel {
        std::string                name;
        MTL::ComputePipelineState* pipeline = nullptr;
    };

    /// Matches [numthreads(16, 16, 1)] in the kernels and the CUDA backend.
    static constexpr uint32_t kGroupX = 16;
    static constexpr uint32_t kGroupY = 16;
    /// Matches the three-frame pipeline, plus one so a slot is never reused
    /// while its blit is still running.
    static constexpr int kUploadSlots = 4;

    /// A shared staging buffer of at least `bytes`, from a small ring.
    ///
    /// Grown on demand and never shrunk. It grows during prepare(), when the
    /// first uploads set the size; every frame of a playback uploads the same
    /// shape and finds it already there.
    /// The shared buffer `src` lives in, if this backend made it.
    [[nodiscard]] MTL::Buffer* hostBufferFor(const void* src, size_t bytes) const {
        const auto* p = static_cast<const unsigned char*>(src);
        for (const HostBuffer& host : hostBuffers_) {
            const auto* start = static_cast<const unsigned char*>(host.contents);
            if (p >= start && p + bytes <= start + host.buffer->length()) {
                return host.buffer;
            }
        }
        return nullptr;
    }

    [[nodiscard]] MTL::Buffer* uploadStaging(size_t bytes) {
        Staging& slot = uploads_[uploadAt_];
        uploadAt_ = (uploadAt_ + 1) % kUploadSlots;
        if (slot.buffer != nullptr && slot.buffer->length() >= bytes) {
            return slot.buffer;
        }
        if (slot.buffer != nullptr) {
            // Prepare-time, so waiting is allowed -- and releasing a buffer a
            // blit is reading is not.
            if (lastBlit_ != nullptr) {
                lastBlit_->waitUntilCompleted();
            }
            slot.buffer->release();
        }
        slot.buffer = device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
        return slot.buffer;
    }

    [[nodiscard]] MTL::Buffer** find(BufferId id) {
        if (id == kInvalidBuffer || id > buffers_.size()) {
            return nullptr;
        }
        return &buffers_[id - 1];
    }

    [[nodiscard]] static std::string describe(NS::Error* error) {
        if (error == nullptr || error->localizedDescription() == nullptr) {
            return "no reason given";
        }
        return error->localizedDescription()->utf8String();
    }

    MTL::Device*       device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    MTL::CommandQueue*  blitQueue_ = nullptr;
    MTL::Event*         uploadEvent_ = nullptr;
    uint64_t            uploadValue_ = 0;
    bool                uploadPending_ = false;
    MTL::CommandBuffer* lastBlit_ = nullptr;
    Staging             uploads_[kUploadSlots]{};
    int                 uploadAt_ = 0;
    MTL::CommandBuffer* previous_ = nullptr;
    /// This backend's own count, which matches the pool's because both
    /// increment once per dispatch and nothing else.
    uint64_t            submitted_ = 0;

    std::vector<HostBuffer>   hostBuffers_;
    std::vector<MTL::Buffer*> buffers_;
    std::vector<Kernel>       kernels_;
};

}   // namespace

std::unique_ptr<Device> Device::create() {
    auto device = std::make_unique<MetalDevice>();
    if (!device->open()) {
        return nullptr;
    }
    return device;
}

}   // namespace gpe
