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
// metal-cpp's selector tables are defined exactly once per program, by
// whichever translation unit defines these. gpe is that unit on its own; linked
// beside another metal-cpp user that already defines them -- slang-rhi, in
// lucabRTrender -- two definitions are a link error, so the embedding build
// turns GPE_METAL_CPP_IMPLEMENTATION off and gpe uses the other one's.
#if !defined(GPE_NO_METAL_CPP_IMPLEMENTATION)
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#endif

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "completion.h"
#include "hostring.h"
#include "gpe/adopt.h"
#include "gpe/args.h"
#include "gpe/device.h"
#include "kernel_registry.h"

namespace gpe {
namespace {

/// An autorelease pool for the length of one call.
///
/// `commandBuffer()`, `blitCommandEncoder()`, `computeCommandEncoder()` and
/// `NS::String::string()` all return *autoreleased* objects. In a Cocoa
/// application the run loop drains them once a turn and nobody thinks about
/// it; this engine's caller is a render loop that never enters one, so without
/// a pool of its own every frame leaves a command buffer, two encoders and
/// their attachments behind -- for as long as the application runs.
///
/// One pool per call rather than one per frame: the engine has no notion of a
/// frame, and a call is the largest scope it can be sure has ended.
class ScopedPool {
public:
    ScopedPool() : pool_(NS::AutoreleasePool::alloc()->init()) {}
    ~ScopedPool() { pool_->release(); }

    ScopedPool(const ScopedPool&) = delete;
    ScopedPool& operator=(const ScopedPool&) = delete;

private:
    NS::AutoreleasePool* pool_;
};

class MetalDevice final : public Device,
                         public CompletionReporting,
                         public HostStaging {
public:
    ~MetalDevice() override {
        sync();
        if (previous_ != nullptr) {
            previous_->release();
        }
        if (lastBlit_ != nullptr) {
            lastBlit_->release();
        }
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
            if (slot.blit != nullptr) {
                slot.blit->waitUntilCompleted();
                slot.blit->release();
            }
            if (slot.buffer != nullptr) {
                slot.buffer->release();
            }
        }
        if (readback_ != nullptr) {
            readback_->release();
        }
        for (MTL::Buffer* spare : spareStaging_) {
            spare->release();
        }
        if (uploadEvent_ != nullptr) {
            uploadEvent_->release();
        }
        if (computeEvent_ != nullptr) {
            computeEvent_->release();
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

    /// Takes a device somebody else made, retaining it for as long as this
    /// object lives. The caller keeps its own reference: it is still drawing
    /// with it.
    [[nodiscard]] bool adopt(MTL::Device* device, MTL::CommandQueue* queue = nullptr) {
        if (device == nullptr) {
            return false;
        }
        device_ = device->retain();
        // The renderer's queue when it offers one: Metal orders command buffers
        // on a queue by commit, so its work and ours need nothing between them.
        queue_ = queue != nullptr ? queue->retain() : device_->newCommandQueue();
        blitQueue_ = device_->newCommandQueue();
        uploadEvent_ = device_->newEvent();
        computeEvent_ = device_->newEvent();
        return queue_ != nullptr && blitQueue_ != nullptr &&
               uploadEvent_ != nullptr && computeEvent_ != nullptr;
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
        computeEvent_ = device_->newEvent();
        return queue_ != nullptr && blitQueue_ != nullptr &&
               uploadEvent_ != nullptr && computeEvent_ != nullptr;
    }

    [[nodiscard]] Backend backend() const override { return Backend::Metal; }

    [[nodiscard]] uint64_t backendBuffer(BufferId id) const override {
        MTL::Buffer* const* slot = const_cast<MetalDevice*>(this)->find(id);
        return slot != nullptr ? reinterpret_cast<uint64_t>(*slot) : 0;
    }
    [[nodiscard]] uint64_t backendDevice() const override {
        return reinterpret_cast<uint64_t>(device_);
    }
    [[nodiscard]] uint64_t backendQueue() const override {
        return reinterpret_cast<uint64_t>(queue_);
    }

    /// An `MTL::Buffer*` somebody else made on this device, retained for as
    /// long as the handle lives. Metal has had the notion all along -- a buffer
    /// is an object, and taking a reference to one is what a renderer sharing
    /// this device hands over -- and the backend answered "no" only because
    /// the first foreign memory it met was a CUDA pointer.
    [[nodiscard]] BufferId adopt(uint64_t buffer, size_t bytes) override {
        auto* foreign = reinterpret_cast<MTL::Buffer*>(buffer);
        if (foreign == nullptr || bytes == 0 || foreign->length() < bytes ||
            foreign->device() != device_) {
            return kInvalidBuffer;
        }
        buffers_.push_back(foreign->retain());
        return static_cast<BufferId>(buffers_.size());
    }

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
        const ScopedPool drain;
        MTL::Buffer** slot = find(id);
        // `*slot` as well as `slot`, which release and download both check and
        // this did not. A released buffer leaves a live slot holding null, and
        // handing null to the blit encoder as a destination is a segmentation
        // fault inside the driver -- a long way from whoever kept the handle
        // too long, and with nothing on the stack to say so.
        if (slot == nullptr || *slot == nullptr || src == nullptr || bytes == 0) {
            std::fprintf(stderr,
                         "gpe/metal: upload to buffer %llu, which is not live\n",
                         static_cast<unsigned long long>(id));
            return;
        }
        // Already one of ours? Then blit straight from it.
        //
        // The decode ring hands out the contents of a shared MTLBuffer this
        // backend made, so the bytes are already where the GPU can read them.
        // Copying them into a second shared buffer first would be 33 MB of
        // memcpy per frame to arrive where they already were.
        if (MTL::Buffer* source = hostBufferFor(src, bytes); source != nullptr) {
            flushBatch();
            MTL::CommandBuffer* direct = blitQueue_->commandBuffer();
            awaitCompute(direct);
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
        int          at = 0;
        MTL::Buffer* staging = uploadStaging(bytes, at);
        if (staging == nullptr) {
            return;
        }
        std::memcpy(staging->contents(), src, bytes);

        flushBatch();
        MTL::CommandBuffer* commands = blitQueue_->commandBuffer();
        awaitCompute(commands);
        MTL::BlitCommandEncoder* blit = commands->blitCommandEncoder();
        blit->copyFromBuffer(staging, 0, *slot, 0, bytes);
        blit->endEncoding();
        // Signalled here, waited for by the next dispatch and by nothing else.
        commands->encodeSignalEvent(uploadEvent_, ++uploadValue_);
        commands->commit();
        // Kept on the slot, so the next lap round the ring knows whether this
        // blit has read the bytes yet. See uploadStaging.
        if (uploads_[at].blit != nullptr) {
            uploads_[at].blit->release();
        }
        commands->retain();
        uploads_[at].blit = commands;

        if (lastBlit_ != nullptr) {
            lastBlit_->release();
        }
        commands->retain();
        lastBlit_ = commands;
        uploadPending_ = true;
    }

    void download(void* dst, BufferId id, size_t bytes) override {
        const ScopedPool drain;
        MTL::Buffer** slot = find(id);
        // `*slot`, which upload's comment twenty lines above already said this
        // one checked, and it did not. A released buffer leaves a live slot
        // holding null, and null as a blit *source* is the same segmentation
        // fault inside the driver that upload describes.
        if (slot == nullptr || *slot == nullptr || dst == nullptr ||
            bytes == 0) {
            return;
        }
        // Reused, not allocated per call. At 4K a display readback is about
        // 8 MB, and this is the one call the display pass makes every frame --
        // an allocation and a free per frame in the backend a pool exists to
        // keep away from the driver.
        //
        // Safe to reuse because this call waits: by the time it returns, the
        // blit that read this buffer has finished.
        if (readback_ == nullptr || readback_->length() < bytes) {
            if (readback_ != nullptr) {
                readback_->release();
            }
            readback_ = device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
        }
        if (readback_ == nullptr) {
            return;
        }
        flushBatch();
        MTL::CommandBuffer* commands = queue_->commandBuffer();
        // Ordered against the transfers, exactly as a dispatch is.
        //
        // The uploads go to `blitQueue_` and this reads on `queue_`: two
        // queues, so without the wait a download issued right after an upload
        // -- with no dispatch between them to carry the wait -- can overtake it
        // and return the previous contents. The CUDA side handles the same case
        // by waiting on its copy event, and says so.
        if (uploadPending_) {
            commands->encodeWait(uploadEvent_, uploadValue_);
            uploadPending_ = false;
        }
        MTL::BlitCommandEncoder* blit = commands->blitCommandEncoder();
        blit->copyFromBuffer(*slot, 0, readback_, 0, bytes);
        blit->endEncoding();
        commands->commit();
        // The one call in the interface that has to wait: the bytes must be
        // there when it returns.
        commands->waitUntilCompleted();
        std::memcpy(dst, readback_->contents(), bytes);
    }

    /// Writes `byte` over the whole range, on the device.
    ///
    /// Metal has had this since the beginning -- `fillBuffer` on a blit
    /// encoder -- and the backend simply never offered it, so every "start
    /// transparent black" on this machine fell back to zeroing host memory and
    /// sending it across, which for a 4K plate is 135 MB per image that never
    /// needed to move.
    ///
    /// On the COMPUTE queue, not the transfer one.
    ///
    /// The same rule as the CUDA side, and learned the same way it warns about:
    /// "a memset that raced ahead of a kernel still writing the buffer would
    /// clear the picture it had just made, and only sometimes". Put on the blit
    /// queue this filled buffers that dispatches on the other queue were in the
    /// middle of writing, and the symptom was a node thumbnail that came out
    /// black -- the render was right, the clear landed after it.
    ///
    /// Queued on `queue_` it is ordered against every kernel by the queue
    /// itself, which is what the buffer being cleared is about to be used by.
    /// Nothing to signal and nothing to wait for.
    [[nodiscard]] bool fill(BufferId id, uint8_t byte, size_t bytes) override {
        const ScopedPool drain;
        MTL::Buffer** slot = find(id);
        if (slot == nullptr || *slot == nullptr || bytes == 0) {
            return false;
        }
        if (bytes > (*slot)->length()) {
            return false;
        }
        flushBatch();
        MTL::CommandBuffer* commands = queue_->commandBuffer();
        MTL::BlitCommandEncoder* blit = commands->blitCommandEncoder();
        blit->fillBuffer(*slot, NS::Range::Make(0, bytes), byte);
        blit->endEncoding();
        noteCompute(commands);
        commands->commit();
        // Kept as the last thing on the compute queue, so `sync()` waits for
        // it like it waits for a dispatch.
        commands->retain();
        if (previous_ != nullptr) {
            previous_->release();
        }
        previous_ = commands;
        return true;
    }

    /// What the device has, and what is left.
    ///
    /// `recommendedMaxWorkingSetSize` rather than physical memory: on unified
    /// memory the GPU shares the machine's RAM and the physical figure would
    /// invite a host to budget the whole of it. Apple's recommendation is the
    /// working set it will let this process hold before it starts paging, which
    /// is the number a budget wants.
    void memory(size_t& total, size_t& available) const override {
        total = static_cast<size_t>(device_->recommendedMaxWorkingSetSize());
        const size_t used = static_cast<size_t>(device_->currentAllocatedSize());
        available = used < total ? total - used : 0;
    }

    [[nodiscard]] KernelId load(std::string_view name) override {
        const ScopedPool drain;   // NS::String::string is autoreleased
        for (size_t i = 0; i < kernels_.size(); ++i) {
            if (kernels_[i].name == name) {
                return static_cast<KernelId>(i + 1);
            }
        }
        // Through the registry, so a kernel a plugin brought with it is
        // found the same way one compiled into this library is -- and with any
        // trailer already taken off: newLibrary refuses bytes after the end
        // of a metallib.
        const kernels::Resolved* resolved = kernels::resolve(name);
        const kernels::Blob* blob = resolved != nullptr ? &resolved->blob : nullptr;
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
        kernels_.push_back(Kernel{std::string(name), pipeline, resolved->info});
        return static_cast<KernelId>(kernels_.size());
    }

    void dispatch(KernelId id, Grid grid, const void* args,
                  size_t bytes) override {
        const ScopedPool drain;
        if (id == kInvalidKernel || id > kernels_.size()) {
            return;
        }
        const Args::View view = Args::read(args, bytes);
        if (!view.valid) {
            std::fprintf(stderr, "gpe/metal: malformed dispatch args\n");
            return;
        }
        const Kernel& kernel = kernels_[id - 1];
        // What the kernel's reflection says, when its blob carried it. A
        // dispatch handed the wrong number of buffers used to bind whatever it
        // was given to the first slots and render something (openFXplayer
        // D30); now it is refused, with the kernel's name.
        if (kernel.info.has_value()) {
            if (kernel.info->elementBytes.size() != view.bufferCount) {
                std::fprintf(stderr,
                             "gpe/metal: %s declares %zu buffers and was handed %u\n",
                             kernel.name.c_str(), kernel.info->elementBytes.size(),
                             view.bufferCount);
                return;
            }
            if (kernel.info->uniformBytes != view.uniformBytes) {
                std::fprintf(stderr,
                             "gpe/metal: %s declares %u bytes of uniforms and was handed %u\n",
                             kernel.name.c_str(), kernel.info->uniformBytes,
                             view.uniformBytes);
                return;
            }
        }

        // Uploads issued since the batch opened must land before this
        // kernel, and a wait can only be encoded before a command buffer's
        // first encoder: so the batch that cannot carry it goes now.
        if (batch_ != nullptr && uploadPending_) {
            flushBatch();
        }
        openBatch();
        MTL::ComputeCommandEncoder* encoder = batchEncoder_;
        encoder->setComputePipelineState(kernel.pipeline);

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
        if (kernel.info.has_value()) {
            // Whole groups of the kernel's own shape, as D3D, Vulkan and now
            // the CUDA backend launch them. Clipping the partial group at the
            // edge -- dispatchThreads -- is what left a 1080-high picture's
            // last row of tiles with half a group while a kernel sharing group
            // memory counted on all of it, and a [numthreads(256,1,1)]
            // reduction run as one thread here and 256 on CUDA. Every kernel
            // bounds-checks; that rule is what makes whole groups safe.
            const auto groups = [](uint32_t threads, uint32_t size) {
                return (threads + size - 1) / size;
            };
            const auto& group = kernel.info->threadGroup;
            encoder->dispatchThreadgroups(
                MTL::Size::Make(groups(grid.x, group[0]), groups(grid.y, group[1]),
                                groups(grid.z, group[2])),
                MTL::Size::Make(group[0], group[1], group[2]));
        } else {
            const MTL::Size threads = MTL::Size::Make(grid.x, grid.y, grid.z);
            const MTL::Size perGroup = MTL::Size::Make(kGroupX, kGroupY, 1);
            encoder->dispatchThreads(threads, perGroup);
        }
        // Numbered now, reported when the batch it is in completes: the
        // pool's rule is "everything up to N", which a batch satisfies for
        // its last number.
        batchLast_ = ++submitted_;
        if (++batched_ >= kBatchLimit) {
            flushBatch();
        }
    }

    void flush() override {
        const ScopedPool drain;
        flushBatch();
    }

    // --- HostStaging -------------------------------------------------------
    //
    // A shared MTLBuffer, handed out as its contents pointer. On unified memory
    // there is no separate host allocation to pin: a shared buffer already is
    // memory both sides can reach, which is the same thing this interface is
    // for on CUDA and arrived at from the other direction.

    [[nodiscard]] void* allocShared(size_t bytes, BufferId& out) override {
        // On Apple silicon a Shared buffer is one allocation the CPU writes and
        // the GPU reads, with no copy between them and no transfer to schedule.
        // It goes into the same table as every other buffer, so a dispatch
        // cannot tell the difference -- which is the point.
        MTL::Buffer* buffer =
            device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (buffer == nullptr) {
            out = kInvalidBuffer;
            return nullptr;
        }
        // Into `buffers_` only. It is a device buffer that happens to be
        // host-visible, and its lifetime belongs to the handle -- putting it in
        // `hostBuffers_` as well gave two owners and two releases, which
        // arrives as objc_msgSend on freed memory long after the mistake.
        buffers_.push_back(buffer);
        out = static_cast<BufferId>(buffers_.size());
        return buffer->contents();
    }

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

    /// The download that does not wait: a blit into a shared buffer of its
    /// own, and the bytes copied out by the completion handler.
    ///
    /// Its own staging buffer rather than `readback_`, which the synchronous
    /// download reuses on the promise that it has finished before returning --
    /// a promise this call exists not to make.
    void downloadAsync(BufferId id, void* dst, size_t bytes,
                       std::function<void(bool)> done) override {
        const ScopedPool drain;
        MTL::Buffer** slot = find(id);
        if (slot == nullptr || *slot == nullptr || dst == nullptr || bytes == 0) {
            if (done) {
                done(false);
            }
            return;
        }
        MTL::Buffer* staging = takeReadbackStaging(bytes);
        if (staging == nullptr) {
            if (done) {
                done(false);
            }
            return;
        }
        flushBatch();
        MTL::CommandBuffer* commands = queue_->commandBuffer();
        if (uploadPending_) {
            commands->encodeWait(uploadEvent_, uploadValue_);
            uploadPending_ = false;
        }
        MTL::BlitCommandEncoder* blit = commands->blitCommandEncoder();
        blit->copyFromBuffer(*slot, 0, staging, 0, bytes);
        blit->endEncoding();
        // A copy of the callable on the heap, captured by pointer: a block that
        // captured a std::function by value would copy it through byref storage
        // on two threads, the race the dispatch handler below already avoids.
        auto* callback = new std::function<void(bool)>(std::move(done));
        commands->addCompletedHandler(^(MTL::CommandBuffer* finished) {
            const bool fine = finished->status() == MTL::CommandBufferStatusCompleted;
            if (fine) {
                std::memcpy(dst, staging->contents(), bytes);
            }
            returnReadbackStaging(staging);
            if (*callback) {
                (*callback)(fine);
            }
            delete callback;
        });
        commands->commit();
        commands->retain();
        if (previous_ != nullptr) {
            previous_->release();
        }
        previous_ = commands;
    }

    void sync() override {
        {
            const ScopedPool drain;
            flushBatch();
        }
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
        /// The blit that last read this slot, retained until it has run.
        MTL::CommandBuffer* blit = nullptr;
    };

    struct HostBuffer {
        void*        contents = nullptr;
        MTL::Buffer* buffer = nullptr;
    };

    struct Kernel {
        std::string                name;
        MTL::ComputePipelineState* pipeline = nullptr;
        /// The blob's trailer, if it had one. See gpe/kernels.h.
        std::optional<KernelInfo>  info;
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

    [[nodiscard]] MTL::Buffer* uploadStaging(size_t bytes, int& at) {
        at = uploadAt_;
        Staging& slot = uploads_[uploadAt_];
        uploadAt_ = (uploadAt_ + 1) % kUploadSlots;

        // The blit that last read this slot has to have finished before the
        // caller memcpys over it.
        //
        // Without this the ring was a promise and not a mechanism: a slot comes
        // round again after kUploadSlots uploads, and with a three-frame
        // pipeline and more than one upload per frame it comes round while its
        // blit is still queued -- so the memcpy overwrote bytes a DMA had not
        // read yet, and the picture that reached the device was half of one
        // frame and half of another. Load-dependent, size-dependent, and
        // invisible in anything short of playback.
        //
        // The CUDA backend hit exactly this and answered it with an event per
        // slot; this is the same answer in Metal's vocabulary. It costs
        // nothing in the steady state, where a slot's blit finished two
        // uploads ago and the wait returns at once.
        if (slot.blit != nullptr) {
            slot.blit->waitUntilCompleted();
            slot.blit->release();
            slot.blit = nullptr;
        }

        if (slot.buffer != nullptr && slot.buffer->length() >= bytes) {
            return slot.buffer;
        }
        if (slot.buffer != nullptr) {
            slot.buffer->release();
        }
        slot.buffer = device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
        return slot.buffer;
    }

    /// Staging for downloadAsync, kept rather than made per call: a display
    /// that reads back every frame asked the driver for a new shared buffer
    /// sixty times a second. Taken on this thread, returned from Metal's
    /// completion thread, hence the lock.
    MTL::Buffer* takeReadbackStaging(size_t bytes) {
        {
            const std::lock_guard<std::mutex> held(stagingGuard_);
            for (auto it = spareStaging_.begin(); it != spareStaging_.end(); ++it) {
                if ((*it)->length() >= bytes) {
                    MTL::Buffer* found = *it;
                    spareStaging_.erase(it);
                    return found;
                }
            }
        }
        return device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
    }

    void returnReadbackStaging(MTL::Buffer* staging) {
        const std::lock_guard<std::mutex> held(stagingGuard_);
        if (spareStaging_.size() < kSpareStaging) {
            spareStaging_.push_back(staging);
        } else {
            staging->release();
        }
    }

    /// BATCHING
    ///
    /// Dispatches are encoded into one open command buffer and committed
    /// together, not one command buffer each. Measured on an M5 Pro, 2000
    /// small dispatches: a command buffer, encoder, completion block and
    /// commit apiece cost 12-15 us of CPU per dispatch, nearly all of the
    /// time; batched, about 1 us to queue and 3.5 us including the device's
    /// work (tests/test_dispatch_order.cpp).
    ///
    /// The batch goes -- is committed, never waited for -- at every point
    /// something must run after what it holds: an upload (which orders itself
    /// behind compute), a download, a fill, sync(), flush(), a dispatch that
    /// must wait for an upload, and every kBatchLimit dispatches so that work
    /// keeps flowing to the device inside a long frame. A host that shares
    /// this queue with another runtime calls flush() before that runtime
    /// submits work reading gpe's results.
    void openBatch() {
        if (batch_ != nullptr) {
            return;
        }
        batch_ = queue_->commandBuffer()->retain();
        // Wait for whatever was uploaded since the last batch, and for nothing
        // else. Without it a kernel could read a buffer whose blit is still
        // running; with a full wait instead there would be no overlap left.
        if (uploadPending_) {
            batch_->encodeWait(uploadEvent_, uploadValue_);
            uploadPending_ = false;
        }
        batchEncoder_ = batch_->computeCommandEncoder()->retain();
    }

    void flushBatch() {
        if (batch_ == nullptr) {
            return;
        }
        batchEncoder_->endEncoding();
        batchEncoder_->release();
        batchEncoder_ = nullptr;
        noteCompute(batch_);
        const uint64_t submission = batchLast_;
        // The **block** overload, not the std::function one.
        //
        // metal-cpp's std::function overload copies it into a `__block`
        // variable and hands Metal a block that reads it back. The copy
        // helper writes that byref storage on this thread and Metal's own
        // thread reads it later, and nothing in between is visible to a race
        // detector -- so every dispatch produced a ThreadSanitizer report
        // pointing into MTLCommandBuffer.hpp. A block captures `this` and the
        // number by value directly: no byref storage, no std::function.
        batch_->addCompletedHandler(^(MTL::CommandBuffer* done) {
            // The device's own clock, read off the command buffer: the whole
            // batch, which is what a frame's dispatches are.
            const double seconds = done->GPUEndTime() - done->GPUStartTime();
            reportCompleted(submission, seconds * 1000.0);
        });
        // Returns without waiting. `sync` is what waits.
        batch_->commit();
        if (previous_ != nullptr) {
            previous_->release();
        }
        previous_ = batch_;   // already retained
        batch_ = nullptr;
        batched_ = 0;
    }

    /// Remembers that the compute queue has work an upload must not overtake.
    void noteCompute(MTL::CommandBuffer* commands) {
        commands->encodeSignalEvent(computeEvent_, ++computeValue_);
        computePending_ = true;
    }

    /// Orders a transfer behind that work, once.
    ///
    /// The mirror of the wait in `dispatch`, and it is not hypothetical: a
    /// host that clears an image and then uploads its pixels puts a fill on
    /// the compute queue and a blit on the transfer queue, and with nothing
    /// between them the fill can land last and wipe the picture that just
    /// arrived. The CUDA backend was caught doing exactly this -- five runs in
    /// ten, reading as an effect that sometimes did nothing, because the kernel
    /// then blurred the black correctly.
    ///
    /// Skipped when the compute queue has nothing outstanding, which is the
    /// common case during a prepare, so the overlap these two queues exist for
    /// is kept everywhere it is safe to have.
    void awaitCompute(MTL::CommandBuffer* commands) {
        if (computePending_) {
            commands->encodeWait(computeEvent_, computeValue_);
            computePending_ = false;
        }
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
    /// The other direction. See awaitCompute.
    MTL::Event*         computeEvent_ = nullptr;
    uint64_t            computeValue_ = 0;
    bool                computePending_ = false;
    MTL::CommandBuffer* lastBlit_ = nullptr;
    /// The readback buffer, grown on demand and kept. See download.
    MTL::Buffer*        readback_ = nullptr;
    Staging             uploads_[kUploadSlots]{};
    int                 uploadAt_ = 0;
    MTL::CommandBuffer* previous_ = nullptr;
    /// The open batch. See openBatch.
    MTL::CommandBuffer*         batch_ = nullptr;
    MTL::ComputeCommandEncoder* batchEncoder_ = nullptr;
    uint32_t                    batched_ = 0;
    uint64_t                    batchLast_ = 0;
    static constexpr uint32_t   kBatchLimit = 64;
    /// Spare downloadAsync staging buffers. See takeReadbackStaging.
    std::mutex                  stagingGuard_;
    std::vector<MTL::Buffer*>   spareStaging_;
    static constexpr size_t     kSpareStaging = 4;
    /// This backend's own count, which matches the pool's because both
    /// increment once per dispatch and nothing else.
    uint64_t            submitted_ = 0;

    std::vector<HostBuffer>   hostBuffers_;
    std::vector<MTL::Buffer*> buffers_;
    std::vector<Kernel>       kernels_;
};

}   // namespace

std::unique_ptr<Device> adoptMetalDevice(void* mtlDevice) {
    auto device = std::make_unique<MetalDevice>();
    if (!device->adopt(static_cast<MTL::Device*>(mtlDevice))) {
        return nullptr;
    }
    return device;
}

std::unique_ptr<Device> adoptMetalDevice(void* mtlDevice, void* mtlCommandQueue) {
    auto device = std::make_unique<MetalDevice>();
    if (!device->adopt(static_cast<MTL::Device*>(mtlDevice),
                       static_cast<MTL::CommandQueue*>(mtlCommandQueue))) {
        return nullptr;
    }
    return device;
}

// No CUDA on this backend. Defined so a client that asks links and is told.
std::unique_ptr<Device> adoptCudaContext(void*, void*) { return nullptr; }

std::unique_ptr<Device> Device::create() {
    auto device = std::make_unique<MetalDevice>();
    if (!device->open()) {
        return nullptr;
    }
    return device;
}

}   // namespace gpe
