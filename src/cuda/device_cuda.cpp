// Copyright (c) 2026 gpe contributors.
//
// The CUDA backend. Nothing here knows the pool exists.
//
// The driver API rather than the runtime API, for one reason: the kernels are
// PTX in a blob, and loading a module from memory is cuModuleLoadData. The
// runtime API's <<<>>> launch needs the kernel to have been compiled into this
// binary by nvcc, which is exactly what embedding a blob is instead of.
//
// WHAT SLANG'S OUTPUT ASKS FOR
//
// Read off the generated .cu rather than assumed:
//
//     template<typename T> struct StructuredBuffer { T* data; size_t count; };
//     struct GlobalParams_0 {
//         StructuredBuffer<float4>   src_0;      // 16 bytes
//         RWStructuredBuffer<float4> dst_0;      // 16 bytes
//         ScaleParams_0*             params_0;   //  8 bytes, a POINTER
//     };
//     extern "C" __constant__ GlobalParams_0 SLANG_globalParams;
//
// So every global -- buffers and uniforms alike -- goes into one __constant__
// symbol, and the uniforms are reached through a pointer, which means they have
// to be in device memory of their own. Both are done here, per dispatch, out of
// a small ring of staging buffers so that a dispatch never allocates.
#include <cuda.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "completion.h"
#include "gpe/args.h"
#include "gpe/device.h"
#include "gpe_kernels.h"

namespace gpe {
namespace {

/// Slang's StructuredBuffer<T>, host side. The layout is the contract with the
/// generated code and the static_asserts below are what keep it honest.
struct SlangBuffer {
    CUdeviceptr data = 0;
    uint64_t    count = 0;
};
static_assert(sizeof(SlangBuffer) == 16);

bool ok(CUresult result, const char* what) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "gpe/cuda: %s failed: %s\n", what,
                 name != nullptr ? name : "?");
    return false;
}

class CudaDevice final : public Device, public CompletionReporting {
public:
    ~CudaDevice() override {
        if (context_ != nullptr) {
            cuCtxSetCurrent(context_);
            for (const CUdeviceptr staging : staging_) {
                cuMemFree(staging);
            }
            for (void* const pinned : pinned_) {
                cuMemFreeHost(pinned);
            }
            for (const Timing& timing : timings_) {
                if (timing.start != nullptr) {
                    cuEventDestroy(timing.start);
                }
                if (timing.stop != nullptr) {
                    cuEventDestroy(timing.stop);
                }
            }
            for (const CUmodule module : modules_) {
                cuModuleUnload(module);
            }
            cuDevicePrimaryCtxRelease(device_);
        }
    }

    [[nodiscard]] bool open() {
        if (!ok(cuInit(0), "cuInit")) {
            return false;
        }
        int count = 0;
        if (!ok(cuDeviceGetCount(&count), "cuDeviceGetCount") || count == 0) {
            return false;
        }
        if (!ok(cuDeviceGet(&device_, 0), "cuDeviceGet") ||
            !ok(cuDevicePrimaryCtxRetain(&context_, device_), "ctxRetain") ||
            !ok(cuCtxSetCurrent(context_), "ctxSetCurrent")) {
            return false;
        }
        // A stream of our own, not the default one. The default stream
        // synchronises against every other stream in the process, which would
        // make "asynchronous by default" a promise the first library we link
        // against could break.
        if (!ok(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "streamCreate")) {
            return false;
        }
        // Uniforms live in device memory because the generated code reaches
        // them through a pointer. A small ring, taken once: a dispatch that
        // allocated would be a dispatch that synchronises.
        for (CUdeviceptr& staging : staging_) {
            if (!ok(cuMemAlloc(&staging, kStagingBytes), "cuMemAlloc(staging)")) {
                return false;
            }
        }
        for (Timing& timing : timings_) {
            if (!ok(cuEventCreate(&timing.start, CU_EVENT_DEFAULT), "eventCreate") ||
                !ok(cuEventCreate(&timing.stop, CU_EVENT_DEFAULT), "eventCreate")) {
                return false;
            }
        }
        // And a pinned host ring to copy them *from*.
        //
        // This is not an optimisation, it is the fix for a real bug: an async
        // copy reads its host source when the stream reaches it, not when the
        // call returns, so copying from a std::vector inside dispatch() reads a
        // buffer that has already been destroyed. What arrived in __constant__
        // was whatever the stack held next, and the kernel dereferenced it --
        // CUDA_ERROR_ILLEGAL_ADDRESS, every pixel wrong. Pinned memory the
        // device owns outlives the call and does not get paged either.
        for (void*& pinned : pinned_) {
            if (!ok(cuMemAllocHost(&pinned, kStagingBytes), "cuMemAllocHost")) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] Backend backend() const override { return Backend::CUDA; }

    [[nodiscard]] BufferId alloc(size_t bytes) override {
        CUdeviceptr ptr = 0;
        if (!ok(cuMemAlloc(&ptr, bytes), "cuMemAlloc")) {
            return kInvalidBuffer;
        }
        // The handle is an index into a table rather than the pointer itself:
        // a CUdeviceptr is 64 bits and so is BufferId, but zero has to mean
        // invalid and a device pointer of zero is a legal-looking value.
        buffers_.push_back(Allocation{ptr, bytes});
        return static_cast<BufferId>(buffers_.size());   // 1-based
    }

    void release(BufferId id) override {
        if (Allocation* a = find(id); a != nullptr && a->ptr != 0) {
            cuMemFree(a->ptr);
            a->ptr = 0;
            a->bytes = 0;
        }
    }

    void upload(BufferId id, const void* src, size_t bytes) override {
        const Allocation* a = find(id);
        if (a == nullptr || src == nullptr) {
            return;
        }
        // Synchronous, and deliberately.
        //
        // An async copy would read `src` when the stream got to it, and `src`
        // belongs to the caller -- who has every right to destroy it the moment
        // this returns. Overlapping uploads with compute is worth doing and is
        // its own piece of work: it needs the caller's bytes in pinned memory
        // the engine owns, which is a change to how a client hands over an
        // image, not a change to this line.
        (void)ok(cuMemcpyHtoD(a->ptr, src, bytes), "HtoD");
    }

    void download(void* dst, BufferId id, size_t bytes) override {
        const Allocation* a = find(id);
        if (a == nullptr || dst == nullptr) {
            return;
        }
        // The one synchronous call in the interface: it has to be, because the
        // bytes must be there when it returns.
        (void)ok(cuMemcpyDtoH(dst, a->ptr, bytes), "DtoH");
    }

    [[nodiscard]] KernelId load(std::string_view name) override {
        for (size_t i = 0; i < kernels_.size(); ++i) {
            if (kernels_[i].name == name) {
                return static_cast<KernelId>(i + 1);
            }
        }
        const kernels::Blob* blob = kernels::find(name);
        if (blob == nullptr) {
            std::fprintf(stderr, "gpe/cuda: no kernel '%.*s' in this binary\n",
                         static_cast<int>(name.size()), name.data());
            return kInvalidKernel;
        }
        CUmodule module = nullptr;
        // The PTX is text and the blob is not terminated, so it is copied into
        // a string first. Once per kernel, ever.
        const std::string ptx(reinterpret_cast<const char*>(blob->data),
                              blob->size);
        if (!ok(cuModuleLoadData(&module, ptx.c_str()), "cuModuleLoadData")) {
            return kInvalidKernel;
        }
        CUfunction function = nullptr;
        const std::string entry(blob->entry);
        if (!ok(cuModuleGetFunction(&function, module, entry.c_str()),
                "cuModuleGetFunction")) {
            cuModuleUnload(module);
            return kInvalidKernel;
        }
        CUdeviceptr globals = 0;
        size_t globalsBytes = 0;
        if (!ok(cuModuleGetGlobal(&globals, &globalsBytes, module,
                                  "SLANG_globalParams"),
                "cuModuleGetGlobal")) {
            cuModuleUnload(module);
            return kInvalidKernel;
        }
        modules_.push_back(module);
        kernels_.push_back(Kernel{std::string(name), function, globals,
                                  globalsBytes});
        return static_cast<KernelId>(kernels_.size());
    }

    void dispatch(KernelId id, Grid grid, const void* args,
                  size_t bytes) override {
        if (id == kInvalidKernel || id > kernels_.size()) {
            return;
        }
        const Kernel& kernel = kernels_[id - 1];
        const Args::View view = Args::read(args, bytes);
        if (!view.valid) {
            std::fprintf(stderr, "gpe/cuda: malformed dispatch args\n");
            return;
        }

        // The uniforms go to device memory first, because the generated struct
        // holds a pointer to them and not a copy.
        CUdeviceptr uniforms = 0;
        if (view.uniformBytes > 0) {
            if (view.uniformBytes > kStagingBytes) {
                std::fprintf(stderr, "gpe/cuda: %u bytes of uniforms is too many\n",
                             view.uniformBytes);
                return;
            }
            const int slot = stagingAt_;
            stagingAt_ = (stagingAt_ + 1) % kStagingSlots;
            uniforms = staging_[slot];
            std::memcpy(pinned_[slot], view.uniforms, view.uniformBytes);
            if (!ok(cuMemcpyHtoDAsync(uniforms, pinned_[slot], view.uniformBytes,
                                      stream_),
                    "uniforms HtoDAsync")) {
                return;
            }
        }

        // Then the globals struct: every buffer as {pointer, count}, in the
        // order the kernel declares them, with the uniform pointer last.
        std::vector<unsigned char> globals;
        globals.reserve(kernel.globalsBytes);
        for (uint32_t i = 0; i < view.bufferCount; ++i) {
            const Allocation* a = find(view.buffers[i]);
            SlangBuffer entry{};
            if (a != nullptr) {
                entry.data = a->ptr;
                entry.count = a->bytes / kBytesPerPixel;
            }
            appendBytes(globals, &entry, sizeof(entry));
        }
        appendBytes(globals, &uniforms, sizeof(uniforms));
        if (globals.size() != kernel.globalsBytes) {
            // The kernel's parameter block is not the shape this dispatch
            // describes. Refusing beats writing the wrong number of bytes into
            // __constant__ memory, which corrupts the next kernel too.
            std::fprintf(stderr,
                         "gpe/cuda: kernel wants %zu bytes of globals, got %zu\n",
                         kernel.globalsBytes, globals.size());
            return;
        }
        const int slot = stagingAt_;
        stagingAt_ = (stagingAt_ + 1) % kStagingSlots;
        std::memcpy(pinned_[slot], globals.data(), globals.size());
        if (!ok(cuMemcpyHtoDAsync(kernel.globals, pinned_[slot], globals.size(),
                                  stream_),
                "globals HtoDAsync")) {
            return;
        }

        // Threads to groups. The client asks in threads because that is the
        // number both backends agree about; the rounding up is why every kernel
        // starts with a bounds check.
        const auto groups = [](uint32_t threads, uint32_t size) {
            return (threads + size - 1) / size;
        };
        (void)cuEventRecord(timings_[timingAt_].start, stream_);

        // No kernel parameters at all: Slang puts everything in the
        // __constant__ block, so the launch passes nothing.
        (void)ok(cuLaunchKernel(kernel.function, groups(grid.x, kGroupX),
                                groups(grid.y, kGroupY), groups(grid.z, 1),
                                kGroupX, kGroupY, 1, 0, stream_, nullptr,
                                nullptr),
                 "cuLaunchKernel");

        // Bracketed by events, so the device times itself.
        //
        // Read later, never here: cuEventElapsedTime blocks until the pair has
        // happened, and blocking is the one thing a dispatch may not do. The
        // ring is harvested on the way into the *next* dispatch, by which time
        // the older pairs have long finished.
        harvestTimings();
        Timing& timing = timings_[timingAt_];
        timingAt_ = (timingAt_ + 1) % kTimingSlots;
        if (timing.pending) {
            timing.pending = false;   // dropped: the ring wrapped before it was read
        }
        (void)cuEventRecord(timing.stop, stream_);
        timing.pending = true;

        // And a host function behind it, which the driver runs when the stream
        // reaches it -- Metal's completion handler, spelled differently.
        //
        // `this` is the whole payload. Allocating a {device, submission} pair
        // per dispatch would be a malloc on the frame path; instead the
        // callback counts, and because host functions on one stream run in
        // order the Nth of them is the Nth dispatch. Nothing inside it touches
        // a CUDA API, which the driver forbids there.
        ++submitted_;
        (void)ok(cuLaunchHostFunc(stream_, &CudaDevice::onCompleted, this),
                 "cuLaunchHostFunc");
    }

    void sync() override { (void)ok(cuStreamSynchronize(stream_), "streamSync"); }

private:
    struct Allocation {
        CUdeviceptr ptr = 0;
        size_t      bytes = 0;
    };
    struct Timing {
        CUevent start = nullptr;
        CUevent stop = nullptr;
        bool    pending = false;
    };

    struct Kernel {
        std::string name;
        CUfunction  function = nullptr;
        CUdeviceptr globals = 0;
        size_t      globalsBytes = 0;
    };

    /// Matches [numthreads(16, 16, 1)] in the kernels. One place, and
    /// common.slang says the same numbers on the other side.
    static constexpr uint32_t kGroupX = 16;
    static constexpr uint32_t kGroupY = 16;

    /// Enough for any uniform block a kernel has, and small enough that four of
    /// them are free. The ring is so that a dispatch does not overwrite the
    /// uniforms of one still in flight.
    static constexpr size_t kStagingBytes = 4096;
    static constexpr int    kStagingSlots = 4;
    /// Deeper than the three-frame pipeline, so a pair is always finished long
    /// before its slot comes round again.
    static constexpr int    kTimingSlots = 8;

    /// Reads any event pair the device has finished with. Never waits:
    /// cuEventQuery is a poll, and a pair that is not ready is left for the
    /// next dispatch to find.
    void harvestTimings() {
        for (Timing& timing : timings_) {
            if (!timing.pending || cuEventQuery(timing.stop) != CUDA_SUCCESS) {
                continue;
            }
            float ms = 0.0f;
            if (cuEventElapsedTime(&ms, timing.start, timing.stop) == CUDA_SUCCESS) {
                gpuMs_.store(ms, std::memory_order_relaxed);
            }
            timing.pending = false;
        }
    }

    static void CUDA_CB onCompleted(void* userData) {
        auto* self = static_cast<CudaDevice*>(userData);
        self->reportCompleted(
            self->finished_.fetch_add(1, std::memory_order_relaxed) + 1);
    }

    [[nodiscard]] Allocation* find(BufferId id) {
        if (id == kInvalidBuffer || id > buffers_.size()) {
            return nullptr;
        }
        Allocation& a = buffers_[id - 1];
        return a.ptr != 0 ? &a : nullptr;
    }
    [[nodiscard]] const Allocation* find(BufferId id) const {
        return const_cast<CudaDevice*>(this)->find(id);
    }

    static void appendBytes(std::vector<unsigned char>& to, const void* from,
                            size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(from);
        to.insert(to.end(), p, p + bytes);
    }

    CUdevice    device_ = 0;
    CUcontext   context_ = nullptr;
    CUstream    stream_ = nullptr;
    CUdeviceptr staging_[kStagingSlots]{};
    void*       pinned_[kStagingSlots]{};
    int         stagingAt_ = 0;
    /// This backend's own count, matching the pool's because both increment
    /// once per dispatch and nothing else does.
    uint64_t              submitted_ = 0;
    std::atomic<uint64_t> finished_{0};
    Timing                timings_[kTimingSlots]{};
    int                   timingAt_ = 0;

    std::vector<Allocation> buffers_;
    std::vector<Kernel>     kernels_;
    std::vector<CUmodule>   modules_;
};

}   // namespace

std::unique_ptr<Device> Device::create() {
    auto device = std::make_unique<CudaDevice>();
    if (!device->open()) {
        return nullptr;
    }
    return device;
}

}   // namespace gpe
