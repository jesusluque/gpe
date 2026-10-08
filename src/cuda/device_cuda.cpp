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

#if defined(_WIN32)
// For the security attributes a Win32 shareable allocation is made with, and
// nothing else. See `shareableSecurity`.
#include <windows.h>
#include <sddl.h>
#include <winternl.h>
#endif

#include <atomic>
#include <cstdio>
#include <optional>
#include <cstring>
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

#if defined(_WIN32)
/// The object attributes a shareable allocation is created with.
///
/// Mandatory, and the type is the trap. The field is called
/// `win32HandleMetaData` and it wants a **POBJECT_ATTRIBUTES** -- not the
/// `SECURITY_ATTRIBUTES` that every other Win32 sharing call takes, and not a
/// null pointer either. Given the wrong struct, or none, `cuMemCreate` reads
/// the layout it expected and the process goes with an access violation and no
/// message. Measured here twice before the type was read properly.
///
/// The descriptor itself is NVIDIA's own, as an SDDL string: the point of the
/// allocation is that another process can open it, so it grants the World the
/// rights a section object is opened with. Built once for the life of the
/// process -- it describes the class of allocation, not any one of them.
struct ShareableSecurity {
    ShareableSecurity() {
        // D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD) -- a protected DACL
        // granting the World (WD) generic-all and the specific rights a
        // section object needs. Taken from CUDA's own IPC samples rather than
        // composed here: the right set is not obvious and getting it wrong
        // fails at import time in the child, which is a long way from here.
        static const char kSddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
        if (ConvertStringSecurityDescriptorToSecurityDescriptorA(
                kSddl, SDDL_REVISION_1, &descriptor, nullptr) == 0) {
            return;
        }
        InitializeObjectAttributes(&attributes, nullptr, 0, nullptr, descriptor);
        good = true;
    }

    OBJECT_ATTRIBUTES    attributes{};
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    bool                 good = false;
};

/// One per process. Never destroyed: it outlives every allocation made with it,
/// and tearing it down at exit would race the driver's own teardown.
ShareableSecurity& shareableSecurity() {
    static ShareableSecurity instance;
    return instance;
}
#endif

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

class CudaDevice final : public Device,
                        public CompletionReporting,
                        public HostStaging {
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
            for (const UploadSlot& slot : uploads_) {
                if (slot.host != nullptr) {
                    cuMemFreeHost(slot.host);
                }
                if (slot.done != nullptr) {
                    cuEventDestroy(slot.done);
                }
            }
            for (const CUevent done : stagingDone_) {
                if (done != nullptr) {
                    cuEventDestroy(done);
                }
            }
            if (copyDone_ != nullptr) {
                cuEventDestroy(copyDone_);
            }
            if (computeDone_ != nullptr) {
                cuEventDestroy(computeDone_);
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
            if (!adoptedContext_) {
                cuDevicePrimaryCtxRelease(device_);
            }
        }
    }

    /// `context`/`stream` non-null adopts them: a renderer on the same card
    /// (slang-rhi's CUDA device) made the context, and gpe's allocations and
    /// kernels live in it rather than in a second one. Neither is released
    /// here; they belong to whoever made them.
    [[nodiscard]] bool open(CUcontext context = nullptr, CUstream stream = nullptr) {
        if (!ok(cuInit(0), "cuInit")) {
            return false;
        }
        if (context != nullptr) {
            context_ = context;
            adoptedContext_ = true;
            if (!ok(cuCtxSetCurrent(context_), "ctxSetCurrent(adopted)") ||
                !ok(cuCtxGetDevice(&device_), "ctxGetDevice(adopted)")) {
                return false;
            }
        } else {
            int count = 0;
            if (!ok(cuDeviceGetCount(&count), "cuDeviceGetCount") || count == 0) {
                return false;
            }
            if (!ok(cuDeviceGet(&device_, 0), "cuDeviceGet") ||
                !ok(cuDevicePrimaryCtxRetain(&context_, device_), "ctxRetain") ||
                !ok(cuCtxSetCurrent(context_), "ctxSetCurrent")) {
                return false;
            }
        }
        // A stream of our own, not the default one. The default stream
        // synchronises against every other stream in the process, which would
        // make "asynchronous by default" a promise the first library we link
        // against could break.
        //
        // Unless one is handed over: the renderer's stream, so that its work
        // and gpe's are ordered by the stream itself, exactly as openFXplayer
        // orders TensorRT behind gpe's kernels.
        if (stream != nullptr) {
            stream_ = stream;
        } else if (!ok(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "streamCreate")) {
            return false;
        }
        // A second stream for transfers.
        //
        // This is where the overlap comes from, and it is worth being precise
        // about why: an upload on one stream and a kernel on another run at the
        // same time because the copy engine and the SMs are different hardware,
        // not because two threads issued them. One thread issuing to two
        // streams gets the whole of the benefit.
        //
        // They are ordered where it matters and nowhere else, and it matters in
        // *both* directions. An event recorded after an upload, waited on before
        // the next dispatch, so a kernel never reads a buffer whose bytes are
        // still in flight -- and an event recorded after compute work, waited on
        // before the next upload, so a DMA never lands in a buffer a memset or a
        // kernel is about to write.
        //
        // Only the first half existed, and the second half is not hypothetical:
        // `clear()` on a mirrored image memsets the device copy on the compute
        // stream, and the host's pixels go up on the copy stream. Nothing
        // ordered the two, so the memset could overtake the upload and wipe the
        // picture that had just arrived. Measured at five runs in ten, as an
        // input that reached the kernel as solid black -- and the kernel then
        // did exactly the right thing with it, which is why it read as a broken
        // effect rather than a broken transfer.
        if (!ok(cuStreamCreate(&copyStream_, CU_STREAM_NON_BLOCKING),
                "copyStreamCreate") ||
            !ok(cuEventCreate(&copyDone_, CU_EVENT_DISABLE_TIMING),
                "copyEventCreate") ||
            !ok(cuEventCreate(&computeDone_, CU_EVENT_DISABLE_TIMING),
                "computeEventCreate")) {
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
        // One completion event per staging slot. The ring alone is not the
        // fix it looks like: a slot is reused after kStagingSlots/2
        // dispatches, and an async copy reads its pinned source when the
        // stream reaches it -- under a deep queue the memcpy for a later
        // dispatch was overwriting bytes a DMA had not yet read. The event
        // is recorded behind each copy and waited on before the slot is
        // written again; in the shallow-queue case the wait is a no-op.
        for (CUevent& done : stagingDone_) {
            if (!ok(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING),
                    "eventCreate(staging)")) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] Backend backend() const override { return Backend::CUDA; }

    /// The CUDA context is a per-thread notion and this device is not: images
    /// are allocated on render threads and pictures prepared on the painting
    /// thread, by design. Every entry point that touches the driver makes the
    /// context current first -- a TLS write, idempotent and cheap. Without it
    /// the first call from a new thread fails with CUDA_ERROR_INVALID_CONTEXT,
    /// which stayed hidden for as long as discrete cards refused the shared
    /// context and every caller happened to sit on one thread.
    void ensureCurrent() const { (void)cuCtxSetCurrent(context_); }

    [[nodiscard]] BufferId alloc(size_t bytes) override {
        ensureCurrent();
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

    [[nodiscard]] BufferId adopt(uint64_t devicePtr, size_t bytes) override {
        if (devicePtr == 0 || bytes == 0) {
            return kInvalidBuffer;
        }
        buffers_.push_back(
            Allocation{static_cast<CUdeviceptr>(devicePtr), bytes, true});
        return static_cast<BufferId>(buffers_.size());
    }

    [[nodiscard]] BufferId allocShareable(size_t bytes) override {
        ensureCurrent();
        if (bytes == 0) {
            return kInvalidBuffer;
        }
        if (!shareableSupported()) {
            return kInvalidBuffer;
        }
        const CUmemAllocationProp prop = shareableProp();
#if defined(_WIN32)
        if (prop.win32HandleMetaData == nullptr) {
            // Refused rather than attempted: cuMemCreate dereferences this
            // field and takes the process with it if it is not there.
            std::fprintf(stderr,
                         "gpe: no object attributes for a shareable "
                         "allocation; not making one\n");
            return kInvalidBuffer;
        }
#endif
        size_t granularity = 0;
        if (!ok(cuMemGetAllocationGranularity(&granularity, &prop,
                                              CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                "cuMemGetAllocationGranularity") ||
            granularity == 0) {
            return kInvalidBuffer;
        }
        // Up to the granularity, which is two megabytes on every card this
        // runs on. The caller is told `bytes`; the driver is told the rounded
        // size, and every call about this allocation afterwards has to use the
        // rounded one or it fails with an alignment error that names neither.
        const size_t reserved = ((bytes + granularity - 1) / granularity) * granularity;

        CUmemGenericAllocationHandle memory = 0;
        if (!ok(cuMemCreate(&memory, reserved, &prop, 0), "cuMemCreate")) {
            return kInvalidBuffer;
        }
        const CUdeviceptr ptr = mapShareable(memory, reserved);
        if (ptr == 0) {
            (void)cuMemRelease(memory);
            return kInvalidBuffer;
        }
        Allocation made;
        made.ptr = ptr;
        made.bytes = bytes;
        made.memory = memory;
        made.reserved = reserved;
        buffers_.push_back(made);
        return static_cast<BufferId>(buffers_.size());
    }

    [[nodiscard]] uint64_t exportShareable(BufferId id) const override {
        ensureCurrent();
        const Allocation* a = find(id);
        if (a == nullptr || a->memory == 0) {
            return 0;   // an ordinary allocation cannot be exported
        }
#if defined(_WIN32)
        // A fresh HANDLE every time, which is what the driver does. The caller
        // closes it; this cannot, because it does not know when the child that
        // was given a duplicate is finished with it.
        HANDLE handle = nullptr;
        if (!ok(cuMemExportToShareableHandle(&handle, a->memory,
                                             CU_MEM_HANDLE_TYPE_WIN32, 0),
                "cuMemExportToShareableHandle")) {
            return 0;
        }
        return reinterpret_cast<uint64_t>(handle);
#else
        int fd = -1;
        if (!ok(cuMemExportToShareableHandle(
                    &fd, a->memory, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
                "cuMemExportToShareableHandle")) {
            return 0;
        }
        return static_cast<uint64_t>(fd);
#endif
    }

    [[nodiscard]] BufferId importShareable(uint64_t handle, size_t bytes) override {
        ensureCurrent();
        if (handle == 0 || bytes == 0) {
            return kInvalidBuffer;
        }
        if (!shareableSupported()) {
            return kInvalidBuffer;
        }
        const CUmemAllocationProp prop = shareableProp();
        size_t granularity = 0;
        if (!ok(cuMemGetAllocationGranularity(&granularity, &prop,
                                              CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                "cuMemGetAllocationGranularity") ||
            granularity == 0) {
            return kInvalidBuffer;
        }
        // The same rounding the exporter did. It is not written down anywhere
        // that crosses, so both ends compute it from the same granularity --
        // which is a property of the device, and both ends are on this one.
        const size_t reserved = ((bytes + granularity - 1) / granularity) * granularity;

        CUmemGenericAllocationHandle memory = 0;
#if defined(_WIN32)
        void* osHandle = reinterpret_cast<void*>(handle);
        constexpr CUmemAllocationHandleType kType = CU_MEM_HANDLE_TYPE_WIN32;
#else
        void* osHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(handle));
        constexpr CUmemAllocationHandleType kType =
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
        if (!ok(cuMemImportFromShareableHandle(&memory, osHandle, kType),
                "cuMemImportFromShareableHandle")) {
            return kInvalidBuffer;
        }
        const CUdeviceptr ptr = mapShareable(memory, reserved);
        if (ptr == 0) {
            (void)cuMemRelease(memory);
            return kInvalidBuffer;
        }
        Allocation made;
        made.ptr = ptr;
        made.bytes = bytes;
        made.memory = memory;
        made.reserved = reserved;
        made.imported = true;
        buffers_.push_back(made);
        return static_cast<BufferId>(buffers_.size());
    }

    void release(BufferId id) override {
        ensureCurrent();
        if (Allocation* a = find(id); a != nullptr && a->ptr != 0) {
            if (a->memory != 0) {
                // The three objects a shareable allocation is, in the order
                // that leaves nothing behind: the mapping, then the allocation
                // -- unless it is somebody else's, in which case only their
                // reference goes -- then the address range.
                (void)cuMemUnmap(a->ptr, a->reserved);
                (void)cuMemRelease(a->memory);
                (void)cuMemAddressFree(a->ptr, a->reserved);
            } else if (!a->borrowed) {
                cuMemFree(a->ptr);
            }
            *a = Allocation{};
        }
    }

    void upload(BufferId id, const void* src, size_t bytes) override {
        ensureCurrent();
        const Allocation* a = find(id);
        if (a == nullptr || src == nullptr) {
            return;
        }
        // Bounded against the allocation, the way `fill` already was.
        //
        // Without it a caller that works out its own byte count wrongly writes
        // past the end of a device allocation, and nothing anywhere says so:
        // there is no fault to take, only another buffer's pixels changing. The
        // pool rounding every size up to a bucket hides most of the small
        // cases, which makes the ones that survive rarer and no less wrong.
        if (bytes > a->bytes) {
            std::fprintf(stderr,
                         "gpe/cuda: upload of %zu bytes to buffer %llu, which "
                         "holds %zu\n",
                         bytes, static_cast<unsigned long long>(id), a->bytes);
            return;
        }
        // Behind whatever the compute stream was doing, and no further. The
        // mirror image of the wait in `dispatch`: without it a DMA can land in
        // a buffer that a memset queued earlier is about to clear.
        awaitCompute(copyStream_);
        // Already pinned? Then there is nothing to stage.
        //
        // The decode ring hands out memory this backend allocated, so the bytes
        // are somewhere the driver can DMA out of directly. Copying them into
        // another pinned buffer first would be 33 MB of memcpy per frame to
        // arrive exactly where they already were.
        if (isHostBlock(src, bytes)) {
            if (ok(cuMemcpyHtoDAsync(a->ptr, src, bytes, copyStream_),
                   "upload HtoDAsync(direct)")) {
                (void)ok(cuEventRecord(copyDone_, copyStream_), "copyEventRecord");
                uploadPending_ = true;
            }
            return;
        }

        // Through pinned memory the engine owns, on the copy stream.
        //
        // `src` belongs to the caller, who may destroy it the moment this
        // returns, so the bytes are taken now -- a memcpy, which is fast and
        // synchronous -- and the DMA out of pinned memory is what overlaps.
        // Copying straight from the caller's pointer asynchronously would read
        // a buffer that no longer exists, which is a bug this file has already
        // made once.
        void* staging = uploadStaging(bytes);
        if (staging == nullptr) {
            // No pinned buffer to be had: correct beats fast.
            (void)ok(cuMemcpyHtoD(a->ptr, src, bytes), "HtoD");
            return;
        }
        std::memcpy(staging, src, bytes);
        if (!ok(cuMemcpyHtoDAsync(a->ptr, staging, bytes, copyStream_),
                "upload HtoDAsync")) {
            return;
        }
        // Recorded on the copy stream; the next dispatch waits on it, and the
        // slot's own event guards its pinned buffer against early reuse.
        (void)ok(cuEventRecord(copyDone_, copyStream_), "copyEventRecord");
        if (lastUploadSlot_ != nullptr && lastUploadSlot_->done != nullptr) {
            (void)ok(cuEventRecord(lastUploadSlot_->done, copyStream_),
                     "uploadSlotRecord");
        }
        uploadPending_ = true;
    }

    void download(void* dst, BufferId id, size_t bytes) override {
        ensureCurrent();
        const Allocation* a = find(id);
        if (a == nullptr || dst == nullptr) {
            return;
        }
        // Bounded, as upload now is. Reading past the end of an allocation is
        // the quieter half of the same mistake: it does not corrupt anything on
        // the device, it just hands the caller whatever was next in memory.
        if (bytes > a->bytes) {
            std::fprintf(stderr,
                         "gpe/cuda: download of %zu bytes from buffer %llu, "
                         "which holds %zu\n",
                         bytes, static_cast<unsigned long long>(id), a->bytes);
            return;
        }
        // The one synchronous call in the interface: it has to be, because the
        // bytes must be there when it returns.
        //
        // Enqueued on the compute stream, not issued as a plain cuMemcpyDtoH.
        // The plain call synchronises with the *legacy default* stream, and
        // stream_ is CU_STREAM_NON_BLOCKING -- so the copy engine overtook a
        // kernel still writing the buffer and handed back the frame from
        // before it. The display test caught it: the 33-cube LUT won that
        // race every run and the 65-cube lost it every run, which read as a
        // size-dependent sampling bug and was nothing of the kind.
        if (uploadPending_) {
            // A download straight after an upload, with no dispatch between:
            // order it behind the copy stream's work too, as dispatch would.
            (void)ok(cuStreamWaitEvent(stream_, copyDone_, 0), "streamWaitEvent");
            uploadPending_ = false;
        }
        if (!ok(cuMemcpyDtoHAsync(dst, a->ptr, bytes, stream_), "DtoHAsync")) {
            return;
        }
        (void)ok(cuStreamSynchronize(stream_), "downloadSync");
    }

    [[nodiscard]] uint64_t devicePointer(BufferId id) const override {
        const Allocation* a = find(id);
        return a != nullptr ? static_cast<uint64_t>(a->ptr) : 0;
    }

    [[nodiscard]] uint64_t stream() const override {
        return reinterpret_cast<uint64_t>(stream_);
    }

    [[nodiscard]] uint64_t backendBuffer(BufferId id) const override {
        return devicePointer(id);
    }
    [[nodiscard]] uint64_t backendDevice() const override {
        return reinterpret_cast<uint64_t>(context_);
    }
    [[nodiscard]] uint64_t backendQueue() const override { return stream(); }

    void memory(size_t& total, size_t& available) const override {
        ensureCurrent();
        size_t free = 0;
        size_t all = 0;
        if (ok(cuMemGetInfo(&free, &all), "cuMemGetInfo")) {
            total = all;
            available = free;
        } else {
            total = 0;
            available = 0;
        }
    }

    /// Remembers that the compute stream has work an upload must not overtake.
    void noteCompute() noexcept {
        if (computeDone_ != nullptr &&
            ok(cuEventRecord(computeDone_, stream_), "computeEventRecord")) {
            computePending_ = true;
        }
    }

    /// Orders `into` behind that work, once. Cheap when there is none: the
    /// whole point is to keep the overlap everywhere it is safe to have.
    void awaitCompute(CUstream into) noexcept {
        if (!computePending_) {
            return;
        }
        (void)ok(cuStreamWaitEvent(into, computeDone_, 0), "computeWaitEvent");
        computePending_ = false;
    }

    [[nodiscard]] bool fill(BufferId id, uint8_t byte, size_t bytes) override {
        ensureCurrent();
        const Allocation* a = find(id);
        if (a == nullptr || bytes == 0 || bytes > a->bytes) {
            return false;
        }
        // On the compute stream, for the same reason `download` is: a memset
        // that raced ahead of a kernel still writing the buffer would clear the
        // picture it had just made, and only sometimes.
        if (!ok(cuMemsetD8Async(a->ptr, byte, bytes, stream_), "memsetD8Async")) {
            return false;
        }
        noteCompute();
        return true;
    }

    [[nodiscard]] bool copy(BufferId destination, BufferId source,
                            size_t bytes) override {
        ensureCurrent();
        const Allocation* to = find(destination);
        const Allocation* from = find(source);
        if (to == nullptr || from == nullptr || bytes == 0 ||
            bytes > to->bytes || bytes > from->bytes) {
            return false;
        }
        // On the compute stream, for the reason `fill` is on it: a copy that
        // ran ahead of the kernel still writing the source would carry the
        // picture from before, and only sometimes.
        if (!ok(cuMemcpyDtoDAsync(to->ptr, from->ptr, bytes, stream_),
                "memcpyDtoDAsync")) {
            return false;
        }
        noteCompute();
        return true;
    }

    [[nodiscard]] KernelId load(std::string_view name) override {
        ensureCurrent();
        for (size_t i = 0; i < kernels_.size(); ++i) {
            if (kernels_[i].name == name) {
                return static_cast<KernelId>(i + 1);
            }
        }
        // Through the registry, so a kernel a plugin brought with it is
        // found the same way one compiled into this library is -- and with its
        // trailer, if it has one, already taken off: PTX is text and a
        // trailer after it is a module the driver refuses.
        const kernels::Resolved* resolved = kernels::resolve(name);
        if (resolved == nullptr) {
            std::fprintf(stderr, "gpe/cuda: no kernel '%.*s' in this binary\n",
                         static_cast<int>(name.size()), name.data());
            return kInvalidKernel;
        }
        const kernels::Blob* blob = &resolved->blob;
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
                                  globalsBytes, resolved->info});
        return static_cast<KernelId>(kernels_.size());
    }

    void dispatch(KernelId id, Grid grid, const void* args,
                  size_t bytes) override {
        ensureCurrent();
        if (id == kInvalidKernel || id > kernels_.size()) {
            return;
        }
        const Kernel& kernel = kernels_[id - 1];
        const Args::View view = Args::read(args, bytes);
        if (!view.valid) {
            std::fprintf(stderr, "gpe/cuda: malformed dispatch args\n");
            return;
        }
        // What the kernel's reflection says, when its blob carried it. A
        // dispatch that does not match is refused here rather than run over
        // the wrong buffers -- openFXplayer's D30, made an error.
        if (kernel.info.has_value()) {
            if (kernel.info->elementBytes.size() != view.bufferCount) {
                std::fprintf(stderr,
                             "gpe/cuda: %s declares %zu buffers and was handed %u\n",
                             kernel.name.c_str(), kernel.info->elementBytes.size(),
                             view.bufferCount);
                return;
            }
            if (kernel.info->uniformBytes != view.uniformBytes) {
                std::fprintf(stderr,
                             "gpe/cuda: %s declares %u bytes of uniforms and was handed %u\n",
                             kernel.name.c_str(), kernel.info->uniformBytes,
                             view.uniformBytes);
                return;
            }
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
            // The slot's previous copy must be out of the pinned buffer
            // before new bytes go in. A never-recorded event answers
            // immediately, so the first lap costs nothing.
            (void)ok(cuEventSynchronize(stagingDone_[slot]), "stagingWait");
            std::memcpy(pinned_[slot], view.uniforms, view.uniformBytes);
            if (!ok(cuMemcpyHtoDAsync(uniforms, pinned_[slot], view.uniformBytes,
                                      stream_),
                    "uniforms HtoDAsync")) {
                return;
            }
            (void)ok(cuEventRecord(stagingDone_[slot], stream_),
                     "stagingRecord");
        }

        // Then the globals struct: every buffer as {pointer, count}, in the
        // order the kernel declares them, with the uniform pointer last.
        //
        // A member, cleared and refilled, rather than a vector built here. A
        // heap allocation per dispatch is exactly what this file says a
        // dispatch may not do, and there are several per frame; `clear()` keeps
        // the capacity, so after the first dispatch of a session there is no
        // allocation left in this path at all.
        std::vector<unsigned char>& globals = globals_;
        globals.clear();
        globals.reserve(kernel.globalsBytes);
        for (uint32_t i = 0; i < view.bufferCount; ++i) {
            const Allocation* a = find(view.buffers[i]);
            SlangBuffer entry{};
            if (a != nullptr) {
                // Elements, not bytes, and the element size comes from the
                // caller because the host cannot infer it. Slang bound-checks
                // against this number: too small and it drops writes without
                // saying so.
                //
                // The kernel's own answer wins over the caller's. A host
                // relaying a plugin's buffers cannot know T and passed one byte
                // to keep this check from truncating; reflection knows T.
                uint32_t stride = view.elementBytes != nullptr &&
                                          view.elementBytes[i] > 0
                                      ? view.elementBytes[i]
                                      : static_cast<uint32_t>(kBytesPerPixel);
                if (kernel.info.has_value() && kernel.info->elementBytes[i] > 0) {
                    stride = kernel.info->elementBytes[i];
                }
                entry.data = a->ptr;
                entry.count = a->bytes / stride;
            }
            // What the kernel is actually handed, which is not the same
            // question as what the caller believes it passed -- and the only
            // place the two can be compared. Behind an environment variable
            // and a static, so it costs one predicted branch per buffer when
            // nobody is looking.
            static const bool trace =
                std::getenv("GPE_TRACE_BUFFERS") != nullptr;
            if (trace) {
                std::fprintf(stderr,
                             "gpe/cuda: %s slot %u -> %p (%zu, stride %u)\n",
                             kernel.name.c_str(), i,
                             reinterpret_cast<void*>(entry.data), entry.count,
                             view.elementBytes != nullptr ? view.elementBytes[i]
                                                          : 0);
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
        (void)ok(cuEventSynchronize(stagingDone_[slot]), "stagingWait");
        std::memcpy(pinned_[slot], globals.data(), globals.size());
        if (!ok(cuMemcpyHtoDAsync(kernel.globals, pinned_[slot], globals.size(),
                                  stream_),
                "globals HtoDAsync")) {
            return;
        }
        (void)ok(cuEventRecord(stagingDone_[slot], stream_), "stagingRecord");

        // Threads to groups. The client asks in threads because that is the
        // number both backends agree about; the rounding up is why every kernel
        // starts with a bounds check.
        const auto groups = [](uint32_t threads, uint32_t size) {
            return (threads + size - 1) / size;
        };
        // The block takes the grid's shape. Metal launches the exact grid
        // (dispatchThreads clips the partial group), so a 1D kernel there
        // never sees a thread with y > 0. CUDA can only round up whole
        // blocks -- and rounding a 1D grid up to 16x16 blocks launches
        // sixteen threads for every x, each passing the kernel's x-only
        // bounds check. The atomics probe counted exactly 16x too many, and
        // every 1D splat pass was doing its work sixteen times over.
        //
        // Two shapes cover the house's two kernel families: image kernels
        // are 2D, declare 16x16, and the cooperative ones (tile staging,
        // barriers) rely on exactly that group geometry; the 1D passes
        // declare 64x1 or 1x1 and cooperate through nothing but atomics, so
        // a flat block is both correct and fully occupied.
        const bool     flat = grid.y == 1 && grid.z == 1;
        uint32_t       blockX = flat ? kGroupX * kGroupY : kGroupX;
        uint32_t       blockY = flat ? 1 : kGroupY;
        uint32_t       blockZ = 1;
        // A kernel that says what its group is gets exactly that, as it would
        // on D3D or Vulkan -- the guess above is for blobs that do not say.
        if (kernel.info.has_value()) {
            blockX = kernel.info->threadGroup[0];
            blockY = kernel.info->threadGroup[1];
            blockZ = kernel.info->threadGroup[2];
        }
        // Compute waits for whatever was uploaded since the last dispatch, and
        // for nothing else. Without this the kernel could read a buffer whose
        // DMA is still running; with a full synchronisation instead, there
        // would be no overlap left to have.
        if (uploadPending_) {
            (void)ok(cuStreamWaitEvent(stream_, copyDone_, 0), "streamWaitEvent");
            uploadPending_ = false;
        }

        (void)cuEventRecord(timings_[timingAt_].start, stream_);

        // No kernel parameters at all: Slang puts everything in the
        // __constant__ block, so the launch passes nothing.
        (void)ok(cuLaunchKernel(kernel.function, groups(grid.x, blockX),
                                groups(grid.y, blockY), groups(grid.z, blockZ),
                                blockX, blockY, blockZ, 0, stream_, nullptr,
                                nullptr),
                 "cuLaunchKernel");
        noteCompute();

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

    // --- HostStaging -------------------------------------------------------

    [[nodiscard]] void* allocHost(size_t bytes) override {
        ensureCurrent();
        void* memory = nullptr;
        if (!ok(cuMemAllocHost(&memory, bytes), "cuMemAllocHost(ring)")) {
            return nullptr;
        }
        hostBlocks_.push_back(HostBlock{memory, bytes});
        return memory;
    }
    void freeHost(void* memory) override {
        ensureCurrent();
        if (memory == nullptr) {
            return;
        }
        for (auto it = hostBlocks_.begin(); it != hostBlocks_.end(); ++it) {
            if (it->memory == memory) {
                hostBlocks_.erase(it);
                break;
            }
        }
        cuMemFreeHost(memory);
    }

    void sync() override {
        ensureCurrent();
        // Both, because a caller asking for everything to be finished means
        // both the work and the transfers.
        (void)ok(cuStreamSynchronize(copyStream_), "copyStreamSync");
        (void)ok(cuStreamSynchronize(stream_), "streamSync");
    }

private:
    struct Allocation {
        CUdeviceptr ptr = 0;
        size_t      bytes = 0;
        /// Somebody else's memory. Read and dispatched over like any other,
        /// and never freed here: freeing it would take a frame out from under
        /// the decoder that is still recycling it.
        bool        borrowed = false;
        /// The virtual-memory allocation behind a shareable buffer, and the
        /// size it was rounded up to. Zero for an ordinary `cuMemAlloc`, which
        /// is freed with `cuMemFree` and knows nothing about any of this.
        ///
        /// Three objects, not one, and releasing takes all three in order:
        /// the mapping, then the allocation, then the address range. Skipping
        /// the last leaks address space rather than memory, which is the kind
        /// of leak a long-running render host notices a day later.
        CUmemGenericAllocationHandle memory = 0;
        size_t                       reserved = 0;
        /// Mapped from somebody else's export. Unmapped here, never released:
        /// the allocation belongs to the process that made it.
        bool                         imported = false;
    };
    struct HostBlock {
        void*  memory = nullptr;
        size_t bytes = 0;
    };

    /// True if `src` lies wholly inside memory this backend page-locked.
    [[nodiscard]] bool isHostBlock(const void* src, size_t bytes) const {
        const auto* p = static_cast<const unsigned char*>(src);
        for (const HostBlock& block : hostBlocks_) {
            const auto* start = static_cast<const unsigned char*>(block.memory);
            if (p >= start && p + bytes <= start + block.bytes) {
                return true;
            }
        }
        return false;
    }

    struct UploadSlot {
        void*   host = nullptr;
        size_t  bytes = 0;
        /// Recorded behind this slot's DMA; waited on before the slot's
        /// pinned buffer is written again. Same bug and same cure as the
        /// dispatch staging ring.
        CUevent done = nullptr;
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
        /// The blob's trailer, if it had one. See gpe/kernels.h.
        std::optional<KernelInfo> info;
    };

    /// Matches [numthreads(16, 16, 1)] in the kernels. One place, and
    /// common.slang says the same numbers on the other side.
    static constexpr uint32_t kGroupX = 16;
    static constexpr uint32_t kGroupY = 16;

    /// Enough for any uniform block a kernel has, and small enough that a
    /// dozen of them are free. The ring is so that a dispatch does not
    /// overwrite the uniforms of one still in flight.
    static constexpr size_t kStagingBytes = 4096;
    /// Twelve, not four.
    ///
    /// Each dispatch takes *two* slots -- uniforms and globals -- and taking
    /// one waits on that slot's event. Four slots is therefore a margin of two
    /// dispatches, against a three-frame pipeline with several dispatches in
    /// each: the wait was not a rare guard against an unusual burst, it was
    /// hit constantly, on the path this file says may not block. Twelve is six
    /// dispatches of margin at 48 KB, which is nothing on any card that runs
    /// this.
    static constexpr int    kStagingSlots = 12;
    /// Deeper than the three-frame pipeline, so a pair is always finished long
    /// before its slot comes round again.
    static constexpr int    kTimingSlots = 8;
    /// Matches the three-frame pipeline, plus one so a slot is never reused
    /// while its DMA is still running.
    static constexpr int    kUploadSlots = 4;

    /// A pinned host buffer of at least `bytes`, from a small ring.
    ///
    /// Grown on demand and never shrunk. It grows during prepare(), when the
    /// first frame's uploads set the size; a frame that uploads the same shape
    /// as the one before -- which is every frame of a playback -- finds it
    /// already there and allocates nothing.
    [[nodiscard]] void* uploadStaging(size_t bytes) {
        UploadSlot& slot = uploads_[uploadAt_];
        uploadAt_ = (uploadAt_ + 1) % kUploadSlots;
        if (slot.done == nullptr) {
            (void)ok(cuEventCreate(&slot.done, CU_EVENT_DISABLE_TIMING),
                     "eventCreate(upload)");
        } else {
            // The slot's previous DMA must be done reading before the caller
            // memcpys new bytes over it.
            (void)ok(cuEventSynchronize(slot.done), "uploadSlotWait");
        }
        if (slot.bytes < bytes) {
            if (slot.host != nullptr) {
                // In flight or not, this is prepare-time: waiting here is
                // allowed and freeing pinned memory a DMA is reading is not.
                (void)cuStreamSynchronize(copyStream_);
                cuMemFreeHost(slot.host);
                slot.host = nullptr;
                slot.bytes = 0;
            }
            if (!ok(cuMemAllocHost(&slot.host, bytes), "cuMemAllocHost(upload)")) {
                return nullptr;
            }
            slot.bytes = bytes;
        }
        lastUploadSlot_ = &slot;
        return slot.host;
    }

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

    /// Whether this card can do any of this, asked once.
    ///
    /// Asked rather than attempted, because the failure mode is not an error
    /// code. `cuMemCreate` on a device without virtual-memory management, or
    /// with a handle type it does not support, is not documented to return
    /// anything in particular, and the one thing measured here is that it can
    /// take the process with it. A capability question has an answer; asking
    /// it is cheaper than finding out.
    [[nodiscard]] bool shareableSupported() const {
        static const int supported = [this] {
            int vmm = 0;
            (void)cuDeviceGetAttribute(
                &vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                device_);
            int handle = 0;
            (void)cuDeviceGetAttribute(
                &handle,
#if defined(_WIN32)
                CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED,
#else
                CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
#endif
                device_);
            if (vmm == 0 || handle == 0) {
                std::fprintf(stderr,
                             "gpe: this device shares no allocations "
                             "(virtual memory %d, handle type %d)\n",
                             vmm, handle);
            }
            return vmm != 0 && handle != 0 ? 1 : 0;
        }();
        return supported != 0;
    }

    /// What a shareable allocation on this device is made of.
    ///
    /// One function because both ends need the identical description: the
    /// granularity the exporter rounds to is computed from it, and so is the
    /// one the importer rounds to.
    [[nodiscard]] CUmemAllocationProp shareableProp() const {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;
#if defined(_WIN32)
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
        // Mandatory. See ShareableSecurity for what happens without it.
        if (ShareableSecurity& security = shareableSecurity(); security.good) {
            prop.win32HandleMetaData = &security.attributes;
        }
#else
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
        return prop;
    }

    /// Address space for `memory`, mapped and made readable and writable.
    ///
    /// The half that is the same whether the allocation was made here or
    /// imported from elsewhere. Zero on failure, with everything it did
    /// undone: a reservation left behind on a failed map is address space
    /// nobody can name and nobody will free.
    [[nodiscard]] CUdeviceptr mapShareable(CUmemGenericAllocationHandle memory,
                                           size_t reserved) const {
        CUdeviceptr ptr = 0;
        if (!ok(cuMemAddressReserve(&ptr, reserved, 0, 0, 0),
                "cuMemAddressReserve")) {
            return 0;
        }
        if (!ok(cuMemMap(ptr, reserved, 0, memory, 0), "cuMemMap")) {
            (void)cuMemAddressFree(ptr, reserved);
            return 0;
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (!ok(cuMemSetAccess(ptr, reserved, &access, 1), "cuMemSetAccess")) {
            (void)cuMemUnmap(ptr, reserved);
            (void)cuMemAddressFree(ptr, reserved);
            return 0;
        }
        return ptr;
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
    /// Made by somebody else; not released here.
    bool        adoptedContext_ = false;
    CUstream    stream_ = nullptr;
    CUdeviceptr staging_[kStagingSlots]{};
    void*       pinned_[kStagingSlots]{};
    CUevent     stagingDone_[kStagingSlots]{};
    int         stagingAt_ = 0;
    UploadSlot* lastUploadSlot_ = nullptr;
    /// The globals struct being built, kept so that building it allocates
    /// nothing after the first dispatch. See its use.
    std::vector<unsigned char> globals_;
    /// This backend's own count, matching the pool's because both increment
    /// once per dispatch and nothing else does.
    uint64_t              submitted_ = 0;
    std::atomic<uint64_t> finished_{0};
    Timing                timings_[kTimingSlots]{};
    int                   timingAt_ = 0;
    CUstream              copyStream_ = nullptr;
    CUevent               copyDone_ = nullptr;
    CUevent               computeDone_ = nullptr;
    bool                  computePending_ = false;
    bool                  uploadPending_ = false;
    std::vector<HostBlock> hostBlocks_;
    UploadSlot            uploads_[kUploadSlots]{};
    int                   uploadAt_ = 0;

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

std::unique_ptr<Device> adoptCudaContext(void* cuContext, void* cuStream) {
    if (cuContext == nullptr) {
        return nullptr;
    }
    auto device = std::make_unique<CudaDevice>();
    if (!device->open(static_cast<CUcontext>(cuContext), static_cast<CUstream>(cuStream))) {
        return nullptr;
    }
    return device;
}

// No Metal on this backend. Defined so a client that asks links and is told.
std::unique_ptr<Device> adoptMetalDevice(void*) { return nullptr; }
std::unique_ptr<Device> adoptMetalDevice(void*, void*) { return nullptr; }

}   // namespace gpe
