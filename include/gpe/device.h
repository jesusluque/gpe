// Copyright (c) 2026 gpe contributors.
//
// The whole engine boundary. Eight virtual calls and a destructor.
//
// Client code names no backend and includes no backend header. `create()`
// decides once, at startup, and everything after that is this interface.
#pragma once

#include <functional>
#include <memory>
#include <string_view>

#include "gpe/types.h"

namespace gpe {

class Device {
public:
    enum class Backend { CUDA, Metal };

    /// The device for this machine, or null if there is none it can drive.
    ///
    /// Null rather than a throw: failing to find a GPU is a condition a host
    /// application decides what to do about -- fall back to CPU, tell the user,
    /// refuse to start -- and none of those want to be written as a catch.
    [[nodiscard]] static std::unique_ptr<Device> create();

    [[nodiscard]] virtual Backend backend() const = 0;

    // --- memory ------------------------------------------------------------
    //
    // Every one of these goes through the pool. `alloc` does not talk to the
    // driver in the common case and `release` does not free: cudaMalloc
    // synchronises the device implicitly and costs about 100 microseconds, so
    // an allocation per operation is an allocation per operation too many.

    /// A device buffer of at least `bytes`, or `kInvalidBuffer`.
    [[nodiscard]] virtual BufferId alloc(size_t bytes) = 0;
    /// Hands it back to the pool. The memory stays on the device.
    virtual void release(BufferId) = 0;

    virtual void upload(BufferId, const void* src, size_t) = 0;
    /// Reads back. This is the one call that is *not* asynchronous: it cannot
    /// be, because the bytes have to be there when it returns.
    virtual void download(void* dst, BufferId, size_t) = 0;

    /// Writes `byte` over the whole of `bytes`, on the device, copying nothing.
    ///
    /// The ninth call, and the one that earns its place: a host that keeps its
    /// images on the device has to be able to say "this one starts transparent
    /// black" without sending 33 MB of zeroes over PCIe for every node of every
    /// frame. That upload is the entire cost this interface exists to avoid,
    /// and there was no way to express the alternative.
    ///
    /// False where the backend cannot, and the caller zeroes host memory and
    /// uploads it the ordinary way -- which is what it did before this existed.
    /// Not pure for the same reason: a backend that has no answer keeps
    /// compiling and keeps working.
    [[nodiscard]] virtual bool fill(BufferId, uint8_t /*byte*/,
                                    size_t /*bytes*/) {
        return false;
    }

    /// `bytes` from one device buffer into another, on the device.
    ///
    /// The same argument `fill` makes. The caller this exists for is moving a
    /// picture between an allocation the pool made and one that can be shared
    /// with another process, and doing that through host memory would be two
    /// crossings of the bus to move something that never leaves the card:
    /// tens of microseconds against milliseconds each way.
    ///
    /// Queued on the same stream as everything else, so it orders itself
    /// against the kernels either side of it without a synchronise.
    ///
    /// False where the backend cannot, and the caller downloads and uploads --
    /// slower, and not a missing capability.
    [[nodiscard]] virtual bool copy(BufferId /*destination*/,
                                    BufferId /*source*/, size_t /*bytes*/) {
        return false;
    }

    // --- work --------------------------------------------------------------

    /// A kernel by name, from the blobs built into the binary. Compiling
    /// happens at build time; this is a lookup and, on first use, a link.
    [[nodiscard]] virtual KernelId load(std::string_view name) = 0;

    /// Queues `kernel` over `grid` threads. Returns without waiting.
    ///
    /// `args` is the kernel's parameter struct, copied by the call -- so the
    /// caller may reuse or destroy it immediately, which is what makes a loop
    /// of dispatches writable without keeping every argument alive.
    virtual void dispatch(KernelId, Grid grid, const void* args, size_t) = 0;

    /// Waits for everything queued so far. Once a frame, not once a dispatch.
    virtual void sync() = 0;

    /// Hands everything queued so far to the device, without waiting.
    ///
    /// A backend may hold several dispatches back to submit them together
    /// (Metal does: one command buffer, not one each). Everything this
    /// interface does that must follow them -- upload, download, fill, sync --
    /// already flushes. What cannot is another runtime submitting on the same
    /// queue (`backendQueue`, adopt.h): call this before it reads what gpe
    /// wrote. A backend that submits each dispatch at once has nothing to do.
    virtual void flush() {}

    /// The backend's own address for a buffer, and the queue it runs on.
    ///
    /// FOR ONE KIND OF CALLER ONLY
    ///
    /// A third-party runtime that compiles and launches its own work -- an
    /// inference engine is the case this exists for -- cannot go through
    /// `dispatch`, because what it has is not a kernel this library loaded. It
    /// needs the pointer to bind and the stream to enqueue on, and without
    /// both, the only way to hand it a picture is to copy that picture out and
    /// back, which on a 24 GB card is the whole point thrown away.
    ///
    /// Enqueuing on *this* stream rather than one of its own is what makes the
    /// ordering free: the work lands behind the kernels that produced its input
    /// and ahead of the ones that read its output, by the same rule everything
    /// else here obeys. A caller that takes these and then uses a stream of its
    /// own has to synchronise by hand, and will get it wrong.
    ///
    /// Zero from a backend with no such notion, and the caller has no business
    /// guessing. `Metal` returns zero for both: a `MTLBuffer` is not an address
    /// and a command queue is not a stream.
    /// Takes a handle to device memory somebody else owns.
    ///
    /// For memory that arrives already on the device and is not this engine's
    /// to allocate or free: a decoder's output frame, a capture card's buffer,
    /// anything another library hands over. The returned handle behaves like
    /// any other for reading and dispatching, and releasing it gives up the
    /// claim without freeing anything -- because freeing it is the owner's job
    /// and doing it here would take the frame out from under them.
    ///
    /// **The lifetime is the caller's problem and it is a short one.** A
    /// decoder recycles its buffers; a handle adopted from one is only valid
    /// while whatever kept that buffer alive is still held. Release it before
    /// letting go of the thing it came from, in that order.
    ///
    /// `kInvalidBuffer` from a backend with no notion of a foreign pointer,
    /// which is every backend but CUDA. A caller that gets one should fall back
    /// to uploading rather than fail: it is a missing optimisation, not a
    /// missing capability.
    [[nodiscard]] virtual BufferId adopt(uint64_t /*devicePtr*/,
                                         size_t /*bytes*/) {
        return kInvalidBuffer;
    }

    /// An allocation another process can be given, and the two calls around it.
    ///
    /// **What this is for.** A model worker is a separate process -- it has to
    /// be, because the model's environment is half its install -- and the whole
    /// design depends on handles crossing that boundary rather than pixels. On
    /// Linux that has been `cudaIpcGetMemHandle` over an ordinary allocation.
    /// That family is Linux-only: there is no Windows build of it at any
    /// driver, and the documented replacement is the virtual-memory API, which
    /// exports a file descriptor there and an NT handle here.
    ///
    /// So sharing is asked for at allocation time. An ordinary `alloc` is a
    /// plain device allocation and cannot be exported however much a caller
    /// would like it to be.
    ///
    /// **Not for everything.** The virtual-memory granularity is two
    /// megabytes and every shareable allocation is rounded up to it, so a pool
    /// of small scratch buffers made this way would waste more than it held.
    /// This is for the few stable buffers that leave the process -- a worker's
    /// input and output canvases, a delivered frame -- and `alloc` stays what
    /// everything else uses.
    ///
    /// `kInvalidBuffer` where the backend has no such notion, which is every
    /// backend but CUDA. A caller that gets one falls back to copying through
    /// host memory: slower, and not a missing capability.
    [[nodiscard]] virtual BufferId allocShareable(size_t /*bytes*/) {
        return kInvalidBuffer;
    }

    /// An operating-system handle for `id`, or zero.
    ///
    /// A file descriptor on POSIX, an NT `HANDLE` on Windows, widened to 64
    /// bits either way. **The caller owns it and must close it** -- `close`
    /// there, `CloseHandle` here -- and a fresh one comes back from every call,
    /// because that is what the driver does and pretending otherwise would leak
    /// one per frame.
    ///
    /// Zero for a buffer that did not come from `allocShareable`. That is the
    /// honest answer rather than an error: the caller asked whether this buffer
    /// can cross, and it cannot.
    ///
    /// **On Windows the handle alone is not enough.** NT handles are
    /// per-process, so the child has to be given a duplicate --
    /// `DuplicateHandle` against its process, or an inheritable handle and
    /// `bInheritHandles` on the spawn. A file descriptor travels by other
    /// means again. There is no spelling of this that hides the difference and
    /// stays truthful, so the caller is told which of the two it is holding by
    /// the platform it is compiled for.
    [[nodiscard]] virtual uint64_t exportShareable(BufferId) const { return 0; }

    /// The other end: somebody else's exported allocation, mapped here.
    ///
    /// `bytes` is what the exporter asked for, before rounding. The handle
    /// stays the caller's to close -- this maps what it names and does not take
    /// it over -- and `release` unmaps without touching the exporter's memory,
    /// the way `adopt` gives up a claim rather than freeing.
    [[nodiscard]] virtual BufferId importShareable(uint64_t /*handle*/,
                                                   size_t /*bytes*/) {
        return kInvalidBuffer;
    }

    [[nodiscard]] virtual uint64_t devicePointer(BufferId) const { return 0; }
    [[nodiscard]] virtual uint64_t stream() const { return 0; }

    /// The backend's own objects: an `MTL::Buffer*` or a CUDA device pointer
    /// for a buffer; the `MTL::Device*` or `CUcontext`; the `MTL::CommandQueue*`
    /// or `CUstream`.
    ///
    /// For one caller: a second runtime on the *same* device -- a renderer that
    /// adopted this device or that this device adopted -- which wraps these in
    /// its own buffer objects and reads what gpe wrote without a copy.
    /// `devicePointer` answers the same question for CUDA only, and for Metal
    /// says zero on purpose; this one answers for both, because a Metal caller
    /// that has a device of its own does know what to do with an MTLBuffer.
    ///
    /// Zero where there is no such object.
    [[nodiscard]] virtual uint64_t backendBuffer(BufferId) const { return 0; }
    [[nodiscard]] virtual uint64_t backendDevice() const { return 0; }
    [[nodiscard]] virtual uint64_t backendQueue() const { return 0; }

    /// Reads back without waiting: `done(true)` runs once `bytes` are in `dst`,
    /// on whatever thread the driver finishes on, and must only store.
    ///
    /// `dst` must stay valid until then. For a renderer that wants last
    /// frame's counts without stalling this frame for them.
    ///
    /// The default is the synchronous `download` followed by `done` -- correct,
    /// and exactly as slow as before, on a backend that has not been taught.
    virtual void downloadAsync(BufferId id, void* dst, size_t bytes,
                               std::function<void(bool)> done) {
        download(dst, id, bytes);
        if (done) {
            done(true);
        }
    }

    /// How much memory this device has, and how much of it is unspoken for.
    ///
    /// Both zero where the backend cannot say, and a caller that gets zero
    /// falls back to whatever constant it was using before. That constant is
    /// the problem this answers: a budget chosen for a laptop is a small
    /// fraction of a 24 GB card, and a host that keeps its pictures on the
    /// device then starts declining them a third of the way through a
    /// playback -- quietly, and only under load, which reads as the whole
    /// optimisation not working rather than as a number set too low.
    virtual void memory(size_t& total, size_t& available) const {
        total = 0;
        available = 0;
    }

    virtual ~Device() = default;
};

}   // namespace gpe
