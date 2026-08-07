// Copyright (c) 2026 gpe contributors.
//
// The whole engine boundary. Eight virtual calls and a destructor.
//
// Client code names no backend and includes no backend header. `create()`
// decides once, at startup, and everything after that is this interface.
#pragma once

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

    [[nodiscard]] virtual uint64_t devicePointer(BufferId) const { return 0; }
    [[nodiscard]] virtual uint64_t stream() const { return 0; }

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
