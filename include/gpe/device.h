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

    virtual ~Device() = default;
};

}   // namespace gpe
