// Copyright (c) 2026 gpe contributors.
//
// How a backend says what has finished, without knowing who is asking.
//
// The pool needs to know which submissions have retired, or a released buffer
// waits for a full `sync()` that a frame is not allowed to make. The backends
// must not know the pool exists, and `Device` is eight methods with no room for
// a ninth.
//
// So: a backend publishes a number, and whoever cares reads it. This is a
// second, tiny base class that a backend may inherit; PooledDevice looks for it
// once with a dynamic_cast in its constructor and never again. A backend that
// does not inherit it still works -- the pool falls back to `sync()`, which is
// correct and blunt.
//
// Pull rather than push, deliberately. A push needs the backend to hold a
// pointer to something it must not know about; a pull is an atomic load on the
// render thread and a store from whatever thread the driver runs its handler
// on, which is the only place the two ever meet.
#pragma once

#include <atomic>
#include <cstdint>

namespace gpe {

class CompletionReporting {
public:
    virtual ~CompletionReporting() = default;

    /// Every submission up to and including this one has finished on the
    /// device. Monotonic.
    [[nodiscard]] uint64_t completedSubmissions() const noexcept {
        return completed_.load(std::memory_order_acquire);
    }

    /// How long the device spent on the most recent submission it finished, in
    /// milliseconds. Zero from a backend that does not measure.
    ///
    /// Measured by the device, not by a clock on this side. The CPU clock round
    /// a dispatch measures how long it took to *queue* the work -- which on a
    /// pipeline that is working is a fraction of the work itself, and is the
    /// number that looks best exactly when the GPU is furthest behind.
    [[nodiscard]] double lastGpuMs() const noexcept {
        return gpuMs_.load(std::memory_order_relaxed);
    }

protected:
    /// Called from the driver's completion handler, on the driver's thread.
    /// Stores and returns; no locks, no allocation, nothing that could call
    /// back into the engine.
    void reportCompleted(uint64_t submission, double gpuMs = 0.0) noexcept {
        if (gpuMs > 0.0) {
            gpuMs_.store(gpuMs, std::memory_order_relaxed);
        }
        // Monotonic even if handlers arrive out of order, which they may:
        // two command buffers can finish in either order and the pool's rule is
        // "everything up to N", not "N".
        uint64_t seen = completed_.load(std::memory_order_relaxed);
        while (submission > seen &&
               !completed_.compare_exchange_weak(seen, submission,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed)) {
        }
    }

    std::atomic<uint64_t> completed_{0};
    std::atomic<double>   gpuMs_{0.0};
};

}   // namespace gpe
