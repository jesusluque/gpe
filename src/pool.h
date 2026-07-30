// Copyright (c) 2026 gpe contributors.
//
// The pool, as a decorator.
//
// `PooledDevice` is a `Device` that wraps another `Device`. The backends know
// nothing about any of this: their `alloc` is a `cudaMalloc` or a
// `newBufferWithLength` and nothing else, which keeps the part that has to be
// written twice as small as it can be and the part that is easy to get wrong
// written once.
//
// Everything here that is not one of the eight is a plain non-virtual method.
// Trim, statistics and completion reporting happen when a chain is built or
// when a submission retires, and neither wants a virtual call or a place in
// the interface.
//
// WHEN EACH CALL IS ALLOWED
//
// `alloc` belongs to prepare(), not to frame(). It can walk a free list, it can
// reclaim, it can synchronise, it can trim, it can ask its owner to give memory
// back, and it can fail -- and a call that can do any of those has no business
// on a path with a frame deadline. Playback at a fixed rate is the point of the
// engine; an API that reads well and misses a frame is the wrong trade.
//
// So: the chain reserves what it needs through `alloc` while it is being built
// or changed, and if the peak does not fit, prepare() fails then, before
// anything is played. During frame() the reservation is O(1) out of arenas that
// already exist, cannot fail, and never synchronises. Arenas are the next piece
// and are built on this one; what this file has to get right for them is the
// submission watermark, which is how "slot S has finished" is answered without
// asking the driver.
//
// `sync()` is likewise a prepare-time call. Inside a frame it is forbidden.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "gpe/device.h"

namespace gpe {

/// A submission is one dispatch, counted. The pool never asks the backend what
/// has finished; the backend tells it, and until it does a released buffer is
/// not reusable however idle the device looks.
using Submission = uint64_t;

class PooledDevice final : public Device {
public:
    /// What the pool did, for the caller that wants to know why a frame was
    /// slow. Cheap to read: these are counters, not a walk of anything.
    struct Stats {
        size_t   bytesHeld = 0;      ///< allocated from the driver, in total
        size_t   bytesInUse = 0;     ///< handed out and not yet released
        size_t   bytesPending = 0;   ///< released, waiting on a submission
        size_t   liveBuffers = 0;
        uint64_t hits = 0;           ///< allocs served from a free list
        uint64_t misses = 0;         ///< allocs that had to reach the driver
        uint64_t reclaimed = 0;      ///< buffers returned by a completed submission
    };

    /// Takes ownership of `native`.
    ///
    /// `budgetBytes` is the pool's own ceiling, not the driver's. On Apple
    /// silicon there is no clean allocation failure to detect -- unified memory
    /// pages and the machine degrades quietly -- so the limit has to be one we
    /// impose and can therefore test identically on both backends.
    ///
    /// `onPressure` is the last thing tried before giving up: the pool asks its
    /// owner to release something, and the owner answers whether it did. The
    /// pool knows nothing about an image cache and must not.
    PooledDevice(std::unique_ptr<Device> native, size_t budgetBytes,
                 std::function<bool(size_t)> onPressure = {});
    ~PooledDevice() override;

    // --- the eight ---------------------------------------------------------

    [[nodiscard]] Backend backend() const override;
    [[nodiscard]] BufferId alloc(size_t bytes) override;
    void release(BufferId) override;
    void upload(BufferId, const void* src, size_t) override;
    void download(void* dst, BufferId, size_t) override;
    [[nodiscard]] KernelId load(std::string_view name) override;
    void dispatch(KernelId, Grid grid, const void* args, size_t) override;
    void sync() override;

    // --- everything else ---------------------------------------------------

    /// Gives every free buffer back to the driver. Never called on the frame
    /// path -- it is the opposite of what the pool is for.
    void trim();

    [[nodiscard]] Stats stats() const noexcept { return stats_; }

    /// The backend's completion handler calls this, from whatever thread the
    /// driver runs handlers on. It publishes a number and does nothing else:
    /// no locks, no allocation, no call back into the pool.
    void notifyCompleted(Submission done) noexcept {
        completed_.store(done, std::memory_order_release);
    }

    /// The submission a dispatch queued now would belong to.
    [[nodiscard]] Submission submission() const noexcept { return submitted_; }

    /// Has everything up to `at` finished?
    ///
    /// A read of the watermark and nothing else -- no driver call, so it is
    /// safe to ask inside a frame and cheap enough to ask every frame.
    [[nodiscard]] bool retired(Submission at) const noexcept {
        return completed_.load(std::memory_order_acquire) >= at;
    }

    /// Blocks until `retired(at)`.
    ///
    /// This is the stall path: on a pipeline that is keeping up nobody calls
    /// it, and calling it means a deadline has already been missed.
    ///
    /// A limitation worth naming rather than hiding: waiting for one slot needs
    /// a fence or an event, and the eight-method interface has neither -- it
    /// has `sync()`, which waits for everything. So this waits on the number a
    /// backend's completion handler publishes, which costs nothing and is
    /// exactly per-slot; and if a backend publishes nothing it falls back to
    /// `sync()` after a bounded spin, which is correct and blunt. A backend
    /// that wants the sharp version calls notifyCompleted from its handler.
    void waitFor(Submission at);

    /// The native handle behind a pooled one, for a backend that has to reach
    /// the real buffer -- and for the test that checks reuse really is reuse.
    /// `kInvalidBuffer` if the handle is stale.
    [[nodiscard]] BufferId nativeHandle(BufferId) const noexcept;

    /// Moves retired buffers from the pending queue to the free lists. Called
    /// on the way into `alloc`; exposed because a test needs to drive it.
    void reclaim();

private:
    struct Slot {
        BufferId native = kInvalidBuffer;
        size_t   bucketBytes = 0;
        uint32_t generation = 1;   ///< odd while live, even while free
        bool     live = false;
    };

    struct Retiring {
        uint32_t   slot = 0;
        Submission at = 0;
    };

    [[nodiscard]] static int bucketFor(size_t bytes) noexcept;
    [[nodiscard]] static size_t bucketBytes(int bucket) noexcept;

    /// The slot behind a handle, or null if the generation does not match.
    [[nodiscard]] Slot* resolve(BufferId) noexcept;
    [[nodiscard]] const Slot* resolve(BufferId) const noexcept;

    /// Everything short of failing, in order, until `bytes` will fit.
    [[nodiscard]] bool makeRoom(size_t bytes);

    std::unique_ptr<Device>     native_;
    size_t                      budget_ = 0;
    std::function<bool(size_t)> onPressure_;

    std::vector<Slot>                  slots_;
    std::vector<std::vector<uint32_t>> freeByBucket_;
    std::deque<Retiring>               retiring_;
    std::vector<uint32_t>              recycledSlots_;   ///< slot indices to reuse

    /// Written by the render thread, read by it too. The completion thread only
    /// ever writes `completed_`.
    Submission              submitted_ = 0;
    std::atomic<Submission> completed_{0};

    Stats stats_{};
};

}   // namespace gpe
