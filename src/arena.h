// Copyright (c) 2026 gpe contributors.
//
// What frame() allocates from.
//
// The engine plays at a fixed rate, and that decides the shape of this file.
// `PooledDevice::alloc` can walk a free list, reclaim, synchronise, trim, ask
// its owner for memory and fail; every one of those is fine while a chain is
// being built and none of them belongs on a path with a deadline. So the
// reservation is split in two:
//
//   prepare()  works out what the chain needs at its peak, takes it from the
//              pool, and fails here -- before anything is played -- if it does
//              not fit.
//   frame()    hands out what prepare() already reserved. O(1), no driver call,
//              no synchronisation, and no failure path to write at the call
//              site because there is nothing left that can fail.
//
// WHY BUFFERS AND NOT A BUMP POINTER
//
// An arena usually means one big allocation and an offset into it. It cannot
// here: `gpe::Image` is {BufferId, w, h, stride} and has nowhere to put an
// offset, and adding one changes a type the interface fixed. So an arena is a
// set of whole buffers reserved up front and handed out in order. The property
// that matters -- allocation during a frame is a pointer bump and cannot fail
// -- is the same either way; what is given up is packing several small images
// into one allocation, which at one image per node and 32 MiB buckets is not
// where the memory goes.
//
// THREE SLOTS
//
// A frame's buffers cannot be reused until the GPU has finished reading them,
// so a single set would make the CPU wait for the GPU every frame and the two
// would take turns. Three sets let the CPU run two frames ahead. The wait is
// never on the driver: each slot remembers the submission it ended on, and
// beginning a frame waits for that number to retire -- the same watermark the
// pool already keeps.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pool.h"

namespace gpe {

/// One frame's worth of buffers, three deep.
class FrameArenas {
public:
    /// Two frames ahead of the GPU, which is what keeps both busy without
    /// letting the queue grow until latency is visible.
    static constexpr int kSlots = 3;

    explicit FrameArenas(PooledDevice& device) : device_(&device) {}
    ~FrameArenas();

    FrameArenas(const FrameArenas&) = delete;
    FrameArenas& operator=(const FrameArenas&) = delete;

    /// Reserves `count` images of `w` x `h` for every slot.
    ///
    /// Prepare-time, and the only place this can fail. Everything the chain
    /// will ask for during a frame has to be counted here: `image()` has no way
    /// to report running out and must not acquire one.
    ///
    /// Calling it again releases what was reserved and reserves afresh, so a
    /// chain that changes shape re-prepares rather than accumulating.
    [[nodiscard]] bool prepare(int w, int h, size_t count);

    /// Starts frame `index`, waiting for the slot it lands on to be free.
    ///
    /// This is the one wait in the system. It is not a device synchronisation:
    /// it is a compare against the submission watermark the completion handler
    /// publishes, and on a pipeline that is keeping up it does not wait at all.
    void begin(uint64_t index);

    /// The next reserved image. O(1) and cannot fail.
    ///
    /// Asking for more than prepare() reserved is a bug in the caller, not a
    /// condition to handle: it returns an empty Image and says so on stderr,
    /// because a chain that quietly renders one node less is worse than one
    /// that stops.
    [[nodiscard]] Image image();

    /// Ends the frame, recording what the slot has to outlive.
    void end();

    /// Where the pipeline is, for a caller deciding whether it is keeping up.
    [[nodiscard]] uint64_t frame() const noexcept { return frame_; }
    /// How many images a slot holds, and how many of them this frame has used.
    [[nodiscard]] size_t capacity() const noexcept { return perSlot_; }
    [[nodiscard]] size_t used() const noexcept { return handed_; }
    /// Frames that began without their slot having retired. Zero on a pipeline
    /// that is keeping up; anything else is the number to look at first.
    [[nodiscard]] uint64_t stalls() const noexcept { return stalls_; }

private:
    struct Slot {
        std::vector<Image> images;
        /// The submission this slot's work ended on. Beginning a frame on it
        /// again waits for this to retire and nothing else.
        Submission endedAt = 0;
    };

    void releaseAll();

    PooledDevice*                device_ = nullptr;
    Slot                         slots_[kSlots];
    size_t                       perSlot_ = 0;
    uint64_t                     frame_ = 0;
    int                          current_ = 0;
    size_t                       handed_ = 0;
    uint64_t                     stalls_ = 0;
};

}   // namespace gpe
