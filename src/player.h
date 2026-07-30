// Copyright (c) 2026 gpe contributors.
//
// The playback loop, and the one rule it exists to keep.
//
//   The clock is the truth. Frames are dropped; the timeline never slips.
//
// Those are not two statements. A player that waits for a late frame makes the
// next one late too, and the error accumulates: after a few hundred frames the
// picture is visibly behind the sound and nothing later can recover it, because
// the time it should have been shown at has gone. So when the wall clock says
// the moment for frame N has passed, frame N is not shown -- the player moves
// to whichever frame is due now and carries on. What the viewer sees instead is
// DropPolicy's business, and drift stays at zero.
//
// This is why `run()` works out which frame is due from the clock rather than
// counting the loop round. Counting is what makes a player that gets slowly,
// permanently late while every individual frame looks fine.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "arena.h"
#include "drop.h"
#include "hostring.h"
#include "pool.h"

namespace gpe {

class Player {
public:
    struct Stats {
        uint64_t presented = 0;   ///< frames rendered and shown
        uint64_t dropped = 0;     ///< frames the clock had already passed
        uint64_t starved = 0;     ///< frames the decode ring had nothing for
        uint64_t stalls = 0;      ///< frames that waited on a slot
        /// How far the last frame's presentation was from where the clock said
        /// it should be. The number the rule is about: it stays near zero
        /// however many frames were dropped getting there.
        double   driftMs = 0.0;
        double   gpuP50Ms = 0.0;
        double   gpuP99Ms = 0.0;
        double   gpuWorstMs = 0.0;
    };

    /// What a chain does with one frame. Called with views over the slot's
    /// arena; it may dispatch as much as it likes and must not synchronise.
    using Render = std::function<void(const FrameArenas::FrameContext&)>;
    /// Fills a host slot with the next picture. Stands in for a decoder; it is
    /// handed memory and an index, never the device.
    using Decode = std::function<void(void* into, size_t bytes, uint64_t frame)>;

    Player(PooledDevice& device, DropPolicy& policy)
        : device_(&device), policy_(&policy), arenas_(device) {}

    /// Reserves everything, and fails here if it does not fit.
    ///
    /// After this returns true, nothing in `run` allocates and nothing in `run`
    /// can fail for want of memory.
    [[nodiscard]] bool prepare(int w, int h, size_t imagesPerFrame);

    /// Plays `frames` frames at `fps`.
    [[nodiscard]] Stats run(uint64_t frames, double fps, const Decode& decode,
                            const Render& render);

private:
    PooledDevice*             device_ = nullptr;
    DropPolicy*               policy_ = nullptr;
    FrameArenas               arenas_;
    std::unique_ptr<HostRing> decoded_;
    size_t                    plateBytes_ = 0;
};

}   // namespace gpe
