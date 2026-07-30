// Copyright (c) 2026 gpe contributors.
#include "player.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

namespace gpe {
namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

}   // namespace

bool Player::prepare(int w, int h, size_t imagesPerFrame) {
    if (!arenas_.prepare(w, h, imagesPerFrame)) {
        return false;
    }
    plateBytes_ = static_cast<size_t>(w) * static_cast<size_t>(h) * kBytesPerPixel;
    // One deeper than the frame pipeline, so the producer always has somewhere
    // to write while three slots are in flight.
    auto* staging =
        dynamic_cast<HostStaging*>(const_cast<Device*>(device_->nativeDevice()));
    decoded_ = std::make_unique<HostRing>(FrameArenas::kSlots + 1, plateBytes_,
                                          staging);
    return decoded_->count() == FrameArenas::kSlots + 1;
}

Player::Stats Player::run(uint64_t frames, double fps, const Decode& decode,
                          const Render& render) {
    Stats stats;
    if (frames == 0 || fps <= 0.0 || plateBytes_ == 0) {
        return stats;
    }

    const auto* reporter =
        dynamic_cast<const CompletionReporting*>(device_->nativeDevice());
    const Seconds period(1.0 / fps);
    const auto    start = Clock::now();
    std::vector<double> gpuMs;
    gpuMs.reserve(frames);

    const uint64_t stallsBefore = arenas_.stalls();

    for (uint64_t frame = 0; frame < frames;) {
        // Which frame the clock says is due. Worked out from the wall clock,
        // not by counting rounds of this loop -- counting is what makes a
        // player that is permanently a little late while every individual
        // frame looks fine.
        const auto elapsed = Seconds(Clock::now() - start);
        const auto due = static_cast<uint64_t>(elapsed / period);

        if (due > frame) {
            // The moment for these has gone. They are not shown, and the
            // player moves to the one that is due now -- which is the whole
            // rule: the timeline does not slip to accommodate them.
            const uint64_t late = std::min(due, frames) - frame;
            for (uint64_t i = 0; i < late; ++i) {
                (void)policy_->onDrop();
                ++stats.dropped;
            }
            frame = due;
            if (frame >= frames) {
                break;
            }
        }

        arenas_.begin(frame);
        const FrameArenas::FrameContext context = arenas_.context();

        const uint64_t retired =
            reporter != nullptr ? reporter->completedSubmissions() : 0;
        if (void* plate = decoded_->acquire(retired); plate != nullptr) {
            decode(plate, plateBytes_, frame);
            device_->upload(context.input.buf, plate, plateBytes_);
            decoded_->commit(device_->submission() + 1);
        } else {
            // Nothing decoded in time. The frame is still presented -- with
            // whatever the slot holds -- and counted, because a ring that runs
            // dry is the transfers being behind and is worth telling apart
            // from a kernel that is too slow.
            ++stats.starved;
        }

        render(context);
        arenas_.end();

        policy_->onFrame();
        ++stats.presented;
        if (reporter != nullptr) {
            if (const double ms = reporter->lastGpuMs(); ms > 0.0) {
                gpuMs.push_back(ms);
            }
        }

        ++frame;
        // Sleep until the next frame is due, if there is time. If there is not,
        // the loop goes straight round and the check at the top drops whatever
        // the clock has already passed.
        const auto next = start + std::chrono::duration_cast<Clock::duration>(
                                      period * frame);
        if (Clock::now() < next) {
            std::this_thread::sleep_until(next);
        }
    }

    // What the rule is measured by: where the last frame landed against where
    // the clock said it should. Dropping frames is how this stays near zero.
    const auto elapsed = Seconds(Clock::now() - start);
    stats.driftMs = (elapsed.count() - period.count() * static_cast<double>(frames)) *
                    1000.0;
    stats.stalls = arenas_.stalls() - stallsBefore;

    if (!gpuMs.empty()) {
        std::sort(gpuMs.begin(), gpuMs.end());
        stats.gpuP50Ms = gpuMs[gpuMs.size() / 2];
        stats.gpuP99Ms = gpuMs[static_cast<size_t>(gpuMs.size() * 0.99)];
        stats.gpuWorstMs = gpuMs.back();
    }
    return stats;
}

}   // namespace gpe
