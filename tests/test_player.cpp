// The rule the player exists to keep: frames are dropped, the timeline is not.
//
// A player that waits for a late frame makes the next one late too, and the
// error accumulates until the picture is visibly behind the sound. So the
// interesting case is not the one where everything is fast -- it is the one
// where a frame takes too long, and what has to survive that is the clock.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "drop.h"
#include "gpe/args.h"
#include "gpe/device.h"
#include "player.h"
#include "pool.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int    kWidth = 960;
constexpr int    kHeight = 540;
constexpr double kFps = 60.0;

}   // namespace

int main() {
    using namespace gpe;

    std::unique_ptr<Device> native = Device::create();
    if (native == nullptr) {
        std::puts("test_player: skipped, no device");
        return 0;
    }
    const char* which =
        native->backend() == Device::Backend::CUDA ? "CUDA" : "Metal";
    PooledDevice device(std::move(native), size_t{2} << 30);

    const KernelId scale = device.load("scale");
    check(scale != kInvalidKernel, "the scale kernel loads");
    if (scale == kInvalidKernel) {
        return 1;
    }

    const auto render = [&](const FrameArenas::FrameContext& frame) {
        Args args;
        args.buffer(frame.input.buf)
            .buffer(frame.output.buf)
            .uniforms(ScaleUniforms{static_cast<uint32_t>(kWidth),
                                    static_cast<uint32_t>(kHeight),
                                    static_cast<uint32_t>(kWidth), 2.0f});
        device.dispatch(scale,
                        Grid{static_cast<uint32_t>(kWidth),
                             static_cast<uint32_t>(kHeight), 1},
                        args.data(), args.size());
    };

    // --- nothing is late: nothing is dropped -------------------------------
    {
        DropPolicy policy;
        Player     player(device, policy);
        check(player.prepare(kWidth, kHeight, 2), "the chain prepares");

        const Player::Stats stats = player.run(
            180, kFps,
            [](void* into, size_t bytes, uint64_t frame) {
                std::memset(into, static_cast<int>(frame % 251), bytes);
            },
            render);
        device.sync();

        std::printf("test_player: %s  180 frames  presented %llu  dropped %llu"
                    "  starved %llu  drift %+.2f ms  gpu p99 %.3f ms\n",
                    which, static_cast<unsigned long long>(stats.presented),
                    static_cast<unsigned long long>(stats.dropped),
                    static_cast<unsigned long long>(stats.starved),
                    stats.driftMs, stats.gpuP99Ms);

        check(stats.presented == 180, "every frame was presented");
        check(stats.dropped == 0, "and none was dropped");
        check(policy.drops() == 0, "the policy agrees");
        // Within a frame period. A player that finished early would have
        // slipped the other way, which is just as wrong.
        check(std::abs(stats.driftMs) < 1000.0 / kFps,
              "the timeline landed where the clock said");
    }

    // --- a slow frame is dropped, and the timeline still lands -------------
    {
        // The case the rule is for. Every twentieth decode takes three frame
        // periods, which cannot be absorbed -- the player has to give those
        // frames up rather than let everything after them slide.
        DropPolicy policy;
        Player     player(device, policy);
        check(player.prepare(kWidth, kHeight, 2), "the chain prepares again");

        const Player::Stats stats = player.run(
            180, kFps,
            [](void* into, size_t bytes, uint64_t frame) {
                std::memset(into, static_cast<int>(frame % 251), bytes);
                // Not on the last frames. A stall on the final frame cannot
                // be recovered from -- there is nothing after it to drop --
                // so including one would be measuring an unfixable thing and
                // calling it drift. What is being tested is that the player
                // gets *back* on the clock, and for that it needs frames left
                // to do it in.
                if (frame % 20 == 19 && frame < 160) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(50000));
                }
            },
            render);
        device.sync();

        std::printf("  with a stall every 20th frame: presented %llu  "
                    "dropped %llu  drift %+.2f ms\n",
                    static_cast<unsigned long long>(stats.presented),
                    static_cast<unsigned long long>(stats.dropped),
                    stats.driftMs);

        check(stats.dropped > 0, "frames were given up");
        check(stats.presented + stats.dropped == 180,
              "and every frame is accounted for, shown or dropped");
        check(policy.drops() == stats.dropped,
              "the policy saw exactly the ones that were dropped");

        // The whole point. Nine stalls of 50 ms is 450 ms of lateness that a
        // player which waited would still be carrying at the end; this one
        // finishes where the clock said it would.
        check(std::abs(stats.driftMs) < 1000.0 / kFps,
              "the timeline did not slip, however many frames went");
    }

    if (failures == 0) {
        std::puts("test_player: ok");
    }
    return failures == 0 ? 0 : 1;
}
