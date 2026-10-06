// Lanes: two queues of work on one device, from two threads.
//
// What can go wrong, each checked by name:
//
//   - The same kernel on two lanes at once with different arguments. On CUDA
//     a kernel's buffers and uniform pointer live in one __constant__ block per
//     module, read when the launch runs: with one module shared by two streams
//     each launch reads whichever write landed last. A lane has its own copy
//     of every module it runs; take that away and this test sees one lane's
//     numbers in the other's buffer.
//   - A buffer written on one lane and read on another. With the writer's
//     fence waited for, the reader sees the write; without, the pool waits
//     for it (the fallback in gpe/lane.h), and the reader still does.
//   - A buffer released while another lane's work on it is queued is not
//     handed out again until that lane has finished.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "gpe/args.h"
#include "gpe/device.h"
#include "gpe/lane.h"
#include "gpe/pool.h"

namespace {

std::atomic<int> failures{0};

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        failures.fetch_add(1);
    }
}

struct IotaUniforms {
    uint32_t count = 0;
    uint32_t first = 0;
};

constexpr uint32_t kCount = 1 << 16;

}   // namespace

int main() {
    using namespace gpe;
    std::unique_ptr<Device> native = Device::create();
    if (native == nullptr) {
        std::puts("test_lanes: skipped, no device");
        return 0;
    }
    PooledDevice device(std::move(native), size_t{512} << 20);
    const uint32_t lanes = device.openLanes(2);
    check(lanes == 2, "two lanes open");
    if (lanes < 2) {
        std::puts("test_lanes: this backend has one lane");
        return failures == 0 ? 0 : 1;
    }
    const KernelId iota = device.load("iota");
    check(iota != kInvalidKernel, "iota loads");

    const auto run = [&](BufferId into, uint32_t first) {
        Args args;
        args.buffer(into, sizeof(uint32_t)).uniforms(IotaUniforms{kCount, first});
        device.dispatch(iota, Grid{kCount, 1, 1}, args.data(), args.size());
    };

    // 1. The same kernel, both lanes, at once, different arguments.
    {
        constexpr int kRounds = 1000;
        const auto lane = [&](LaneId which) {
            const LaneScope bound(which);
            const BufferId mine = device.alloc(kCount * sizeof(uint32_t));
            std::vector<uint32_t> out(kCount, 0);
            int wrong = 0;
            for (int k = 1; k <= kRounds; ++k) {
                const uint32_t first = which * 10'000'000u + static_cast<uint32_t>(k) * 7u;
                run(mine, first);
                if (k % 50 == 0) {
                    device.download(out.data(), mine, out.size() * sizeof(uint32_t));
                    if (out[0] != first || out[kCount - 1] != first + kCount - 1) {
                        ++wrong;
                    }
                }
            }
            check(wrong == 0, "lane " + std::to_string(which) + " read back its own numbers (" +
                                  std::to_string(wrong) + " of " +
                                  std::to_string(kRounds / 50) + " wrong)");
            device.release(mine);
        };
        const auto start = std::chrono::steady_clock::now();
        std::thread a(lane, 0);
        std::thread b(lane, 1);
        a.join();
        b.join();
        std::printf("test_lanes: %d dispatches on each of two lanes in %.1f ms\n", kRounds,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              start)
                        .count());
    }

    // 2. Written on lane 0, read on lane 1 -- once after waiting on the
    //    writer's fence, once without, which the pool must settle itself.
    {
        const BufferId shared = device.alloc(kCount * sizeof(uint32_t));
        Fence written;
        {
            const LaneScope bound(0);
            run(shared, 777);
            device.flush();
            written = device.fence();
        }
        check(written.lane == 0 && written.value > 0, "a fence names its lane");
        std::vector<uint32_t> out(kCount, 0);
        std::thread reader([&] {
            const LaneScope bound(1);
            check(device.waitRetired(written, std::chrono::seconds(5)),
                  "lane 1 sees lane 0's fence retire");
            device.download(out.data(), shared, out.size() * sizeof(uint32_t));
        });
        reader.join();
        check(out[5] == 782, "lane 1 reads what lane 0 wrote, after its fence");

        {
            const LaneScope bound(0);
            for (int k = 0; k < 20; ++k) {
                run(shared, 1000 + static_cast<uint32_t>(k));
            }
            // No flush, no fence: the pool has to find lane 0's work itself.
        }
        std::thread unwaited([&] {
            const LaneScope bound(1);
            device.download(out.data(), shared, out.size() * sizeof(uint32_t));
        });
        unwaited.join();
        check(out[0] == 1019, "lane 1 reads lane 0's last write without waiting for it");
        device.release(shared);
    }

    // 3. Released with work still queued on another lane: not handed out
    //    again until that lane is done with it.
    {
        BufferId busy = kInvalidBuffer;
        {
            const LaneScope bound(1);
            busy = device.alloc(kCount * sizeof(uint32_t));
            for (int k = 0; k < 200; ++k) {
                run(busy, static_cast<uint32_t>(k));
            }
            device.release(busy);
        }
        const PooledDevice::Stats held = device.stats();
        {
            const LaneScope bound(0);
            const BufferId next = device.alloc(kCount * sizeof(uint32_t));
            const bool lane1Done = device.retired(Fence{1, device.lanes() > 1 ? 1u : 0u});
            check(next != busy || lane1Done,
                  "a buffer another lane is still using is not handed out again");
            device.release(next);
        }
        std::printf("test_lanes: %zu bytes waiting on a lane when released\n", held.bytesPending);
    }

    device.sync();
    if (failures == 0) {
        std::puts("test_lanes: ok");
    }
    return failures == 0 ? 0 : 1;
}
