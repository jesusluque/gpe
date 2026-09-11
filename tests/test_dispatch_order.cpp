// Dispatches queued together still run in the order they were asked for, and
// in order with the transfers around them.
//
// Written for the Metal backend's batching (several dispatches in one command
// buffer), and true of every backend: a later dispatch wins over an earlier
// one, an upload issued after a dispatch lands after it, a fill after it too,
// a download sees all of it, and `waitFor` on a queued submission returns
// once that work has run rather than after a spin and a full sync. Also
// prints what a small dispatch costs, which is what batching is for.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "gpe/args.h"
#include "gpe/device.h"
#include "gpe/pool.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct IotaUniforms {
    uint32_t count = 0;
    uint32_t first = 0;
};

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}   // namespace

int main() {
    using namespace gpe;
    std::unique_ptr<Device> native = Device::create();
    if (native == nullptr) {
        std::puts("test_dispatch_order: skipped, no device");
        return 0;
    }
    PooledDevice device(std::move(native), size_t{256} << 20);
    const KernelId iota = device.load("iota");
    check(iota != kInvalidKernel, "iota loads");

    constexpr uint32_t kCount = 4096;
    const BufferId values = device.alloc(kCount * sizeof(uint32_t));
    std::vector<uint32_t> out(kCount, 0);
    const auto run = [&](uint32_t first) {
        Args args;
        args.buffer(values, sizeof(uint32_t)).uniforms(IotaUniforms{kCount, first});
        device.dispatch(iota, Grid{kCount, 1, 1}, args.data(), args.size());
    };

    // A later dispatch wins.
    for (uint32_t k = 1; k <= 50; ++k) {
        run(k * 1000);
    }
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    check(out[0] == 50000 && out[kCount - 1] == 50000 + kCount - 1,
          "fifty dispatches without a sync: the download sees the last");

    // An upload after a dispatch lands after it.
    run(7);
    std::vector<uint32_t> uploaded(kCount, 0xABCD);
    device.upload(values, uploaded.data(), uploaded.size() * sizeof(uint32_t));
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    check(out[0] == 0xABCD && out[kCount - 1] == 0xABCD, "an upload after a dispatch is not overwritten by it");

    // And a dispatch after an upload reads it... and overwrites it.
    run(9);
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    check(out[5] == 14, "a dispatch after an upload runs after it");

    // A fill after a dispatch lands after it.
    run(11);
    check(device.fill(values, 0, kCount * sizeof(uint32_t)), "fill is available");
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    check(out[0] == 0 && out[kCount - 1] == 0, "a fill after a dispatch is not overwritten by it");

    // waitFor a queued dispatch: prompt, not a spin and a sync.
    run(13);
    const Submission at = device.submission();
    const auto waitStart = Clock::now();
    device.waitFor(at);
    const double waited = msSince(waitStart);
    check(device.retired(at), "waitFor returns with the submission retired");
    std::printf("test_dispatch_order: waitFor on a queued dispatch took %.2f ms\n", waited);

    // What a small dispatch costs, counted over many.
    constexpr int kDispatches = 2000;
    device.sync();
    const auto start = Clock::now();
    for (int k = 0; k < kDispatches; ++k) {
        run(static_cast<uint32_t>(k));
    }
    const double queued = msSince(start);
    device.sync();
    const double total = msSince(start);
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    check(out[3] == kDispatches - 1 + 3, "the timed run wrote its last dispatch");
    std::printf("test_dispatch_order: %d dispatches in %.1f ms (%.1f ms queuing them), %.4f ms each\n",
                kDispatches, total, queued, total / kDispatches);

    device.release(values);
    device.sync();
    if (failures == 0) {
        std::puts("test_dispatch_order: ok");
    }
    return failures == 0 ? 0 : 1;
}
