// Three hundred frames at a fixed rate, on whichever GPU this machine has.
//
// This is the test the engine exists to pass. Everything else checks that a
// number came back right; this checks that it came back in time, three hundred
// times, without the CPU and the GPU taking turns.
//
// It also proves the one thing the roundtrip cannot: that the pool learns what
// has finished from the backend's completion handler and not from a `sync()`.
// A frame that synchronises would still produce correct pixels and would still
// pass every other test in this directory -- and would make fixed-rate playback
// impossible. So the count of synchronisations is asserted to be zero, and that
// assertion is most of the point of the file.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "arena.h"
#include "completion.h"
#include "gpe/args.h"
#include "gpe/device.h"
#include "pool.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

constexpr int    kFrames = 300;
constexpr double kTargetFps = 60.0;
constexpr int    kWidth = 1920;
constexpr int    kHeight = 1080;
/// Two images per frame: a chain of one operation, source and destination.
constexpr size_t kImagesPerFrame = 2;

}   // namespace

int main() {
    using namespace gpe;

    std::unique_ptr<Device> native = Device::create();
    if (native == nullptr) {
        std::puts("test_realtime: skipped, no device");
        return 0;
    }
    const char* which =
        native->backend() == Device::Backend::CUDA ? "CUDA" : "Metal";

    // Kept before the pool takes ownership: this is the backend's own timing,
    // and the pool deliberately knows nothing about it.
    const auto* reporter = dynamic_cast<const CompletionReporting*>(native.get());

    PooledDevice device(std::move(native), size_t{4} << 30);
    const KernelId scale = device.load("scale");
    check(scale != kInvalidKernel, "the scale kernel loads");
    if (scale == kInvalidKernel) {
        return 1;
    }

    FrameArenas arenas(device);
    // Everything the chain will ever need, taken now. If it does not fit, this
    // is where playback is refused -- not at frame 173.
    check(arenas.prepare(kWidth, kHeight, kImagesPerFrame),
          "the chain's peak reserves before playback starts");
    if (arenas.capacity() != kImagesPerFrame) {
        return 1;
    }

    // Warm up. The first dispatch of a kernel pays for the driver's JIT of the
    // PTX and for whatever Metal does to a pipeline the first time; measuring
    // that as a frame would report a stutter that only ever happens once.
    for (int i = 0; i < 3; ++i) {
        arenas.begin(static_cast<uint64_t>(i));
        const Image src = arenas.image();
        const Image dst = arenas.image();
        Args args;
        args.buffer(src.buf).buffer(dst.buf).uniforms(ScaleUniforms{
            kWidth, kHeight, kWidth, 1.5f});
        device.dispatch(scale, Grid{kWidth, kHeight, 1}, args.data(), args.size());
        arenas.end();
    }
    device.sync();

    const Submission afterWarmup = device.submission();
    const uint64_t   stallsAfterWarmup = arenas.stalls();

    // And now the measured run. Nothing in this loop is allowed to
    // synchronise: `begin` waits on the watermark if it has to, `image` is an
    // index, `dispatch` returns without waiting, and `end` records a number.
    std::vector<double> frameMs;
    std::vector<double> gpuMs;
    frameMs.reserve(kFrames);
    gpuMs.reserve(kFrames);
    const auto period = std::chrono::duration<double>(1.0 / kTargetFps);
    const auto start = Clock::now();
    int missed = 0;

    for (int f = 0; f < kFrames; ++f) {
        const auto deadline = start + std::chrono::duration_cast<Clock::duration>(
                                          period * (f + 1));
        const auto frameStart = Clock::now();

        arenas.begin(static_cast<uint64_t>(f) + 3);
        const Image src = arenas.image();
        const Image dst = arenas.image();
        Args args;
        args.buffer(src.buf).buffer(dst.buf).uniforms(ScaleUniforms{
            kWidth, kHeight, kWidth,
            1.0f + static_cast<float>(f % 8) * 0.125f});
        device.dispatch(scale, Grid{kWidth, kHeight, 1}, args.data(), args.size());
        arenas.end();

        const auto frameEnd = Clock::now();
        frameMs.push_back(Ms(frameEnd - frameStart).count());
        // What the device says it spent, read off whichever completion has
        // landed by now. Two frames behind, which is what a three-deep pipeline
        // means -- and still the only honest measure of the work.
        if (reporter != nullptr) {
            if (const double ms = reporter->lastGpuMs(); ms > 0.0) {
                gpuMs.push_back(ms);
            }
        }
        if (frameEnd > deadline) {
            ++missed;
        } else {
            std::this_thread::sleep_until(deadline);
        }
    }

    // The queue is allowed to be deep at the end; this is the one sync, and it
    // is outside the measured loop.
    device.sync();

    std::sort(frameMs.begin(), frameMs.end());
    const double p50 = frameMs[frameMs.size() / 2];
    const double p99 = frameMs[static_cast<size_t>(frameMs.size() * 0.99)];
    const double worst = frameMs.back();

    std::printf(
        "test_realtime: %s  %d frames at %.0f fps  missed %d  stalls %llu\n"
        "  cpu (queueing):  p50 %.3f ms  p99 %.3f ms  worst %.3f ms\n",
        which, kFrames, kTargetFps, missed,
        static_cast<unsigned long long>(arenas.stalls() - stallsAfterWarmup),
        p50, p99, worst);

    if (!gpuMs.empty()) {
        std::sort(gpuMs.begin(), gpuMs.end());
        std::printf(
            "  gpu (the work):  p50 %.3f ms  p99 %.3f ms  worst %.3f ms"
            "  (%zu samples)\n",
            gpuMs[gpuMs.size() / 2],
            gpuMs[static_cast<size_t>(gpuMs.size() * 0.99)], gpuMs.back(),
            gpuMs.size());
        // The budget is the frame period, and it is the GPU that has to fit in
        // it. A CPU-side number that fits while the device does not is exactly
        // the mistake this measurement exists to prevent.
        check(gpuMs[static_cast<size_t>(gpuMs.size() * 0.99)] <
                  1000.0 / kTargetFps,
              "the device finishes a frame inside the frame period");
    } else {
        std::puts("  gpu: not measured by this backend");
    }

    check(missed == 0, "no frame missed its deadline");

    // The pool learned what had finished from the backend, not from a wait.
    //
    // Three slots and 300 frames means every buffer is reused a hundred times,
    // and each reuse needs its submission to have retired. If the completion
    // handler were not running, `begin` would have fallen back to `sync()` --
    // which is what `stalls` counts.
    check(arenas.stalls() == stallsAfterWarmup,
          "and not one frame stalled waiting for a slot");
    check(device.retired(afterWarmup),
          "the completion handler advanced the watermark");

    // The work really was queued: 300 dispatches, plus the warm-up.
    check(device.submission() >= afterWarmup + kFrames,
          "three hundred dispatches were submitted");

    if (failures == 0) {
        std::puts("test_realtime: ok");
    }
    return failures == 0 ? 0 : 1;
}
