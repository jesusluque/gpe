// The whole road, on whichever GPU this machine has.
//
// Nothing in this file names a backend. That is the point of it: the same
// source builds and passes on an M-series Mac through Metal and on an NVIDIA
// card through CUDA, and if it ever stops doing so on one of them, the thing
// that broke is the abstraction and not the arithmetic.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

constexpr int    kWidth = 1920;
constexpr int    kHeight = 1080;
constexpr size_t kPixels = size_t{kWidth} * size_t{kHeight};
constexpr size_t kFloats = kPixels * 4;
constexpr size_t kBytes = kFloats * sizeof(float);

}   // namespace

int main() {
    using namespace gpe;

    std::unique_ptr<Device> native = Device::create();
    if (native == nullptr) {
        // A machine with no GPU is not a failing test, it is a machine that
        // cannot run this one. Saying so beats a red bar nobody can act on.
        std::puts("test_roundtrip: skipped, no device");
        return 0;
    }
    const char* which = native->backend() == Device::Backend::CUDA ? "CUDA" : "Metal";
    std::printf("test_roundtrip: %s\n", which);

    // Two gigabytes: room for the plates below with nothing near a limit, and
    // the number is ours rather than the driver's on purpose.
    PooledDevice device(std::move(native), size_t{2} << 30);

    const KernelId scale = device.load("scale");
    check(scale != kInvalidKernel, "the scale kernel loads");
    if (scale == kInvalidKernel) {
        return 1;
    }

    const BufferId src = device.alloc(kBytes);
    const BufferId dst = device.alloc(kBytes);
    check(src != kInvalidBuffer && dst != kInvalidBuffer,
          "two 1920x1080 float4 buffers allocate");

    // A pattern that is different in every channel and every pixel, so a kernel
    // that reads the wrong index or swaps a channel produces a wrong number
    // rather than the right one by luck.
    std::vector<float> pattern(kFloats);
    for (size_t i = 0; i < kPixels; ++i) {
        pattern[i * 4 + 0] = static_cast<float>(i % 251) * 0.5f;
        pattern[i * 4 + 1] = static_cast<float>(i % 253) * 0.25f;
        pattern[i * 4 + 2] = static_cast<float>(i % 257) * 0.125f;
        pattern[i * 4 + 3] = static_cast<float>(i % 259) * 0.0625f;
    }
    device.upload(src, pattern.data(), kBytes);

    Args args;
    args.buffer(src).buffer(dst).uniforms(ScaleUniforms{
        static_cast<uint32_t>(kWidth), static_cast<uint32_t>(kHeight),
        static_cast<uint32_t>(kWidth), 2.0f});
    device.dispatch(scale, Grid{static_cast<uint32_t>(kWidth),
                                static_cast<uint32_t>(kHeight), 1},
                    args.data(), args.size());

    std::vector<float> result(kFloats, -1.0f);
    device.download(result.data(), dst, kBytes);

    // Exactly double, not nearly. Multiplying by two is exact in binary
    // floating point for every value that is not already at the top of the
    // range, so a tolerance here would only hide a kernel that is doing
    // something else -- a filter tap, a wrong stride, a half-precision path.
    size_t wrong = 0;
    size_t firstWrong = 0;
    for (size_t i = 0; i < kFloats; ++i) {
        if (result[i] != pattern[i] * 2.0f) {
            if (wrong == 0) {
                firstWrong = i;
            }
            ++wrong;
        }
    }
    if (wrong > 0) {
        std::fprintf(stderr,
                     "  %zu of %zu values wrong; first at %zu: %.9g, wanted %.9g\n",
                     wrong, kFloats, firstWrong, result[firstWrong],
                     pattern[firstWrong] * 2.0f);
    }
    check(wrong == 0, "every pixel is exactly doubled");

    // --- upload and compute are on different queues, and still ordered ------
    //
    // One upload followed by one dispatch barely tests the barrier between
    // them: the copy has every chance to finish while the launch is still
    // being set up. This alternates them without a sync in between, so the
    // kernel is reading a buffer whose transfer was issued moments earlier on
    // another queue. With the barrier missing it reads whatever arrived so far
    // -- torn, and differently torn each run.
    for (int round = 1; round <= 8; ++round) {
        const float fill = static_cast<float>(round);
        std::fill(pattern.begin(), pattern.end(), fill);
        device.upload(src, pattern.data(), kBytes);
        Args round_args;
        round_args.buffer(src).buffer(dst).uniforms(ScaleUniforms{
            static_cast<uint32_t>(kWidth), static_cast<uint32_t>(kHeight),
            static_cast<uint32_t>(kWidth), 3.0f});
        device.dispatch(scale, Grid{static_cast<uint32_t>(kWidth),
                                    static_cast<uint32_t>(kHeight), 1},
                        round_args.data(), round_args.size());
    }
    device.sync();
    device.download(result.data(), dst, kBytes);

    size_t torn = 0;
    for (size_t i = 0; i < kFloats; ++i) {
        if (result[i] != 24.0f) {   // the eighth round: 8 * 3
            ++torn;
        }
    }
    if (torn > 0) {
        std::fprintf(stderr, "  %zu of %zu values are not from the last upload\n",
                     torn, kFloats);
    }
    check(torn == 0,
          "a kernel never reads a buffer whose upload is still in flight");

    // --- and the pool really pools -----------------------------------------
    const BufferId nativeSrc = device.nativeHandle(src);
    device.release(src);
    device.sync();   // prepare-time, so a full wait is allowed here
    const BufferId again = device.alloc(kBytes);
    check(device.nativeHandle(again) == nativeSrc,
          "release then allocate gives the same native buffer back");
    check(device.stats().hits >= 1, "and the pool counted it as a hit");

    device.release(dst);
    device.release(again);
    device.sync();

    if (failures == 0) {
        std::printf("test_roundtrip: ok (%s, %zu pixels)\n", which, kPixels);
    }
    return failures == 0 ? 0 : 1;
}
