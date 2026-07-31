// The upload that is not there.
//
// A 4K float4 plate is 133 MB. Uploading it to a private device buffer is the
// whole cost of the output path, and halving the bytes with half float halves
// the cost. On a machine with unified memory both of those are answers to a
// question that does not need asking: the CPU and the GPU address the same
// pages, so a picture written in the right place has already arrived.
//
// This measures the three, in the units that matter: milliseconds against a
// frame.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "display.h"
#include "gpe/args.h"
#include "gpe/device.h"
#include "gpe/presenter.h"
#include "hostring.h"
#include "pool.h"

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

constexpr int kW = 3840;
constexpr int kH = 2160;
constexpr int kRounds = 20;

class Sink final : public gpe::Presenter {
public:
    void targetSize(int& w, int& h) const override {
        w = 1920;
        h = 1080;
    }
    void present(const void*, int, int, uint64_t) override { ++frames_; }
    uint64_t frames_ = 0;
};

}   // namespace

int main() {
    using namespace gpe;

    std::unique_ptr<Device> native = Device::create();
    if (native == nullptr) {
        std::puts("bench_shared: skipped, no device");
        return 0;
    }
    const bool metal = native->backend() == Device::Backend::Metal;
    auto* staging = dynamic_cast<HostStaging*>(native.get());

    PooledDevice device(std::move(native), size_t{4} << 30);
    DisplayPass  pass(device);
    if (!pass.prepare(1920, 1080)) {
        std::puts("bench_shared: could not prepare");
        return 1;
    }
    Sink sink;

    const size_t       bytes = size_t{kW} * kH * 16;
    std::vector<float> plate(size_t{kW} * kH * 4, 0.5f);

    // --- what the viewer does: a private buffer and an upload --------------
    const BufferId privateBuf = device.alloc(bytes);
    device.upload(privateBuf, plate.data(), bytes);
    device.sync();

    auto start = Clock::now();
    for (int i = 0; i < kRounds; ++i) {
        device.upload(privateBuf, plate.data(), bytes);
        pass.present(Image{privateBuf, kW, kH, kW}, sink, DisplayControls{},
                     static_cast<uint64_t>(i));
    }
    device.sync();
    const double uploaded = Ms(Clock::now() - start).count() / kRounds;

    // --- shared: the CPU writes where the GPU already reads ----------------
    double shared = -1.0;
    (void)staging;
    {
        BufferId sharedBuf = kInvalidBuffer;
        void*    memory = device.allocShared(bytes, sharedBuf);
        if (memory != nullptr && sharedBuf != kInvalidBuffer) {
            std::memcpy(memory, plate.data(), bytes);
            device.sync();
            start = Clock::now();
            for (int i = 0; i < kRounds; ++i) {
                // The producer writing its frame. This is the work that
                // replaces the upload, and it is counted.
                std::memcpy(memory, plate.data(), bytes);
                pass.present(Image{sharedBuf, kW, kH, kW}, sink,
                             DisplayControls{}, static_cast<uint64_t>(i));
            }
            device.sync();
            shared = Ms(Clock::now() - start).count() / kRounds;
            device.release(sharedBuf);
        }
    }
    device.release(privateBuf);
    device.sync();

    std::printf("\n%s, a %dx%d float4 plate to a 1920x1080 viewer, mean of %d:\n",
                metal ? "Metal" : "CUDA", kW, kH, kRounds);
    std::printf("  private + upload    %7.2f ms   %.0f MB copied\n", uploaded,
                static_cast<double>(bytes) / 1e6);
    if (shared >= 0.0) {
        std::printf("  shared, no upload   %7.2f ms   %.0f MB copied\n", shared,
                    static_cast<double>(bytes) / 1e6);
        std::printf("    (the memcpy is the producer writing its frame, which\n"
                    "     it has to do somewhere either way)\n");
    } else {
        std::printf("  shared              not offered by this backend\n");
    }
    // --- and shared with the producer writing in place ---------------------
    //
    // The measurement above still copies 133 MB, because it models a producer
    // that made its frame somewhere else. The design is that it makes it here:
    // the engine renders into the buffer the GPU already reads, and then there
    // is no copy at any point in the path.
    double inPlace = -1.0;
    {
        BufferId sharedBuf = kInvalidBuffer;
        if (void* memory = device.allocShared(bytes, sharedBuf);
            memory != nullptr && sharedBuf != kInvalidBuffer) {
            auto* pixels = static_cast<float*>(memory);
            device.sync();
            start = Clock::now();
            for (int i = 0; i < kRounds; ++i) {
                // A producer touching its output rather than copying into it.
                // One value a frame, so this measures the path and not a
                // synthetic workload.
                pixels[0] = static_cast<float>(i);
                pass.present(Image{sharedBuf, kW, kH, kW}, sink,
                             DisplayControls{}, static_cast<uint64_t>(i));
            }
            device.sync();
            inPlace = Ms(Clock::now() - start).count() / kRounds;
            device.release(sharedBuf);
        }
    }
    if (inPlace >= 0.0) {
        std::printf("  shared, in place    %7.2f ms   0 MB copied\n", inPlace);
        std::printf("    (the producer renders into the buffer the GPU reads;\n"
                    "     this is the display pass and nothing else)\n");
    }
    std::printf("  budget at 60 fps    %7.2f ms\n\n", 1000.0 / 60.0);
    // Flushed before anything can go wrong afterwards. printf to a pipe is
    // fully buffered, so a crash at shutdown eats the measurement -- which is
    // how the first version of this looked like a crash with no output at all.
    std::fflush(stdout);
    return 0;
}
