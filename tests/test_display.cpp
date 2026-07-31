// The display pass: what crosses to the CPU, and whether it is right.
//
// Two things are worth checking and neither is "it produced pixels". That the
// downsample averages rather than point-samples, because a viewer that point
// samples crawls with aliasing on every fine detail the moment a shot moves.
// And that the transfer function is the piecewise sRGB curve rather than a 2.2
// approximation, because the difference lives in the shadows, which is where a
// viewer gets looked at hardest.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "display.h"
#include "gpe/device.h"
#include "gpe/presenter.h"
#include "pool.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

/// Keeps whatever it is handed, so the test can look at it.
class Capture final : public gpe::Presenter {
public:
    Capture(int w, int h) : width_(w), height_(h) {}

    void targetSize(int& width, int& height) const override {
        width = width_;
        height = height_;
    }
    void present(const void* rgba8, int width, int height,
                 uint64_t frame) override {
        width_ = width;
        height_ = height;
        lastFrame_ = frame;
        const auto* bytes = static_cast<const unsigned char*>(rgba8);
        pixels_.assign(bytes, bytes + static_cast<size_t>(width) * height * 4);
        ++count_;
    }

    [[nodiscard]] unsigned char at(int x, int y, int channel) const {
        const size_t index =
            (static_cast<size_t>(y) * width_ + x) * 4 + channel;
        return index < pixels_.size() ? pixels_[index] : 0;
    }

    int                        width_ = 0;
    int                        height_ = 0;
    uint64_t                   lastFrame_ = 0;
    int                        count_ = 0;
    std::vector<unsigned char> pixels_;
};

/// The piecewise sRGB curve, on the host, to compare against.
double srgb(double linear) {
    return linear <= 0.0031308 ? linear * 12.92
                               : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
}

constexpr int kSrcW = 64;
constexpr int kSrcH = 64;

}   // namespace

int main() {
    using namespace gpe;

    std::unique_ptr<Device> native = Device::create();
    if (native == nullptr) {
        std::puts("test_display: skipped, no device");
        return 0;
    }
    PooledDevice device(std::move(native), size_t{1} << 30);

    const size_t srcBytes = size_t{kSrcW} * kSrcH * 16;
    const BufferId source = device.alloc(srcBytes);
    check(source != kInvalidBuffer, "the source allocates");

    DisplayPass pass(device);
    check(pass.prepare(256, 256), "the display pass prepares");

    // --- the transfer function is the real curve ---------------------------
    {
        // A flat plate at a value in the toe, where a 2.2 approximation and the
        // piecewise curve disagree by several code values.
        constexpr float kValue = 0.0025f;   // below the 0.0031308 knee
        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4, kValue);
        for (size_t i = 3; i < plate.size(); i += 4) {
            plate[i] = 1.0f;   // opaque
        }
        device.upload(source, plate.data(), srcBytes);

        Capture capture(kSrcW, kSrcH);
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, capture,
                     DisplayControls{}, 7);
        check(capture.count_ == 1, "one frame was presented");
        check(capture.lastFrame_ == 7, "with the frame number passed through");
        check(pass.lastBytes() == size_t{kSrcW} * kSrcH * 4,
              "and the transfer is the packed size, not the float size");

        const auto expected =
            static_cast<unsigned char>(srgb(kValue) * 255.0 + 0.5);
        const unsigned char got = capture.at(kSrcW / 2, kSrcH / 2, 0);
        if (got != expected) {
            std::fprintf(stderr, "  toe: got %u, the piecewise curve says %u, "
                                 "a 2.2 gamma would say %u\n",
                         got, expected,
                         static_cast<unsigned>(std::pow(kValue, 1.0 / 2.2) *
                                               255.0 + 0.5));
        }
        check(got == expected, "the toe follows the piecewise sRGB curve");
        check(capture.at(kSrcW / 2, kSrcH / 2, 3) == 255, "alpha is opaque");
    }

    // --- the downsample averages, it does not point sample -----------------
    {
        // Every other column black and white. A point sample of a 2:1
        // reduction gives one or the other; an average gives the middle, and
        // the middle is the only answer that does not crawl when the shot
        // moves.
        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4, 0.0f);
        for (int y = 0; y < kSrcH; ++y) {
            for (int x = 0; x < kSrcW; ++x) {
                const float value = (x % 2 == 0) ? 0.0f : 1.0f;
                const size_t at = (static_cast<size_t>(y) * kSrcW + x) * 4;
                plate[at + 0] = value;
                plate[at + 1] = value;
                plate[at + 2] = value;
                plate[at + 3] = 1.0f;
            }
        }
        device.upload(source, plate.data(), srcBytes);

        Capture capture(kSrcW / 2, kSrcH / 2);
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, capture,
                     DisplayControls{}, 1);

        const auto expected =
            static_cast<unsigned char>(srgb(0.5) * 255.0 + 0.5);
        const unsigned char got = capture.at(kSrcW / 4, kSrcH / 4, 0);
        if (got != expected) {
            std::fprintf(stderr,
                         "  downsample: got %u, an average of 0 and 1 encodes "
                         "to %u; 0 or 255 would mean a point sample\n",
                         got, expected);
        }
        check(got == expected, "a 2:1 reduction averages its two source pixels");
        check(capture.width_ == kSrcW / 2, "at the size the presenter asked for");
    }

    // --- exposure is applied in linear, before the transform ---------------
    {
        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4, 0.25f);
        for (size_t i = 3; i < plate.size(); i += 4) {
            plate[i] = 1.0f;
        }
        device.upload(source, plate.data(), srcBytes);

        Capture capture(kSrcW, kSrcH);
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, capture,
                     DisplayControls{1.0f, 1.0f}, 2);   // one stop up

        // One stop is a factor of two in linear, and then encoded -- not a
        // factor of two in the encoded value, which is what applying exposure
        // after the transform would give.
        const auto expected =
            static_cast<unsigned char>(srgb(0.5) * 255.0 + 0.5);
        check(capture.at(1, 1, 0) == expected,
              "a stop of exposure doubles the linear value, then encodes");
    }

    // --- a target big enough to catch a wrong element count ----------------
    {
        // The element size matters because Slang bound-checks against a count
        // in elements. At 64x64 a wrong count happens to be large enough and
        // the picture comes out right by luck; at a real widget size it is four
        // times too small and three quarters of the writes are dropped. This is
        // the size where luck runs out.
        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4, 1.0f);
        device.upload(source, plate.data(), srcBytes);

        Capture capture(256, 256);
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, capture,
                     DisplayControls{}, 3);

        size_t blank = 0;
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                if (capture.at(x, y, 0) == 0) {
                    ++blank;
                }
            }
        }
        if (blank > 0) {
            std::fprintf(stderr,
                         "  %zu of %d pixels were never written\n", blank,
                         256 * 256);
        }
        check(blank == 0, "every pixel of a full-size target is written");
    }

    // --- a baked transform replaces the built-in curve ----------------------
    {
        // An identity lattice: the pass must hand back exactly what went in,
        // with no encode at all. If the axis order were wrong this is the test
        // that would still pass -- so the next one uses a lattice that is
        // different along each axis, which is the one that catches it.
        constexpr int kN = 17;
        std::vector<float> identity(size_t{kN} * kN * kN * 4);
        for (int b = 0; b < kN; ++b) {
            for (int g = 0; g < kN; ++g) {
                for (int r = 0; r < kN; ++r) {
                    const size_t at =
                        ((static_cast<size_t>(b) * kN + g) * kN + r) * 4;
                    identity[at + 0] = static_cast<float>(r) / (kN - 1);
                    identity[at + 1] = static_cast<float>(g) / (kN - 1);
                    identity[at + 2] = static_cast<float>(b) / (kN - 1);
                    identity[at + 3] = 1.0f;
                }
            }
        }
        check(pass.setLut(identity.data(), kN, 0.0f, 1.0f), "the lattice uploads");

        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4);
        for (size_t i = 0; i < plate.size(); i += 4) {
            plate[i + 0] = 0.25f;
            plate[i + 1] = 0.50f;
            plate[i + 2] = 0.75f;
            plate[i + 3] = 1.0f;
        }
        device.upload(source, plate.data(), srcBytes);

        Capture capture(kSrcW, kSrcH);
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, capture,
                     DisplayControls{}, 5);

        // Within a code value: the lattice is 17 on a side and these are not
        // lattice points, so trilinear interpolation is doing real work here.
        const int r = capture.at(1, 1, 0);
        const int g = capture.at(1, 1, 1);
        const int b = capture.at(1, 1, 2);
        if (std::abs(r - 64) > 1 || std::abs(g - 128) > 1 ||
            std::abs(b - 191) > 1) {
            std::fprintf(stderr, "  identity lattice gave %d,%d,%d; wanted about "
                                 "64,128,191\n", r, g, b);
        }
        check(std::abs(r - 64) <= 1 && std::abs(g - 128) <= 1 &&
                  std::abs(b - 191) <= 1,
              "an identity lattice passes the linear values straight through");
    }

    // --- and the axes are not swapped --------------------------------------
    {
        // Red, green and blue each mapped to a different function of
        // themselves, so a lattice indexed in the wrong order comes out
        // visibly wrong rather than almost right. This is the mistake that
        // survives an identity test and shows up as a picture with the reds
        // and blues exchanged.
        constexpr int kN = 9;
        std::vector<float> twisted(size_t{kN} * kN * kN * 4);
        for (int b = 0; b < kN; ++b) {
            for (int g = 0; g < kN; ++g) {
                for (int r = 0; r < kN; ++r) {
                    const size_t at =
                        ((static_cast<size_t>(b) * kN + g) * kN + r) * 4;
                    twisted[at + 0] = static_cast<float>(r) / (kN - 1) * 0.25f;
                    twisted[at + 1] = static_cast<float>(g) / (kN - 1) * 0.50f;
                    twisted[at + 2] = static_cast<float>(b) / (kN - 1) * 1.00f;
                    twisted[at + 3] = 1.0f;
                }
            }
        }
        check(pass.setLut(twisted.data(), kN, 0.0f, 1.0f), "the lattice uploads");

        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4, 1.0f);
        device.upload(source, plate.data(), srcBytes);

        Capture capture(kSrcW, kSrcH);
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, capture,
                     DisplayControls{}, 6);

        const int r = capture.at(1, 1, 0);
        const int g = capture.at(1, 1, 1);
        const int b = capture.at(1, 1, 2);
        if (std::abs(r - 64) > 1 || std::abs(g - 128) > 1 || b != 255) {
            std::fprintf(stderr,
                         "  twisted lattice gave %d,%d,%d; wanted 64,128,255 -- "
                         "a swap of red and blue would give 255,128,64\n",
                         r, g, b);
        }
        check(std::abs(r - 64) <= 1 && std::abs(g - 128) <= 1 && b == 255,
              "each axis of the lattice drives its own channel");

        check(pass.setLut(nullptr, 0, 0.0f, 1.0f),
              "and removing the lattice is not an error");
    }

    device.release(source);
    device.sync();

    if (failures == 0) {
        std::puts("test_display: ok");
    }
    return failures == 0 ? 0 : 1;
}
