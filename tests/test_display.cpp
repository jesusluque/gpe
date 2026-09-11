// The display pass: what crosses to the CPU, and whether it is right.
//
// Two things are worth checking and neither is "it produced pixels". That the
// downsample averages rather than point-samples, because a viewer that point
// samples crawls with aliasing on every fine detail the moment a shot moves.
// And that the transfer function is the piecewise sRGB curve rather than a 2.2
// approximation, because the difference lives in the shadows, which is where a
// viewer gets looked at hardest.
#include <algorithm>
#include <chrono>
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

    // --- channels are isolated before the transform, not after --------------
    {
        // A pixel whose channels differ, so isolating one gives a different
        // answer from isolating another. Isolating *after* the transform would
        // give the encoded value of the channel rather than the channel through
        // the transform -- which is what a matte judged against the wrong curve
        // looks like, and is a mistake nobody notices until they trust it.
        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4);
        for (size_t i = 0; i < plate.size(); i += 4) {
            plate[i + 0] = 0.25f;
            plate[i + 1] = 0.50f;
            plate[i + 2] = 0.75f;
            plate[i + 3] = 1.0f;
        }
        device.upload(source, plate.data(), srcBytes);

        const auto isolated = [&](DisplayControls::Channels which) {
            Capture capture(kSrcW, kSrcH);
            DisplayControls controls;
            controls.channels = which;
            pass.present(Image{source, kSrcW, kSrcH, kSrcW}, capture, controls, 8);
            return static_cast<int>(capture.at(1, 1, 0));
        };

        // Each channel encoded through the same curve the picture uses.
        check(isolated(DisplayControls::Channels::Red) ==
                  static_cast<int>(srgb(0.25) * 255.0 + 0.5),
              "red is the red channel through the transform");
        check(isolated(DisplayControls::Channels::Green) ==
                  static_cast<int>(srgb(0.50) * 255.0 + 0.5),
              "green is the green channel");
        check(isolated(DisplayControls::Channels::Blue) ==
                  static_cast<int>(srgb(0.75) * 255.0 + 0.5),
              "blue is the blue channel");
        check(isolated(DisplayControls::Channels::Alpha) ==
                  static_cast<int>(srgb(1.0) * 255.0 + 0.5),
              "alpha is the matte");

        const double luma = 0.2126 * 0.25 + 0.7152 * 0.50 + 0.0722 * 0.75;
        const int wanted = static_cast<int>(srgb(luma) * 255.0 + 0.5);
        const int got = isolated(DisplayControls::Channels::Luminance);
        if (std::abs(got - wanted) > 1) {
            std::fprintf(stderr, "  luminance: got %d, wanted %d\n", got, wanted);
        }
        check(std::abs(got - wanted) <= 1, "and luminance is Rec.709 weighted");
    }

    // --- the checkerboard is chrome, and goes over the picture --------------
    {
        // Transparent everywhere, so what shows is the checker alone. It must
        // be the two greys as they are -- if it went through the display
        // transform it would change shade with the view, and chrome that moves
        // when the view changes is chrome nobody can judge a matte against.
        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4, 0.0f);
        device.upload(source, plate.data(), srcBytes);

        Capture capture(kSrcW, kSrcH);
        DisplayControls controls;
        controls.checkerboard = true;
        controls.checkerSize = 16.0f;
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, capture, controls, 9);

        // Cell (0,0) is the dark grey, cell (1,0) the light one.
        const int dark = capture.at(4, 4, 0);
        const int light = capture.at(20, 4, 0);
        const auto expectDark = static_cast<int>(0.18 * 255.0 + 0.5);
        const auto expectLight = static_cast<int>(0.28 * 255.0 + 0.5);
        if (dark != expectDark || light != expectLight) {
            std::fprintf(stderr, "  checker: got %d/%d, wanted %d/%d\n", dark,
                         light, expectDark, expectLight);
        }
        check(dark == expectDark && light == expectLight,
              "the checker is its own two greys, untouched by the transform");

        // And off, a transparent pixel is black rather than checkered.
        controls.checkerboard = false;
        Capture plain(kSrcW, kSrcH);
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, plain, controls, 10);
        check(plain.at(4, 4, 0) == 0, "and turning it off leaves black");
    }

    // --- the wipe: two pictures, one dispatch, one transform ----------------
    {
        // Both sides must go through the same transform, because the whole
        // point of a wipe is that the difference you see is the difference
        // between the pictures. Two sides encoded differently would make every
        // comparison a lie, and it is the kind of lie that looks like a grade.
        const BufferId other = device.alloc(srcBytes);
        check(other != kInvalidBuffer, "the second side allocates");

        std::vector<float> left(size_t{kSrcW} * kSrcH * 4, 0.0f);
        std::vector<float> right(size_t{kSrcW} * kSrcH * 4, 0.0f);
        for (size_t i = 0; i < left.size(); i += 4) {
            left[i + 0] = 0.25f;   // a value either side of the divider that
            left[i + 3] = 1.0f;    // encodes to something distinguishable
            right[i + 1] = 0.75f;
            right[i + 3] = 1.0f;
        }
        device.upload(source, left.data(), srcBytes);
        device.upload(other, right.data(), srcBytes);

        Capture capture(kSrcW, kSrcH);
        DisplayControls controls;
        controls.wipeAt = 0.5f;
        pass.presentWipe(Image{source, kSrcW, kSrcH, kSrcW},
                         Image{other, kSrcW, kSrcH, kSrcW}, capture, controls,
                         11);

        const auto expectLeft = static_cast<int>(srgb(0.25) * 255.0 + 0.5);
        const auto expectRight = static_cast<int>(srgb(0.75) * 255.0 + 0.5);

        // Left of the divider: the first picture, red only.
        check(capture.at(kSrcW / 4, kSrcH / 2, 0) == expectLeft,
              "left of the divider is the first picture");
        check(capture.at(kSrcW / 4, kSrcH / 2, 1) == 0,
              "and only the first picture");
        // Right: the second, green only.
        check(capture.at(3 * kSrcW / 4, kSrcH / 2, 1) == expectRight,
              "right of the divider is the second picture");
        check(capture.at(3 * kSrcW / 4, kSrcH / 2, 0) == 0,
              "and only the second picture");

        // The divider moves.
        controls.wipeAt = 0.9f;
        Capture moved(kSrcW, kSrcH);
        pass.presentWipe(Image{source, kSrcW, kSrcH, kSrcW},
                         Image{other, kSrcW, kSrcH, kSrcW}, moved, controls, 12);
        check(moved.at(3 * kSrcW / 4, kSrcH / 2, 0) == expectLeft,
              "a divider at 0.9 leaves three quarters showing the first");

        // And with no second image the pass is unchanged: a wipe of one
        // picture is not a wipe, and must not cost a branch's worth of
        // different answer.
        Capture plain(kSrcW, kSrcH);
        pass.present(Image{source, kSrcW, kSrcH, kSrcW}, plain, controls, 13);
        check(plain.at(3 * kSrcW / 4, kSrcH / 2, 0) == expectLeft,
              "and without a second image nothing wipes");

        device.release(other);
    }

    device.release(source);
    // --- a log2 shaper keeps the shadows of a wide scene-linear range ----------
    {
        const BufferId shaped = device.alloc(srcBytes);
        // A lattice baked from lutInput with the sRGB curve standing in for
        // the host's display transform. Over linear 0..16 on 33 points the
        // first cell is half a unit wide and a shadow value is a blend of
        // black and 0.5's code value; spread in stops from 2^-10 to 16 it
        // lands within a code value of the curve.
        constexpr int kN = 33;
        const auto bake = [](int size, float min, float max, DisplayPass::LutDomain domain) {
            std::vector<float> in;
            DisplayPass::lutInput(size, min, max, domain, in);
            std::vector<float> lattice(in.size() / 3 * 4);
            for (size_t k = 0; k < in.size() / 3; ++k) {
                for (int c = 0; c < 3; ++c) {
                    lattice[k * 4 + static_cast<size_t>(c)] = static_cast<float>(srgb(std::max(in[k * 3 + static_cast<size_t>(c)], 0.0f)));
                }
                lattice[k * 4 + 3] = 1.0f;
            }
            return lattice;
        };
        const float shadow = 0.01f;
        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4, shadow);
        for (size_t i = 3; i < plate.size(); i += 4) {
            plate[i] = 1.0f;
        }
        device.upload(shaped, plate.data(), srcBytes);
        const int wanted = static_cast<int>(std::lround(srgb(shadow) * 255.0));
        const auto through = [&](DisplayPass::LutDomain domain, float min) {
            const std::vector<float> lattice = bake(kN, min, 16.0f, domain);
            check(pass.setLut(lattice.data(), kN, min, 16.0f, domain), "the shaped lattice uploads");
            Capture capture(kSrcW, kSrcH);
            pass.present(Image{shaped, kSrcW, kSrcH, kSrcW}, capture, DisplayControls{}, 14);
            return static_cast<int>(capture.at(2, 2, 0));
        };
        const int even = through(DisplayPass::LutDomain::Linear, 0.0f);
        const int stops = through(DisplayPass::LutDomain::Log2, 1.0f / 1024.0f);
        std::printf("test_display: linear 0.01 through a 33-cube over 0..16: curve %d, even %d, log2 %d\n",
                    wanted, even, stops);
        check(std::abs(stops - wanted) <= 1, "a log2-shaped lattice holds a shadow within a code value");
        check(std::abs(even - wanted) > 4, "where an even lattice over the same range does not");
        check(pass.setLut(nullptr, 0, 0.0f, 1.0f), "and the lattice comes off again");
        device.release(shaped);
    }

    // --- dither: the mean survives rounding, the noise stays within a code ----
    {
        const BufferId shaped = device.alloc(srcBytes);
        // A flat value 0.4 of a code value above 100 after the sRGB curve.
        const double target = (100.4 / 255.0 + 0.055) / 1.055;
        const float linear = static_cast<float>(std::pow(target, 2.4));
        std::vector<float> plate(size_t{kSrcW} * kSrcH * 4, linear);
        for (size_t i = 3; i < plate.size(); i += 4) {
            plate[i] = 1.0f;
        }
        device.upload(shaped, plate.data(), srcBytes);
        const auto stats = [&](bool dither, double& mean, int& lo, int& hi) {
            DisplayControls controls;
            controls.dither = dither;
            Capture capture(kSrcW, kSrcH);
            pass.present(Image{shaped, kSrcW, kSrcH, kSrcW}, capture, controls, 15);
            double sum = 0.0;
            lo = 255;
            hi = 0;
            for (int y = 0; y < kSrcH; ++y) {
                for (int x = 0; x < kSrcW; ++x) {
                    const int v = capture.at(x, y, 0);
                    sum += v;
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
            }
            mean = sum / (kSrcW * kSrcH);
        };
        double plainMean = 0.0, ditheredMean = 0.0;
        int plainLo = 0, plainHi = 0, lo = 0, hi = 0;
        stats(false, plainMean, plainLo, plainHi);
        stats(true, ditheredMean, lo, hi);
        std::printf("test_display: 100.4 rounds to %.2f plain, %.2f dithered (%d..%d)\n", plainMean,
                    ditheredMean, lo, hi);
        check(plainLo == 100 && plainHi == 100, "without dither a flat value rounds to one code");
        check(std::abs(ditheredMean - 100.4) < 0.1, "with it the mean of the area is the value");
        check(lo >= 99 && hi <= 102, "and no pixel moves more than a code value and a half");
        device.release(shaped);
    }

    // --- late readback: never waits, a frame behind, same pixels -------------
    {
        // A flat grey plate and a flat white one on alternate frames, so a
        // picture handed over under the wrong frame number is visible.
        std::vector<float> grey(size_t{kSrcW} * kSrcH * 4, 0.18f);
        std::vector<float> white(size_t{kSrcW} * kSrcH * 4, 1.0f);
        for (size_t i = 3; i < grey.size(); i += 4) {
            grey[i] = 1.0f;
        }
        const BufferId greySource = device.alloc(srcBytes);
        const BufferId whiteSource = device.alloc(srcBytes);
        device.upload(greySource, grey.data(), srcBytes);
        device.upload(whiteSource, white.data(), srcBytes);

        DisplayControls flat;
        flat.checkerboard = false;
        Capture reference(kSrcW, kSrcH);
        pass.present(Image{greySource, kSrcW, kSrcH, kSrcW}, reference, flat, 0);
        const unsigned char greyCode = reference.at(3, 3, 0);

        for (const bool shared : {true, false}) {
        DisplayPass late(device);
        late.setReadback(DisplayPass::Readback::Late, shared);
        check(late.prepare(kSrcW, kSrcH), "a late-readback pass prepares");

        /// Checks, as each picture arrives, that frames only go forward and
        /// that each carries its own plate.
        class Ordered final : public Presenter {
        public:
            explicit Ordered(unsigned char grey) : grey_(grey) {}
            void targetSize(int& width, int& height) const override {
                width = kSrcW;
                height = kSrcH;
            }
            void present(const void* rgba8, int, int, uint64_t frame) override {
                const auto* bytes = static_cast<const unsigned char*>(rgba8);
                const unsigned char code = bytes[(3 * kSrcW + 3) * 4];
                forward = forward && (count == 0 || frame > last);
                matches = matches && code == ((frame % 2) == 0 ? grey_ : 255);
                last = frame;
                ++count;
            }
            unsigned char grey_;
            uint64_t      last = 0;
            int           count = 0;
            bool          forward = true;
            bool          matches = true;
        } ordered(greyCode);

        constexpr int kFrames = 40;
        for (int f = 0; f < kFrames; ++f) {
            const BufferId plate = (f % 2) == 0 ? greySource : whiteSource;
            late.present(Image{plate, kSrcW, kSrcH, kSrcW}, ordered, flat, static_cast<uint64_t>(f));
        }
        device.sync();
        late.drain(ordered);

        check(ordered.forward, "late readback hands frames over in order");
        check(ordered.matches, "each picture arrives under its own frame number");
        check(ordered.last == kFrames - 1, "and after a drain the last frame asked for is the one shown");
        check(ordered.count + static_cast<int>(late.overtaken()) <= kFrames,
              "no picture is shown twice");
        std::printf("test_display: late readback (%s) showed %d of %d frames, %llu overtaken\n",
                    shared ? "shared memory" : "downloadAsync", ordered.count, kFrames,
                    static_cast<unsigned long long>(late.overtaken()));
        }

        device.release(greySource);
        device.release(whiteSource);
    }

    // --- what each readback costs the thread that presents, at 1080p ---------
    {
        constexpr int kW = 1920;
        constexpr int kH = 1080;
        const size_t bytes = size_t{kW} * kH * 16;
        const BufferId plate = device.alloc(bytes);
        std::vector<float> pixels(size_t{kW} * kH * 4, 0.5f);
        device.upload(plate, pixels.data(), bytes);
        class Sink final : public Presenter {
        public:
            void targetSize(int& width, int& height) const override {
                width = kW;
                height = kH;
            }
            void present(const void*, int, int, uint64_t) override {}
        } sink;
        const auto timed = [&](DisplayPass::Readback mode, bool shared) {
            DisplayPass timedPass(device);
            timedPass.setReadback(mode, shared);
            if (!timedPass.prepare(kW, kH)) {
                return -1.0;
            }
            for (int f = 0; f < 5; ++f) {   // warm: kernel load, first allocations
                timedPass.present(Image{plate, kW, kH, kW}, sink, DisplayControls{}, 0);
            }
            device.sync();
            constexpr int kFrames = 60;
            const auto start = std::chrono::steady_clock::now();
            for (int f = 0; f < kFrames; ++f) {
                timedPass.present(Image{plate, kW, kH, kW}, sink, DisplayControls{}, static_cast<uint64_t>(f));
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            device.sync();
            return ms / kFrames;
        };
        std::printf("test_display: 1080p present on the calling thread: wait %.2f ms, late %.2f ms "
                    "(shared memory), late %.2f ms (downloadAsync)\n",
                    timed(DisplayPass::Readback::Wait, true), timed(DisplayPass::Readback::Late, true),
                    timed(DisplayPass::Readback::Late, false));
        device.release(plate);
    }

    device.sync();

    if (failures == 0) {
        std::puts("test_display: ok");
    }
    return failures == 0 ? 0 : 1;
}
