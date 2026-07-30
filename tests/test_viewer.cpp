// The QRhi presenter, offscreen.
//
// No window: QRhi renders offscreen perfectly well, and a test that needs a
// window server is a test that cannot run where builds run. What is checked is
// that the bytes handed over arrive in the texture unchanged, and that two
// consecutive frames go to two different textures -- which is the whole of the
// double buffering, and the part that is invisible until it is wrong.
#include <rhi/qrhi.h>

#include <QGuiApplication>
#include <cstdio>
#include <string>
#include <vector>

#include "rhi_presenter.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int kW = 64;
constexpr int kH = 48;

}   // namespace

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);

#if defined(Q_OS_MACOS)
    QRhiMetalInitParams params;
    std::unique_ptr<QRhi> rhi(QRhi::create(QRhi::Metal, &params));
    const char* which = "Metal";
#else
    QRhiNullInitParams params;
    std::unique_ptr<QRhi> rhi(QRhi::create(QRhi::Null, &params));
    const char* which = "Null";
#endif
    if (rhi == nullptr) {
        std::puts("test_viewer: skipped, no RHI backend");
        return 0;
    }
    std::printf("test_viewer: %s, Qt %s\n", which, qVersion());

    gpe::viewer::RhiPresenter presenter(*rhi);
    presenter.setTargetSize(QSize(kW, kH));

    int askedW = 0;
    int askedH = 0;
    presenter.targetSize(askedW, askedH);
    check(askedW == kW && askedH == kH, "the presenter reports its target size");

    // A pattern where every byte is different from its neighbours, so a row
    // stride mistake or a channel swap shows up as a mismatch rather than as
    // the same picture shifted.
    std::vector<unsigned char> pattern(static_cast<size_t>(kW) * kH * 4);
    for (size_t i = 0; i < pattern.size(); i += 4) {
        const size_t pixel = i / 4;
        pattern[i + 0] = static_cast<unsigned char>(pixel % 251);
        pattern[i + 1] = static_cast<unsigned char>((pixel * 7) % 253);
        pattern[i + 2] = static_cast<unsigned char>((pixel * 13) % 257 % 256);
        pattern[i + 3] = 255;
    }

    QRhiReadbackResult readback;
    {
        QRhiCommandBuffer* cb = nullptr;
        if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) {
            std::puts("test_viewer: skipped, could not begin a frame");
            return 0;
        }
        presenter.setCommandBuffer(cb);
        presenter.present(pattern.data(), kW, kH, 1);
        check(presenter.uploads() == 1, "one frame was uploaded");
        check(presenter.rebuilds() == 1, "and the textures were made once");

        QRhiTexture* written = presenter.current();
        check(written != nullptr, "the presenter has a texture to draw from");

        QRhiResourceUpdateBatch* batch = rhi->nextResourceUpdateBatch();
        batch->readBackTexture(QRhiReadbackDescription(written), &readback);
        cb->resourceUpdate(batch);
        rhi->endOffscreenFrame();
    }

    check(readback.data.size() ==
              static_cast<qsizetype>(pattern.size()),
          "the readback is the size that went in");
    if (readback.data.size() == static_cast<qsizetype>(pattern.size())) {
        size_t wrong = 0;
        for (size_t i = 0; i < pattern.size(); ++i) {
            if (static_cast<unsigned char>(readback.data[static_cast<qsizetype>(i)]) !=
                pattern[i]) {
                ++wrong;
            }
        }
        if (wrong > 0) {
            std::fprintf(stderr, "  %zu of %zu bytes differ\n", wrong,
                         pattern.size());
        }
        check(wrong == 0, "every byte arrived unchanged");
    }

    // --- two frames, two textures ------------------------------------------
    {
        // The double buffering, which exists because QRhi offers no way to ask
        // whether the GPU has finished with a texture. If both frames went to
        // the same one, the second would be overwriting pixels the first may
        // still be being drawn from -- rarely, and only under load.
        QRhiCommandBuffer* cb = nullptr;
        if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) {
            return failures == 0 ? 0 : 1;
        }
        presenter.setCommandBuffer(cb);
        presenter.present(pattern.data(), kW, kH, 2);
        QRhiTexture* second = presenter.current();
        presenter.present(pattern.data(), kW, kH, 3);
        QRhiTexture* third = presenter.current();
        rhi->endOffscreenFrame();

        check(second != nullptr && third != nullptr, "both frames landed");
        check(second != third,
              "consecutive frames go to different textures");
        check(presenter.rebuilds() == 1,
              "and the same size did not rebuild them");
    }

    // --- no command buffer, no silent upload -------------------------------
    {
        // A frame uploaded into no open frame is a frame that never appears,
        // and doing that quietly is worse than refusing.
        presenter.setCommandBuffer(nullptr);
        const uint64_t before = presenter.uploads();
        std::fprintf(stderr, "--- one deliberate warning follows ---\n");
        presenter.present(pattern.data(), kW, kH, 4);
        check(presenter.uploads() == before,
              "a present with no frame open is refused, not swallowed");
    }

    if (failures == 0) {
        std::puts("test_viewer: ok");
    }
    return failures == 0 ? 0 : 1;
}
