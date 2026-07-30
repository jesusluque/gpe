// Copyright (c) 2026 gpe contributors.
//
// The one file that knows what QRhi is.
//
// Everything below gpe/presenter.h is free of Qt, and this is where that ends.
// The insulation is not tidiness. Qt's own header says it:
//
//     // This file is part of the RHI API, with limited compatibility
//     // guarantees. Usage of this API may make your code source and binary
//     // incompatible with future versions of Qt.
//
// and on this machine qrhi.h lives under the versioned private include path,
// so it is reached through Qt6::GuiPrivate. The Qt version is therefore pinned
// in the build and updating it is a change with risk -- and when that update
// breaks something, this is the file it breaks.
//
// DIRECTION, AND WHY IT IS THIS WAY ROUND
//
// QRhi owns the texture and gpe writes into it. The other direction --
// wrapping a gpe buffer with QRhi's createFrom() -- would make the engine's
// allocator responsible for an object Qt's renderer schedules, and the two
// have no shared vocabulary for saying when either has finished with it.
// Uploading into a texture Qt owns needs no such vocabulary.
//
// Two textures, written alternately: the frame being uploaded is never the
// frame being presented. QRhi exposes no hook for asking whether the GPU has
// finished with a texture, so the question is removed rather than answered.
#pragma once

#include <QSize>
#include <memory>

#include "gpe/presenter.h"

QT_BEGIN_NAMESPACE
class QRhi;
class QRhiTexture;
class QRhiCommandBuffer;
QT_END_NAMESPACE

namespace gpe::viewer {

class RhiPresenter final : public Presenter {
public:
    /// Does not own `rhi`. The application does, because the application owns
    /// the window it belongs to.
    explicit RhiPresenter(QRhi& rhi);
    ~RhiPresenter() override;

    RhiPresenter(const RhiPresenter&) = delete;
    RhiPresenter& operator=(const RhiPresenter&) = delete;

    /// The size to render at. Set by whoever owns the window, on every resize.
    void setTargetSize(QSize size);

    /// The command buffer of the frame now open, set by whoever opened it.
    ///
    /// QRhi does not let anyone ask which frame is current, and that is the
    /// right answer: the window owns the frame and its timing, and a presenter
    /// that opened one of its own would be a frame inside a frame. So the owner
    /// says, once per frame, and `present` uses what it was given -- or refuses
    /// and says so, rather than uploading into nothing.
    void setCommandBuffer(QRhiCommandBuffer* commands);

    void targetSize(int& width, int& height) const override;
    void present(const void* rgba8, int width, int height,
                 uint64_t frame) override;

    /// The texture the last frame went into, for the pass that draws it to the
    /// screen. Null before the first frame.
    [[nodiscard]] QRhiTexture* current() const;

    /// Frames uploaded, and how many needed the texture rebuilt. A rebuild per
    /// frame means the size is changing every frame, which is a resize loop
    /// somebody wants to know about.
    [[nodiscard]] uint64_t uploads() const noexcept { return uploads_; }
    [[nodiscard]] uint64_t rebuilds() const noexcept { return rebuilds_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    uint64_t              uploads_ = 0;
    uint64_t              rebuilds_ = 0;
};

}   // namespace gpe::viewer
