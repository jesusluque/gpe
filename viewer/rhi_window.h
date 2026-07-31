// Copyright (c) 2026 gpe contributors.
//
// The window, and the frame loop that owns the swapchain.
//
// A QWindow with a QRhi of its own rather than a QRhiWidget. QRhiWidget always
// composes through an intermediate texture so that Qt's widget stack can put
// things on top of it, and for a viewer that fills the screen there is nothing
// to put on top -- so that texture is a full-screen copy per frame bought for
// nothing. A window renders straight into the swapchain.
//
// WHO OWNS WHAT
//
// This owns the frame. It opens one, tells the presenter which command buffer
// it opened, asks the engine for a picture, draws whatever the presenter has,
// and closes it. The engine never sees the swapchain and the window never sees
// a BufferId: they meet at Presenter, and that is the whole of the contract.
#pragma once

#include <QWindow>
#include <functional>
#include <memory>

QT_BEGIN_NAMESPACE
class QRhi;
class QRhiSwapChain;
QT_END_NAMESPACE

namespace gpe::viewer {

class RhiPresenter;

class RhiWindow : public QWindow {
    Q_OBJECT

public:
    /// Called once per frame, after the frame is open and before anything is
    /// drawn. Whatever it puts through the presenter is what appears.
    using OnFrame = std::function<void(uint64_t frame)>;

    RhiWindow();
    ~RhiWindow() override;

    /// The presenter the engine writes through. Valid after the first exposure.
    [[nodiscard]] RhiPresenter* presenter() const;

    void setOnFrame(OnFrame callback);

    /// Frames the window drew, and frames it could not because the swapchain
    /// was not ready -- a resize, a display change, a machine going to sleep.
    [[nodiscard]] uint64_t drawn() const noexcept { return drawn_; }
    [[nodiscard]] uint64_t skipped() const noexcept { return skipped_; }

protected:
    void exposeEvent(QExposeEvent*) override;
    bool event(QEvent*) override;
    void keyPressEvent(QKeyEvent*) override;

private:
    void initialise();
    void render();

    struct Impl;
    std::unique_ptr<Impl> impl_;
    OnFrame               onFrame_;
    uint64_t              frame_ = 0;
    uint64_t              drawn_ = 0;
    uint64_t              skipped_ = 0;
};

}   // namespace gpe::viewer
