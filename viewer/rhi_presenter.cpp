// Copyright (c) 2026 gpe contributors.
#include "rhi_presenter.h"

#include <rhi/qrhi.h>

#include <QByteArray>
#include <QDebug>

namespace gpe::viewer {

struct RhiPresenter::Impl {
    explicit Impl(QRhi& r) : rhi(&r) {}

    ~Impl() {
        for (QRhiTexture* texture : textures) {
            delete texture;
        }
    }

    /// Makes both textures the given size, if they are not already.
    ///
    /// Both at once, because they alternate: one of them a different size from
    /// the other would show as every second frame being stretched, which is a
    /// bug that looks like a flicker and reads like a driver problem.
    [[nodiscard]] bool resize(QSize wanted) {
        if (size == wanted && textures[0] != nullptr) {
            return true;
        }
        for (QRhiTexture*& texture : textures) {
            delete texture;
            // UsedAsTransferSource so that a test -- and a screenshot -- can
            // read it back. It costs nothing on a texture that is uploaded to
            // and sampled from anyway.
            texture = rhi->newTexture(QRhiTexture::RGBA8, wanted, 1,
                                      QRhiTexture::UsedAsTransferSource);
            if (texture == nullptr || !texture->create()) {
                delete texture;
                texture = nullptr;
                return false;
            }
        }
        size = wanted;
        at = 0;
        return true;
    }

    QRhi*              rhi = nullptr;
    QRhiCommandBuffer* commands = nullptr;
    QRhiTexture* textures[2]{nullptr, nullptr};
    QSize        size;
    QSize        wanted{1280, 720};
    int          at = 0;
};

RhiPresenter::RhiPresenter(QRhi& rhi) : impl_(std::make_unique<Impl>(rhi)) {}
RhiPresenter::~RhiPresenter() = default;

void RhiPresenter::setTargetSize(QSize size) {
    if (size.width() > 0 && size.height() > 0) {
        impl_->wanted = size;
    }
}

void RhiPresenter::setCommandBuffer(QRhiCommandBuffer* commands) {
    impl_->commands = commands;
}

void RhiPresenter::targetSize(int& width, int& height) const {
    width = impl_->wanted.width();
    height = impl_->wanted.height();
}

QRhiTexture* RhiPresenter::current() const {
    // The one most recently written, which is the one *before* `at`.
    const int last = (impl_->at + 1) % 2;
    return impl_->textures[last];
}

void RhiPresenter::present(const void* rgba8, int width, int height,
                           uint64_t frame) {
    if (rgba8 == nullptr || width <= 0 || height <= 0) {
        return;
    }
    const QSize incoming(width, height);
    if (impl_->size != incoming) {
        if (!impl_->resize(incoming)) {
            qWarning("gpe/viewer: could not make a %dx%d texture", width, height);
            return;
        }
        ++rebuilds_;
    }

    QRhiTexture* target = impl_->textures[impl_->at];
    impl_->at = (impl_->at + 1) % 2;
    if (target == nullptr) {
        return;
    }

    // A view over the caller's bytes, not a copy of them.
    //
    // QRhi takes them during resourceUpdate, which happens inside the frame
    // this is called from, and the contract for `present` is that the bytes are
    // valid for the duration of the call. Copying into a QByteArray first would
    // be 8 MB of memcpy per frame to hand Qt the same bytes.
    QRhiTextureSubresourceUploadDescription subresource;
    subresource.setData(QByteArray::fromRawData(
        static_cast<const char*>(rgba8),
        static_cast<qsizetype>(static_cast<size_t>(width) * height * 4)));

    QRhiResourceUpdateBatch* batch = impl_->rhi->nextResourceUpdateBatch();
    batch->uploadTexture(target,
                         QRhiTextureUploadDescription(
                             QRhiTextureUploadEntry(0, 0, subresource)));

    // Handed to the frame the owner opened. There is no way to ask QRhi which
    // frame is current, and that is correct: the window owns the frame and its
    // timing. Without one, this refuses -- an upload into no frame is a frame
    // silently not shown.
    if (impl_->commands == nullptr) {
        batch->release();
        qWarning("gpe/viewer: no command buffer set; frame %llu not uploaded",
                 static_cast<unsigned long long>(frame));
        return;
    }
    impl_->commands->resourceUpdate(batch);
    ++uploads_;
}

}   // namespace gpe::viewer
