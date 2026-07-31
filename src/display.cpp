// Copyright (c) 2026 gpe contributors.
#include "display.h"

#include <algorithm>
#include <cstdio>

#include "gpe/args.h"

namespace gpe {
namespace {

/// Matches DisplayParams in display.slang, field for field. Four bytes each in
/// declaration order, so no padding rule has to agree across two compilers.
struct DisplayUniforms {
    uint32_t srcWidth = 0;
    uint32_t srcHeight = 0;
    uint32_t srcStride = 0;
    uint32_t dstWidth = 0;
    uint32_t dstHeight = 0;
    uint32_t dstStride = 0;
    float    exposure = 0.0f;
    float    gamma = 1.0f;
    uint32_t lutSize = 0;
    float    lutMin = 0.0f;
    float    lutMax = 1.0f;
    uint32_t channels = 0;
    uint32_t checkerboard = 0;
    float    checkerSize = 16.0f;
    uint32_t wipeEnabled = 0;
    float    wipeAt = 0.5f;
    uint32_t srcBWidth = 0;
    uint32_t srcBHeight = 0;
    uint32_t srcBStride = 0;
    float    checkerOriginX = 0.0f;
    float    checkerOriginY = 0.0f;
};
static_assert(sizeof(DisplayUniforms) == 84, "no padding, on any compiler");

}   // namespace

DisplayPass::~DisplayPass() {
    for (const Target& target : targets_) {
        if (target.buffer != kInvalidBuffer) {
            device_->release(target.buffer);
        }
    }
    if (lut_ != kInvalidBuffer) {
        device_->release(lut_);
    }
}

bool DisplayPass::setLut(const float* rgba, int size, float min, float max) {
    if (lut_ != kInvalidBuffer) {
        device_->release(lut_);
        lut_ = kInvalidBuffer;
    }
    lutSize_ = 0;
    if (rgba == nullptr || size < 2) {
        // Not an error: it is how a caller says "no transform", and the pass
        // then encodes sRGB, which is a defensible picture rather than a black
        // one. The single black sample below still gets bound, because a kernel
        // with an unbound buffer reads a null pointer.
        return true;
    }
    const size_t samples = static_cast<size_t>(size) * size * size;
    lut_ = device_->alloc(samples * 4 * sizeof(float));
    if (lut_ == kInvalidBuffer) {
        return false;
    }
    device_->upload(lut_, rgba, samples * 4 * sizeof(float));
    lutSize_ = size;
    lutMin_ = min;
    lutMax_ = max;
    return true;
}

bool DisplayPass::prepare(int maxWidth, int maxHeight) {
    if (maxWidth <= 0 || maxHeight <= 0) {
        return false;
    }
    kernel_ = device_->load("display");
    if (kernel_ == kInvalidKernel) {
        return false;
    }
    // One uint per pixel: packed RGBA8.
    const size_t bytes =
        static_cast<size_t>(maxWidth) * static_cast<size_t>(maxHeight) * 4;
    for (Target& target : targets_) {
        if (target.buffer != kInvalidBuffer) {
            device_->release(target.buffer);
        }
        target.buffer = device_->alloc(bytes);
        if (target.buffer == kInvalidBuffer) {
            return false;
        }
    }
    host_.assign(bytes, 0);
    if (lut_ == kInvalidBuffer) {
        // One black sample, so the binding is always valid even with no
        // transform. Sixteen bytes to remove a null dereference from a kernel.
        lut_ = device_->alloc(4 * sizeof(float));
        if (lut_ == kInvalidBuffer) {
            return false;
        }
        const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        device_->upload(lut_, black, sizeof(black));
    }
    maxWidth_ = maxWidth;
    maxHeight_ = maxHeight;
    return true;
}

void DisplayPass::present(const Image& source, Presenter& presenter,
                          const DisplayControls& controls, uint64_t frame) {
    presentWipe(source, Image{}, presenter, controls, frame);
}

void DisplayPass::presentWipe(const Image& source, const Image& right,
                              Presenter& presenter,
                              const DisplayControls& controls, uint64_t frame) {
    if (kernel_ == kInvalidKernel || source.buf == kInvalidBuffer) {
        return;
    }
    int width = 0;
    int height = 0;
    presenter.targetSize(width, height);
    if (width <= 0 || height <= 0) {
        return;
    }
    // Clamped rather than refused. A window dragged bigger than what prepare()
    // reserved gets a slightly smaller picture for a frame or two, which is a
    // great deal better than a viewer that goes black.
    width = std::min(width, maxWidth_);
    height = std::min(height, maxHeight_);

    const Target& target = targets_[at_];
    at_ = (at_ + 1) % kBuffers;

    Args args;
    // The target is packed RGBA8: one uint per pixel, four bytes an element.
    // The second side is always bound, because a kernel with an unbound buffer
    // reads a null pointer; with no wipe it is the first image again, which
    // costs nothing and is never read.
    const bool  wiping = right.buf != kInvalidBuffer;
    const Image other = wiping ? right : source;

    args.buffer(source.buf)
        .buffer(other.buf)
        .buffer(target.buffer, 4)
        .buffer(lut_)
        .uniforms(DisplayUniforms{
            static_cast<uint32_t>(source.w), static_cast<uint32_t>(source.h),
            static_cast<uint32_t>(source.stride), static_cast<uint32_t>(width),
            static_cast<uint32_t>(height), static_cast<uint32_t>(width),
            controls.exposure, controls.gamma,
            static_cast<uint32_t>(lutSize_), lutMin_, lutMax_,
            static_cast<uint32_t>(controls.channels),
            controls.checkerboard ? 1u : 0u, controls.checkerSize,
            wiping ? 1u : 0u, controls.wipeAt,
            static_cast<uint32_t>(other.w), static_cast<uint32_t>(other.h),
            static_cast<uint32_t>(other.stride),
            // Plus half a pixel, because the caller gives the output's corner
            // and a shader asks about a pixel's centre. Without it the cell
            // edges land a pixel out from the same pattern drawn any other way.
            controls.checkerOriginX + 0.5f, controls.checkerOriginY + 0.5f});
    device_->dispatch(kernel_,
                      Grid{static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height), 1},
                      args.data(), args.size());

    const size_t bytes = static_cast<size_t>(width) *
                         static_cast<size_t>(height) * 4;
    device_->download(host_.data(), target.buffer, bytes);
    presenter.present(host_.data(), width, height, frame);
    ++presented_;
    lastBytes_ = bytes;
}

}   // namespace gpe
