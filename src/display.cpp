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
};
static_assert(sizeof(DisplayUniforms) == 32, "no padding, on any compiler");

}   // namespace

DisplayPass::~DisplayPass() {
    for (const Target& target : targets_) {
        if (target.buffer != kInvalidBuffer) {
            device_->release(target.buffer);
        }
    }
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
    maxWidth_ = maxWidth;
    maxHeight_ = maxHeight;
    return true;
}

void DisplayPass::present(const Image& source, Presenter& presenter,
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
    args.buffer(source.buf).buffer(target.buffer, 4).uniforms(DisplayUniforms{
        static_cast<uint32_t>(source.w), static_cast<uint32_t>(source.h),
        static_cast<uint32_t>(source.stride), static_cast<uint32_t>(width),
        static_cast<uint32_t>(height), static_cast<uint32_t>(width),
        controls.exposure, controls.gamma});
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
