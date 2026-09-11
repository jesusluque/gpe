// Copyright (c) 2026 gpe contributors.
#include "display.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>

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
    float    viewOriginX = 0.0f;
    float    viewOriginY = 0.0f;
    float    viewPerPixelX = 1.0f;
    float    viewPerPixelY = 1.0f;
    uint32_t lutShaper = 0;
    uint32_t dither = 0;
};
static_assert(sizeof(DisplayUniforms) == 108, "no padding, on any compiler");

}   // namespace

DisplayPass::~DisplayPass() {
    releaseTargets();
    if (lut_ != kInvalidBuffer) {
        device_->release(lut_);
    }
}

void DisplayPass::releaseTargets() {
    // A readback still on its way writes into a target's host vector from the
    // device's completion thread: wait for every one before letting go.
    device_->sync();
    for (Target& target : targets_) {
        for (int spins = 0; spins < 100000 && target.state.load() == kPending && target.shared == nullptr;
             ++spins) {
            std::this_thread::yield();
        }
        if (target.buffer != kInvalidBuffer) {
            device_->release(target.buffer);
            target.buffer = kInvalidBuffer;
        }
        target.shared = nullptr;
        target.host.clear();
        target.state.store(kIdle);
    }
}

void DisplayPass::lutInput(int size, float min, float max, LutDomain domain, std::vector<float>& rgb) {
    rgb.assign(size >= 2 ? static_cast<size_t>(size) * size * size * 3 : 0, 0.0f);
    if (size < 2) {
        return;
    }
    const auto at = [&](int k) {
        const double unit = static_cast<double>(k) / (size - 1);
        if (domain == LutDomain::Log2) {
            const double lo = std::log2(std::max(static_cast<double>(min), 1e-10));
            const double hi = std::log2(std::max(static_cast<double>(max), static_cast<double>(min) * 2.0));
            return static_cast<float>(std::pow(2.0, lo + unit * (hi - lo)));
        }
        return static_cast<float>(min + unit * (max - min));
    };
    size_t out = 0;
    for (int b = 0; b < size; ++b) {
        for (int g = 0; g < size; ++g) {
            for (int r = 0; r < size; ++r) {
                rgb[out++] = at(r);
                rgb[out++] = at(g);
                rgb[out++] = at(b);
            }
        }
    }
}

bool DisplayPass::setLut(const float* rgba, int size, float min, float max, LutDomain domain) {
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
    lutDomain_ = domain;
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
    releaseTargets();
    for (int k = 0; k < usableTargets(); ++k) {
        Target& target = targets_[k];
        if (mode_ == Readback::Late) {
            // Memory both sides address, where the machine has it: then there
            // is nothing to read back, only a moment to wait for.
            target.shared = useShared_ ? static_cast<unsigned char*>(device_->allocShared(bytes, target.buffer))
                                       : nullptr;
            if (target.shared == nullptr) {
                target.buffer = device_->alloc(bytes);
                target.host.assign(bytes, 0);
            }
        } else {
            target.buffer = device_->alloc(bytes);
        }
        if (target.buffer == kInvalidBuffer) {
            return false;
        }
    }
    host_.assign(mode_ == Readback::Wait ? bytes : 0, 0);
    ordered_ = 0;
    shownOrder_ = 0;
    at_ = 0;
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
                          const DisplayControls& controls, uint64_t frame,
                          const DisplayView& view) {
    presentWipe(source, Image{}, presenter, controls, frame, view);
}

void DisplayPass::presentWipe(const Image& source, const Image& right,
                              Presenter& presenter,
                              const DisplayControls& controls, uint64_t frame,
                              const DisplayView& view) {
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

    if (mode_ == Readback::Wait) {
        Target& target = targets_[at_];
        at_ = (at_ + 1) % usableTargets();
        dispatchInto(target, source, right, controls, width, height, view);
        const size_t bytes = static_cast<size_t>(width) *
                             static_cast<size_t>(height) * 4;
        device_->download(host_.data(), target.buffer, bytes);
        presenter.present(host_.data(), width, height, frame);
        ++presented_;
        lastBytes_ = bytes;
        return;
    }

    drain(presenter);
    int free = idleTarget();
    if (free < 0) {
        // Every target is still travelling: the device is more than two
        // frames behind. Wait for it -- the only wait this mode has, and one
        // that means a deadline was already missed.
        device_->sync();
        drain(presenter);
        free = idleTarget();
        if (free < 0) {
            return;
        }
    }
    Target& target = targets_[free];
    target.order = ++ordered_;
    target.frame = frame;
    target.width = width;
    target.height = height;
    target.state.store(kPending);
    dispatchInto(target, source, right, controls, width, height, view);
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    if (target.shared != nullptr) {
        target.at = device_->submission();
        // Committed now, so it can finish before the next frame asks.
        device_->flush();
    } else {
        std::atomic<int>* state = &target.state;
        device_->downloadAsync(target.buffer, target.host.data(), bytes,
                               [state](bool fine) { state->store(fine ? kReady : kFailed); });
    }
}

int DisplayPass::idleTarget() {
    for (int k = 0; k < usableTargets(); ++k) {
        int state = targets_[k].state.load();
        if (state == kFailed) {
            targets_[k].state.store(kIdle);
            state = kIdle;
        }
        if (state == kIdle) {
            return k;
        }
    }
    return -1;
}

void DisplayPass::drain(Presenter& presenter) {
    if (mode_ != Readback::Late) {
        return;
    }
    device_->reclaim();
    Target* newest = nullptr;
    for (int k = 0; k < usableTargets(); ++k) {
        Target& target = targets_[k];
        if (target.shared != nullptr && target.state.load() == kPending && device_->retired(target.at)) {
            target.state.store(kReady);
        }
        if (target.state.load() == kReady && (newest == nullptr || target.order > newest->order)) {
            newest = &target;
        }
    }
    if (newest == nullptr) {
        return;
    }
    for (int k = 0; k < usableTargets(); ++k) {
        Target& target = targets_[k];
        if (&target != newest && target.state.load() == kReady && target.order < newest->order) {
            target.state.store(kIdle);
            if (target.order > shownOrder_) {
                ++overtaken_;
            }
        }
    }
    if (newest->order > shownOrder_) {
        const unsigned char* pixels = newest->shared != nullptr ? newest->shared : newest->host.data();
        presenter.present(pixels, newest->width, newest->height, newest->frame);
        ++presented_;
        lastBytes_ = static_cast<size_t>(newest->width) * static_cast<size_t>(newest->height) * 4;
        shownOrder_ = newest->order;
    }
    // The newest stays Ready, not Idle, until something newer arrives: a
    // drain with nothing new then has nothing to hand over, and a buffer the
    // presenter may still be showing is not rewritten.
}

void DisplayPass::dispatchInto(Target& target, const Image& source, const Image& right,
                               const DisplayControls& controls, int width, int height,
                               const DisplayView& view) {
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
            controls.checkerOriginX + 0.5f, controls.checkerOriginY + 0.5f,
            view.originX, view.originY,
            // Zero means "fit the source to the target", which is what every
            // caller wanted before there was a view to give.
            view.perPixelX > 0.0f ? view.perPixelX
                                  : static_cast<float>(source.w) /
                                        static_cast<float>(width),
            view.perPixelY > 0.0f ? view.perPixelY
                                  : static_cast<float>(source.h) /
                                        static_cast<float>(height),
            lutDomain_ == LutDomain::Log2 ? 1u : 0u,
            controls.dither ? 1u : 0u});
    device_->dispatch(kernel_,
                      Grid{static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height), 1},
                      args.data(), args.size());

}

}   // namespace gpe
