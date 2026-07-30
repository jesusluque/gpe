// Copyright (c) 2026 gpe contributors.
//
// The display pass, phase 1: run it on the device, download the small result.
//
// Double buffered, because QRhi exposes no hook for saying "the GPU has
// finished with this texture" -- so the engine cannot know when a presenter is
// done reading. Writing into N+1 while N is being presented removes the
// question: by the time a buffer comes round again a full frame has passed and
// the presenter has long since uploaded it.
#pragma once

#include <cstdint>
#include <vector>

#include "gpe/presenter.h"
#include "pool.h"

namespace gpe {

class DisplayPass {
public:
    explicit DisplayPass(PooledDevice& device) : device_(&device) {}
    ~DisplayPass();

    DisplayPass(const DisplayPass&) = delete;
    DisplayPass& operator=(const DisplayPass&) = delete;

    /// Reserves the device buffers and the host side for a target of at most
    /// `maxWidth` x `maxHeight`. Prepare-time; the only place this can fail.
    ///
    /// The maximum rather than the current size, so that resizing a window
    /// during playback does not allocate. A window bigger than what was
    /// reserved is clamped rather than refused -- a viewer that goes black
    /// because somebody dragged a corner is worse than one that shows a
    /// slightly smaller picture for a moment.
    [[nodiscard]] bool prepare(int maxWidth, int maxHeight);

    /// Transforms, downsamples and packs `source` into the presenter's size,
    /// downloads it, and hands it over.
    ///
    /// The download is synchronous, and that is the cost this phase pays: about
    /// 8 MB at a 4K widget rather than the 33 MB a full-size readback would be.
    /// The interop version behind the same Presenter removes it entirely.
    void present(const Image& source, Presenter& presenter,
                 const DisplayControls& controls, uint64_t frame);

    /// How many frames have been handed over, and how big the last one was.
    [[nodiscard]] uint64_t presented() const noexcept { return presented_; }
    [[nodiscard]] size_t lastBytes() const noexcept { return lastBytes_; }

private:
    struct Target {
        BufferId buffer = kInvalidBuffer;
    };

    /// Two, and the reason is in the file header.
    static constexpr int kBuffers = 2;

    PooledDevice*              device_ = nullptr;
    KernelId                   kernel_ = kInvalidKernel;
    Target                     targets_[kBuffers];
    int                        at_ = 0;
    int                        maxWidth_ = 0;
    int                        maxHeight_ = 0;
    std::vector<unsigned char> host_;
    uint64_t                   presented_ = 0;
    size_t                     lastBytes_ = 0;
};

}   // namespace gpe
