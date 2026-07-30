// Copyright (c) 2026 gpe contributors.
#include "arena.h"

#include <cstdio>

namespace gpe {

FrameArenas::~FrameArenas() { releaseAll(); }

void FrameArenas::releaseAll() {
    for (Slot& slot : slots_) {
        for (const Image& image : slot.images) {
            device_->release(image.buf);
        }
        slot.images.clear();
        slot.endedAt = 0;
    }
    perSlot_ = 0;
}

bool FrameArenas::prepare(int w, int h, size_t count) {
    releaseAll();
    if (w <= 0 || h <= 0 || count == 0) {
        return false;
    }

    // No padding. A stride wider than the image buys alignment that matters to
    // a texture unit and nothing to a linear buffer, and it costs a bucket the
    // whole way up: 1920 pixels lands in the 32 MiB bucket with room to spare,
    // 2048 does not.
    const Image shape{kInvalidBuffer, w, h, w};
    const size_t bytes = shape.bytes();

    for (Slot& slot : slots_) {
        slot.images.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const BufferId id = device_->alloc(bytes);
            if (id == kInvalidBuffer) {
                // Everything or nothing. A half-prepared set would let playback
                // start and fail at a frame boundary, which is the failure this
                // whole split exists to move earlier.
                releaseAll();
                return false;
            }
            slot.images.push_back(Image{id, w, h, w});
        }
    }
    perSlot_ = count;
    // The buffers exist and belong to this object from here on. Nothing during
    // a frame allocates, and nothing during a frame can fail.
    return true;
}

void FrameArenas::begin(uint64_t index) {
    frame_ = index;
    current_ = static_cast<int>(index % kSlots);
    handed_ = 0;

    Slot& slot = slots_[current_];
    if (slot.endedAt == 0) {
        return;   // never used; nothing to wait for
    }

    // The one wait in the system, and it is a comparison rather than a call.
    //
    // Waiting on the driver would mean a full synchronisation -- everything
    // queued, not just this slot's work -- which is both more than is needed
    // and forbidden inside a frame. The completion handler publishes a number;
    // reclaim() reads it. If the pipeline is keeping up this loop does not run
    // at all, and `stalls_` counts the frames where it did.
    if (device_->submission() >= slot.endedAt) {
        device_->reclaim();
    }
    bool stalled = false;
    while (!device_->retired(slot.endedAt)) {
        if (!stalled) {
            ++stalls_;
            stalled = true;
        }
        // Nothing to do but let the device finish. This is the deadline being
        // missed; the caller's policy -- drop the frame or drop resolution --
        // is decided from `stalls()`, not here.
        device_->waitFor(slot.endedAt);
    }
}

Image FrameArenas::image() {
    Slot& slot = slots_[current_];
    if (handed_ >= slot.images.size()) {
        // A caller asking for more than it declared. Not a condition to
        // recover from: prepare() is where the count is agreed, and a chain
        // that quietly renders one node fewer is worse than one that stops.
        std::fprintf(stderr,
                     "gpe: frame %llu asked for image %zu of %zu reserved\n",
                     static_cast<unsigned long long>(frame_), handed_ + 1,
                     slot.images.size());
        return Image{};
    }
    return slot.images[handed_++];
}

void FrameArenas::end() {
    // Whatever was queued during this frame is what the slot has to outlive.
    // Recorded on the way out, so the next time round this slot the wait is for
    // exactly this frame's work and not for anything queued since.
    slots_[current_].endedAt = device_->submission();
}

}   // namespace gpe
