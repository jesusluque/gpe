// Copyright (c) 2026 gpe contributors.
//
// The decode -> GPU boundary.
//
// A ring of host buffers, reserved once, with ownership by index. The decoder
// writes into a slot and publishes its number; the GPU thread uploads from that
// slot and, when the submission it belongs to has retired, the slot comes back.
//
// The same pattern as the device-side arenas, deliberately: nothing is
// transferred except the right to use a slot for a window. No handle crosses
// the boundary, which is the invariant that makes a BufferId safe to be
// meaningless outside its own pool -- here what crosses is an index and a raw
// pointer, and a raw pointer to host memory means the same thing on both sides.
//
// PINNED WHERE THE BACKEND CAN
//
// A DMA out of pageable memory has to be staged by the driver through a bounce
// buffer, which is a copy nobody asked for and a synchronisation point. Pinned
// memory is what makes an upload overlap with compute. But allocating it is a
// driver call, and the decoder must not touch the device -- so the backend
// provides it through the optional HostStaging interface and the ring hands out
// plain `void*`. A backend that does not implement it gets ordinary memory and
// a slower upload, not a broken one.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "completion.h"
#include "gpe/types.h"

namespace gpe {

/// A backend that can allocate memory the driver can DMA out of directly.
///
/// Optional, and found the same way completion reporting is: a second base a
/// backend may inherit, looked for once with a dynamic_cast.
class HostStaging {
public:
    virtual ~HostStaging() = default;
    /// Page-locked host memory, or null.
    [[nodiscard]] virtual void* allocHost(size_t bytes) = 0;
    virtual void freeHost(void* memory) = 0;

    /// One allocation that is both host memory and a device buffer.
    ///
    /// The upload that is not there. On a unified-memory machine the CPU and
    /// the GPU address the same physical pages, so a picture written here is a
    /// picture the GPU can already read -- there is nothing to copy, and the
    /// question of what format to copy it in does not arise.
    ///
    /// `out` receives a BufferId usable in a dispatch like any other. Returns
    /// null, and leaves `out` invalid, on a backend or a machine where this
    /// cannot be done -- a discrete card across PCIe, where the copy is real
    /// and pretending otherwise would trade a fast transfer for a slow kernel
    /// reading over the bus a pixel at a time.
    [[nodiscard]] virtual void* allocShared(size_t bytes, BufferId& out) {
        out = kInvalidBuffer;
        return nullptr;
    }
};

/// Host buffers between a producer thread and the GPU thread.
class HostRing {
public:
    /// `slots` buffers of `bytes` each, taken now. `staging` may be null, in
    /// which case the memory is ordinary and uploads will be slower.
    HostRing(size_t slots, size_t bytes, HostStaging* staging)
        : staging_(staging), bytes_(bytes) {
        slots_.reserve(slots);
        for (size_t i = 0; i < slots; ++i) {
            Slot slot;
            slot.memory = staging_ != nullptr ? staging_->allocHost(bytes)
                                              : std::malloc(bytes);
            if (slot.memory == nullptr) {
                return;   // short: `count()` says how many there really are
            }
            slots_.push_back(slot);
        }
    }

    ~HostRing() {
        for (const Slot& slot : slots_) {
            if (slot.memory == nullptr) {
                continue;
            }
            if (staging_ != nullptr) {
                staging_->freeHost(slot.memory);
            } else {
                std::free(slot.memory);
            }
        }
    }

    HostRing(const HostRing&) = delete;
    HostRing& operator=(const HostRing&) = delete;

    [[nodiscard]] size_t count() const noexcept { return slots_.size(); }
    [[nodiscard]] size_t bytes() const noexcept { return bytes_; }

    /// The next slot to write into, or null if the one whose turn it is has not
    /// come back yet.
    ///
    /// `retired` is the submission watermark: a slot is free once the upload
    /// that read it belongs to a submission that has finished. The producer
    /// asks and is told; it never waits on the device and never touches it.
    [[nodiscard]] void* acquire(uint64_t retired) {
        if (slots_.empty()) {
            return nullptr;
        }
        Slot& slot = slots_[at_];
        if (slot.inFlight && slot.submission > retired) {
            return nullptr;   // still being read
        }
        slot.inFlight = false;
        return slot.memory;
    }

    /// Says the slot `acquire` returned has been handed to the device as part
    /// of `submission`, and moves on to the next.
    void commit(uint64_t submission) {
        if (slots_.empty()) {
            return;
        }
        Slot& slot = slots_[at_];
        slot.inFlight = true;
        slot.submission = submission;
        at_ = (at_ + 1) % slots_.size();
        ++committed_;
    }

    /// How many slots were handed over. Producer-side bookkeeping, for a
    /// caller that wants to know whether it is keeping up.
    [[nodiscard]] uint64_t committed() const noexcept { return committed_; }

private:
    struct Slot {
        void*    memory = nullptr;
        uint64_t submission = 0;
        bool     inFlight = false;
    };

    HostStaging*      staging_ = nullptr;
    size_t            bytes_ = 0;
    std::vector<Slot> slots_;
    size_t            at_ = 0;
    uint64_t          committed_ = 0;
};

}   // namespace gpe
