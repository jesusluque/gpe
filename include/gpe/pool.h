// Copyright (c) 2026 gpe contributors.
//
// The pool, as a decorator.
//
// Public since the lucabRTrender integration: a host builds one of these
// around the device it made or adopted, and a header only reachable through
// gpe's src/ made every other private header reachable along with it.
//
// `PooledDevice` is a `Device` that wraps another `Device`. The backends know
// nothing about any of this: their `alloc` is a `cudaMalloc` or a
// `newBufferWithLength` and nothing else, which keeps the part that has to be
// written twice as small as it can be and the part that is easy to get wrong
// written once.
//
// Everything here that is not one of the eight is a plain non-virtual method.
// Trim, statistics and completion reporting happen when a chain is built or
// when a submission retires, and neither wants a virtual call or a place in
// the interface.
//
// WHEN EACH CALL IS ALLOWED
//
// `alloc` belongs to prepare(), not to frame(). It can walk a free list, it can
// reclaim, it can synchronise, it can trim, it can ask its owner to give memory
// back, and it can fail -- and a call that can do any of those has no business
// on a path with a frame deadline. Playback at a fixed rate is the point of the
// engine; an API that reads well and misses a frame is the wrong trade.
//
// So: the chain reserves what it needs through `alloc` while it is being built
// or changed, and if the peak does not fit, prepare() fails then, before
// anything is played. During frame() the reservation is O(1) out of arenas that
// already exist, cannot fail, and never synchronises. Arenas are the next piece
// and are built on this one; what this file has to get right for them is the
// submission watermark, which is how "slot S has finished" is answered without
// asking the driver.
//
// `sync()` is likewise a prepare-time call. Inside a frame it is forbidden.
//
// THREADS
//
// A BufferId is a slot and a generation in *this* pool's table and means
// nothing anywhere else, so it must never cross a thread boundary: where work
// is handed between threads, what crosses is an index into a ring both sides
// pre-reserved -- host memory on the decode side, arena slots on the device
// side -- and never a handle. That much is unchanged.
//
// The pool itself is locked. It was single-threaded by contract, and the first
// real application broke the contract immediately and correctly: a compositor
// allocates its images on render threads, and an image is where the pool's
// memory goes. The check that caught it is still here and still fires, because
// a handle crossing threads is a different mistake from two threads allocating.
//
// The lock is affordable precisely because of the split this file already
// makes. `alloc` is prepare-time and off the frame path; what the frame path
// touches is the arenas, which hold buffers already taken and never call in
// here. A mutex on a call that happens when a chain is built costs nothing a
// frame can feel, and buys the correctness that the contract was only asking
// for politely.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string_view>
#include <mutex>
#include <thread>
#include <vector>

#include "gpe/completion.h"
#include "gpe/device.h"

namespace gpe {

/// A submission is one dispatch, counted. The pool never asks the backend what
/// has finished; the backend tells it, and until it does a released buffer is
/// not reusable however idle the device looks.
using Submission = uint64_t;

class PooledDevice final : public Device {
public:
    /// What the pool did, for the caller that wants to know why a frame was
    /// slow. Cheap to read: these are counters, not a walk of anything.
    struct Stats {
        size_t   bytesHeld = 0;      ///< allocated from the driver, in total
        size_t   bytesInUse = 0;     ///< handed out and not yet released
        /// What the callers actually asked for, of what is handed out.
        ///
        /// The difference against `bytesInUse` is what the size classes cost.
        /// It is not a rounding error: buckets are powers of two, and a
        /// 4096-wide float plate is 135 MiB, which is a hair over 128 and so
        /// takes 256. Television widths land just under a power of two and
        /// cinema widths just over, so the same policy is free for one and
        /// half again for the other -- and a host with an image cache is
        /// sized by what it *holds*, not by what it uses. Reported so that
        /// deciding what to do about it can be a measurement.
        size_t   bytesAsked = 0;
        size_t   bytesPending = 0;   ///< released, waiting on a submission
        size_t   liveBuffers = 0;
        uint64_t hits = 0;           ///< allocs served from a free list
        uint64_t misses = 0;         ///< allocs that had to reach the driver
        uint64_t reclaimed = 0;      ///< buffers returned by a completed submission
    };

    /// Takes ownership of `native`.
    ///
    /// `budgetBytes` is the pool's own ceiling, not the driver's. On Apple
    /// silicon there is no clean allocation failure to detect -- unified memory
    /// pages and the machine degrades quietly -- so the limit has to be one we
    /// impose and can therefore test identically on both backends.
    ///
    /// `onPressure` is the last thing tried before giving up: the pool asks its
    /// owner to release something, and the owner answers whether it did. The
    /// pool knows nothing about an image cache and must not.
    PooledDevice(std::unique_ptr<Device> native, size_t budgetBytes,
                 std::function<bool(size_t)> onPressure = {});
    ~PooledDevice() override;

    // --- the eight ---------------------------------------------------------

    [[nodiscard]] Backend backend() const override;
    [[nodiscard]] BufferId alloc(size_t bytes) override;
    void release(BufferId) override;
    void upload(BufferId, const void* src, size_t) override;
    void download(void* dst, BufferId, size_t) override;
    [[nodiscard]] bool fill(BufferId, uint8_t byte, size_t bytes) override;
    [[nodiscard]] KernelId load(std::string_view name) override;
    void dispatch(KernelId, Grid grid, const void* args, size_t) override;
    void sync() override;
    void flush() override;
    void memory(size_t& total, size_t& available) const override;
    /// Resolves the pooled handle first, so a caller that reaches for a stale
    /// one gets zero rather than somebody else's picture.
    /// Passed through to the backend and given a slot that is never recycled.
    ///
    /// A pooled buffer's whole life is being reused; an adopted one has no
    /// reuse in it -- the memory belongs to somebody who will take it back --
    /// so it takes a slot, answers while it is live, and the slot is retired
    /// rather than returned to a free list.
    [[nodiscard]] BufferId adopt(uint64_t devicePtr, size_t bytes) override;
    [[nodiscard]] uint64_t devicePointer(BufferId) const override;
    [[nodiscard]] uint64_t stream() const override;
    /// Resolved through the slot table, like `devicePointer`: a stale handle
    /// answers zero rather than the object now in its slot.
    [[nodiscard]] uint64_t backendBuffer(BufferId) const override;
    [[nodiscard]] uint64_t backendDevice() const override;
    [[nodiscard]] uint64_t backendQueue() const override;
    void downloadAsync(BufferId, void* dst, size_t bytes,
                       std::function<void(bool)> done) override;

    // --- everything else ---------------------------------------------------

    /// One allocation the CPU writes and the GPU reads, with no copy between.
    ///
    /// Adopted into the pool's table, because a BufferId only means anything
    /// there -- handing back the backend's own handle is the mistake that cost
    /// this project a day, and it is the same mistake whichever direction the
    /// buffer came from.
    ///
    /// Null, with `out` invalid, where the machine cannot do it: a discrete
    /// card across PCIe can map host memory, but a kernel then reads every
    /// pixel over the bus, and trading a fast DMA for a slow kernel is not an
    /// optimisation. The caller falls back to alloc + upload, which is what it
    /// would have done anyway.
    ///
    /// Not one of the eight, and not on the frame path: this is prepare-time.
    [[nodiscard]] void* allocShared(size_t bytes, BufferId& out);

    /// Page-locked host memory, where the backend can give it.
    ///
    /// For a caller that keeps its own pairing of host memory to a device
    /// buffer -- which is what a discrete card leaves it doing, `allocShared`
    /// having correctly refused. The transfers between the two are then DMA out
    /// of pinned pages rather than a staged copy through a bounce buffer, which
    /// is the difference between a transfer that overlaps with compute and one
    /// that does not.
    ///
    /// Null where the backend has no answer; ordinary memory is then the
    /// caller's business, and works.
    [[nodiscard]] void* allocHost(size_t bytes);
    void freeHost(void* memory);

    /// Gives every free buffer back to the driver. Never called on the frame
    /// path -- it is the opposite of what the pool is for.
    void trim();

    /// Under the lock, because the caller is usually a HUD on another thread
    /// and `alloc`/`release` mutate these counters while it reads them. A torn
    /// read of a byte count is only a wrong number on a display, but it is a
    /// data race, and a race that a sanitiser reports on every run buries the
    /// ones that matter.
    [[nodiscard]] Stats stats() const {
        const std::lock_guard<std::recursive_mutex> held(guard_);
        return stats_;
    }

    /// The backend's completion handler calls this, from whatever thread the
    /// driver runs handlers on. It publishes a number and does nothing else:
    /// no locks, no allocation, no call back into the pool.
    ///
    /// **Monotonic**, like `CompletionReporting::reportCompleted` and for one
    /// reason more than it. Handlers arriving out of order is the reason
    /// there; here the mark could also be pushed *backwards* by this library's
    /// own code, and that hung the player for good:
    ///
    ///   - `dispatch` counts the submission before handing it to the backend,
    ///     deliberately, so that a buffer released afterwards belongs to the
    ///     work that touched it. A backend that then refuses the dispatch --
    ///     malformed args, a kernel that would not load, a staging copy that
    ///     failed -- never runs a completion handler for that number, so the
    ///     reporter stays one behind for the rest of the session.
    ///   - `sync()` is the escape hatch for exactly that: it stores
    ///     `submitted_` here, and it is telling the truth, everything really
    ///     has finished. But it then calls `reclaimLocked()`, which
    ///     republishes the reporter's lower number -- and as a plain store
    ///     that undid the rescue one line after it happened.
    ///
    /// The mark only climbing makes `sync()` stick, and `FrameArenas::begin`
    /// can no longer wait forever on a submission that was counted and never
    /// queued. That wait is untimed on purpose, so backwards was fatal rather
    /// than slow: the symptom is a player that hangs when it stops, because a
    /// refused dispatch is most likely to be the last thing a slot saw.
    void notifyCompleted(Submission done) noexcept {
        Submission seen = completed_.load(std::memory_order_relaxed);
        while (done > seen &&
               !completed_.compare_exchange_weak(seen, done,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed)) {
        }
    }

    /// The submission a dispatch queued now would belong to.
    [[nodiscard]] Submission submission() const noexcept { return submitted_; }

    /// Has everything up to `at` finished?
    ///
    /// A read of the watermark and nothing else -- no driver call, so it is
    /// safe to ask inside a frame and cheap enough to ask every frame.
    [[nodiscard]] bool retired(Submission at) const noexcept {
        return completed_.load(std::memory_order_acquire) >= at;
    }

    /// Blocks until `retired(at)`.
    ///
    /// This is the stall path: on a pipeline that is keeping up nobody calls
    /// it, and calling it means a deadline has already been missed.
    ///
    /// A limitation worth naming rather than hiding: waiting for one slot needs
    /// a fence or an event, and the eight-method interface has neither -- it
    /// has `sync()`, which waits for everything. So this waits on the number a
    /// backend's completion handler publishes, which costs nothing and is
    /// exactly per-slot; and if a backend publishes nothing it falls back to
    /// `sync()` after a bounded spin, which is correct and blunt. A backend
    /// that wants the sharp version calls notifyCompleted from its handler.
    void waitFor(Submission at);

    /// The device underneath, for the optional interfaces a backend may
    /// implement -- completion reporting, host staging. Not for allocating
    /// through: a buffer taken from here would be outside the pool's table and
    /// nothing would ever recycle it.
    [[nodiscard]] const Device* nativeDevice() const noexcept {
        return native_.get();
    }

    /// The native handle behind a pooled one, for a backend that has to reach
    /// the real buffer -- and for the test that checks reuse really is reuse.
    /// `kInvalidBuffer` if the handle is stale.
    [[nodiscard]] BufferId nativeHandle(BufferId) const noexcept;

    /// Moves retired buffers from the pending queue to the free lists. Called
    /// on the way into `alloc`; exposed because a test needs to drive it.
    ///
    /// Reads the backend's completion counter first, when it has one. That is
    /// the whole of the handshake: an acquire load on this thread against a
    /// release store on the driver's.
    void reclaim();

private:
    /// The bodies, with `guard_` already held.
    void reclaimLocked();
    void trimLocked();

    struct Slot {
        BufferId native = kInvalidBuffer;
        size_t   bucketBytes = 0;
        /// What was asked for, which is what the bucket was rounded up from.
        size_t   askedBytes = 0;
        uint32_t generation = 1;   ///< odd while live, even while free
        bool     live = false;
        /// Host-visible and owned by the caller, so never recycled.
        bool     shared = false;
        /// Device memory somebody else owns. Never recycled and never freed.
        bool     borrowed = false;
    };

    struct Retiring {
        uint32_t   slot = 0;
        Submission at = 0;
    };

    [[nodiscard]] static int bucketFor(size_t bytes) noexcept;
    [[nodiscard]] static size_t bucketBytes(int bucket) noexcept;

    /// The slot behind a handle, or null if the generation does not match.
    [[nodiscard]] Slot* resolve(BufferId) noexcept;
    [[nodiscard]] const Slot* resolve(BufferId) const noexcept;

    /// Everything short of failing, in order, until `bytes` will fit.
    [[nodiscard]] bool makeRoom(size_t bytes);

    /// Complains once if a *handle* is used from a thread other than the one
    /// that first used the pool. Not a lock: the lock below makes the tables
    /// safe, and this says the thing the lock cannot -- that a BufferId is
    /// travelling somewhere it does not mean anything.
    void checkThread(const char* where) const;

    std::unique_ptr<Device>     native_;
    size_t                      budget_ = 0;
    std::function<bool(size_t)> onPressure_;
    /// The native device's completion counter, if it publishes one. Found once
    /// with a dynamic_cast in the constructor; null for a backend that does
    /// not, which then falls back to sync().
    const CompletionReporting*  reporter_ = nullptr;

    std::vector<Slot>                  slots_;
    std::vector<std::vector<uint32_t>> freeByBucket_;
    std::deque<Retiring>               retiring_;
    std::vector<uint32_t>              recycledSlots_;   ///< slot indices to reuse
    /// A dispatch's argument blob with its handles translated to native ones.
    /// A member rather than a local: the backend may copy it asynchronously.
    std::vector<unsigned char>         translated_;

    /// Written by the render thread, read by it too. The completion thread only
    /// ever writes `completed_`.
    Submission              submitted_ = 0;
    std::atomic<Submission> completed_{0};

    Stats stats_{};

    /// Guards every table in this object. See the note on threads above for
    /// why a lock here is affordable and a lock in the arenas would not be.
    ///
    /// Recursive because the memory-pressure callback is *expected* to come
    /// back in: the pool asks its owner to give something up, and the way an
    /// owner gives a buffer up is to release it. A plain mutex deadlocks there,
    /// which is exactly what it did.
    mutable std::recursive_mutex guard_;

    /// The thread that first used this pool, and whether it has been told about
    /// a second one. Mutable because the check belongs in const methods too.
    mutable std::thread::id owner_{};
    mutable bool            complainedAboutThread_ = false;
};

}   // namespace gpe
