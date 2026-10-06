// Copyright (c) 2026 gpe contributors.
//
// Lanes: several independent queues of work on one device.
//
// One queue orders everything, and that is its virtue and its limit. Two
// pieces of work that have nothing to do with each other -- two frames, a
// model's inference beside a blur -- still wait for each other, because the
// device runs one queue's commands in order. A lane is a queue of its own: a
// CUDA stream, a Metal command queue, with its own count of what was
// submitted and what has finished. Work on different lanes may run at once;
// work on one lane runs in the order it was asked for, exactly as before.
//
// WHICH LANE A CALL GOES TO
//
// The calling thread's. `LaneScope` binds a thread to a lane for as long as
// it lives, and every call that thread makes -- dispatch, upload, fill,
// download, flush, and `stream()` for a runtime that enqueues its own work --
// goes to that lane. A thread that never binds is on lane 0, which is the
// queue there always was: a host that opens no lanes, or never binds a
// thread, sees no difference at all.
//
// Bound per thread rather than passed per call because the calls are many and
// their callers are layers deep -- an effect, a storage, an inference runtime
// -- and none of them should have to know there is more than one queue.
//
// BETWEEN LANES
//
// Memory is shared and the device does not know who is reading what. The
// pool keeps two promises across lanes and asks the caller for one:
//
//   - A released buffer is reused only once every lane that could have been
//     using it has finished that work. (Per lane, the same rule as ever.)
//   - A buffer last used on another lane, whose work there has not finished,
//     is waited for before this lane uses it: that lane's work is flushed and
//     the calling thread waits until it retires. Correct, and a stall --
//     which is why it is the fallback.
//   - The caller's part: hand work that depends on another lane's result to
//     that lane once the result is known to be done (`waitRetired` on its
//     Fence), not while it is still queued. Then the wait above never fires.
#pragma once

#include <cstdint>

namespace gpe {

using LaneId = uint32_t;

/// More than any card this is written for has queues worth having; the limit
/// keeps every per-lane table a fixed array, with nothing to allocate on the
/// frame path.
inline constexpr uint32_t kMaxLanes = 8;

/// A point in one lane's work: everything submitted on `lane` up to and
/// including `value`. What crosses threads when one thread queues work and
/// another waits for it.
struct Fence {
    LaneId   lane = 0;
    uint64_t value = 0;
};

/// The lane the calling thread's work goes to. 0 unless a LaneScope says
/// otherwise.
[[nodiscard]] LaneId currentLane() noexcept;

/// Binds the calling thread to `lane` until it goes out of scope, then puts
/// back whatever was bound before. A lane the device did not open is clamped
/// to the last one it did, by the device, when the work arrives.
class LaneScope {
public:
    explicit LaneScope(LaneId lane) noexcept;
    ~LaneScope();

    LaneScope(const LaneScope&) = delete;
    LaneScope& operator=(const LaneScope&) = delete;

private:
    LaneId previous_;
};

/// What a backend implements to have lanes. Found with a dynamic_cast, like
/// CompletionReporting: a backend without it has one lane and says so.
class LaneBackend {
public:
    virtual ~LaneBackend() = default;

    /// Makes `count` lanes, lane 0 being the queue that already exists.
    /// Answers how many there are now, at least one. Prepare-time: before any
    /// work is queued on the lanes it adds.
    virtual uint32_t openLanes(uint32_t count) = 0;

    /// Hands lane `lane`'s queued work to the device, from any thread -- the
    /// pool's way of making sure work another lane waits for has been
    /// submitted at all.
    virtual void flushLane(LaneId lane) = 0;
};

}   // namespace gpe
