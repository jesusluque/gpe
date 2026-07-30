// Copyright (c) 2026 gpe contributors.
//
// The bounded queue between two stages.
//
// One producer, one consumer, fixed capacity, no allocation after construction.
// Every one of those is load-bearing for playback at a fixed rate:
//
//   Bounded, because an unbounded queue does not drop frames -- it grows, and
//   latency grows with it until the picture is seconds behind the sound and the
//   machine is out of memory. A full queue is the signal that a stage is behind,
//   and it has to arrive as back-pressure rather than as a slow leak.
//
//   Single producer, single consumer, because that is what a chain of stages
//   is, and it is the only shape that needs no lock on the fast path.
//
//   No allocation, because a stage that allocates has a pause it did not
//   schedule.
//
// The blocking waits use a condition variable, which is a syscall and would be
// wrong inside a frame -- but a stage that is waiting has nothing else to do,
// and spinning a core to find that out is worse than sleeping. What must never
// block is the *render* thread's access to the device, and nothing here is on
// that path.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

namespace gpe {

template <typename T>
class Ring {
public:
    /// `capacity` is how many items may be in flight between the two stages.
    /// Three matches the frame pipelining; deeper buys latency, not throughput.
    explicit Ring(size_t capacity) : slots_(capacity + 1) {}

    /// Blocks while full. False once `close()` has been called.
    [[nodiscard]] bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [this] { return closed_ || !full(); });
        if (closed_) {
            return false;
        }
        slots_[write_] = std::move(value);
        write_ = next(write_);
        ++pushed_;
        lock.unlock();
        notEmpty_.notify_one();
        return true;
    }

    /// Blocks while empty. Empty once closed and drained, which is how a
    /// consumer learns to stop without a sentinel value.
    [[nodiscard]] std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [this] { return closed_ || !empty(); });
        if (empty()) {
            return std::nullopt;   // closed and drained
        }
        T value = std::move(slots_[read_]);
        read_ = next(read_);
        ++popped_;
        lock.unlock();
        notFull_.notify_one();
        return value;
    }

    /// Never blocks. For a stage that has something else worth doing.
    [[nodiscard]] bool tryPush(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || full()) {
                return false;
            }
            slots_[write_] = std::move(value);
            write_ = next(write_);
            ++pushed_;
        }
        notEmpty_.notify_one();
        return true;
    }

    /// Wakes everybody and stops accepting. Idempotent.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return (write_ + slots_.size() - read_) % slots_.size();
    }
    [[nodiscard]] size_t capacity() const noexcept { return slots_.size() - 1; }
    [[nodiscard]] uint64_t pushed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pushed_;
    }
    [[nodiscard]] uint64_t popped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return popped_;
    }

private:
    /// One slot is always left empty, which is what distinguishes full from
    /// empty without a separate count and without a wrap-around case to get
    /// wrong.
    [[nodiscard]] bool full() const { return next(write_) == read_; }
    [[nodiscard]] bool empty() const { return write_ == read_; }
    [[nodiscard]] size_t next(size_t at) const { return (at + 1) % slots_.size(); }

    mutable std::mutex      mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
    std::vector<T>          slots_;
    size_t                  read_ = 0;
    size_t                  write_ = 0;
    uint64_t                pushed_ = 0;
    uint64_t                popped_ = 0;
    bool                    closed_ = false;
};

}   // namespace gpe
