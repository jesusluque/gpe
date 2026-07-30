// Copyright (c) 2026 gpe contributors.
#include "pool.h"

#include "gpe/args.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdio>
#include <thread>
#include <utility>   // std::as_const -- libstdc++ does not pull it in

namespace gpe {
namespace {

/// The smallest bucket. A page: below this the rounding costs nothing and the
/// bookkeeping costs the same as it does for a big one.
constexpr size_t kMinBucketBytes = 4096;
constexpr int    kMinBucketShift = 12;   // 1 << 12 == 4096

/// The largest. 2^40 is a terabyte; no single image allocation approaches it,
/// and the table is 29 entries either way.
constexpr int kMaxBucketShift = 40;
constexpr int kBucketCount = kMaxBucketShift - kMinBucketShift + 1;

}   // namespace

int PooledDevice::bucketFor(size_t bytes) noexcept {
    if (bytes <= kMinBucketBytes) {
        return 0;
    }
    // The power of two at or above `bytes`. bit_width(n-1) is that exponent.
    const int shift = std::bit_width(bytes - 1);
    return std::min(shift - kMinBucketShift, kBucketCount - 1);
}

size_t PooledDevice::bucketBytes(int bucket) noexcept {
    return size_t{1} << (bucket + kMinBucketShift);
}

PooledDevice::PooledDevice(std::unique_ptr<Device> native, size_t budgetBytes,
                           std::function<bool(size_t)> onPressure)
    : native_(std::move(native)),
      budget_(budgetBytes),
      onPressure_(std::move(onPressure)),
      reporter_(dynamic_cast<const CompletionReporting*>(native_.get())),
      freeByBucket_(kBucketCount) {
    // Slot zero is never handed out, so that a zero BufferId is invalid for the
    // same reason it is in the interface: an uninitialised handle must not
    // resolve to a real buffer.
    slots_.emplace_back();
}

PooledDevice::~PooledDevice() {
    // Whatever is still out is a leak in the caller, and saying so at the one
    // moment it is knowable is worth more than a counter nobody reads. Not an
    // assert: a destructor that aborts turns a leak into a crash on exit, and
    // the leak is the thing worth reporting.
    size_t leaked = 0;
    size_t leakedBytes = 0;
    for (const Slot& slot : slots_) {
        if (slot.live) {
            ++leaked;
            leakedBytes += slot.bucketBytes;
        }
    }
    if (leaked > 0) {
        std::fprintf(stderr,
                     "gpe: %zu buffer(s) still held at shutdown, %zu byte(s)\n",
                     leaked, leakedBytes);
    }
    // Everything goes back, held and pending alike -- the native device is
    // about to be destroyed and it owns the memory.
    for (const Slot& slot : slots_) {
        if (slot.native != kInvalidBuffer) {
            native_->release(slot.native);
        }
    }
}

Device::Backend PooledDevice::backend() const { return native_->backend(); }

PooledDevice::Slot* PooledDevice::resolve(BufferId id) noexcept {
    return const_cast<Slot*>(std::as_const(*this).resolve(id));
}

const PooledDevice::Slot* PooledDevice::resolve(BufferId id) const noexcept {
    // Checked here, in every build.
    //
    // A use-after-release does not crash a compositor: the slot is real, it
    // just belongs to a different image now, and what comes out is a frame that
    // looks almost right. The generation is what turns that into nothing at
    // all, and it is worth the compare on the render path.
    const uint32_t slot = bufferSlot(id);
    if (slot == 0 || slot >= slots_.size()) {
        return nullptr;
    }
    const Slot& found = slots_[slot];
    if (!found.live || found.generation != bufferGeneration(id)) {
        return nullptr;
    }
    return &found;
}

BufferId PooledDevice::nativeHandle(BufferId id) const noexcept {
    const Slot* slot = resolve(id);
    return slot != nullptr ? slot->native : kInvalidBuffer;
}

void PooledDevice::reclaim() {
    // The one place the two threads meet, and it is a single acquire load. The
    // completion handler stores a number; this reads it and does all the work
    // on the render thread, so there is no lock and nothing for a driver
    // callback to contend on.
    if (reporter_ != nullptr) {
        notifyCompleted(reporter_->completedSubmissions());
    }
    const Submission done = completed_.load(std::memory_order_acquire);
    while (!retiring_.empty() && retiring_.front().at <= done) {
        const Retiring entry = retiring_.front();
        retiring_.pop_front();
        Slot& slot = slots_[entry.slot];
        freeByBucket_[bucketFor(slot.bucketBytes)].push_back(entry.slot);
        stats_.bytesPending -= slot.bucketBytes;
        ++stats_.reclaimed;
    }
}

bool PooledDevice::makeRoom(size_t bytes) {
    if (stats_.bytesHeld + bytes <= budget_) {
        return true;
    }
    // In order, cheapest first, and each step is only worth taking because the
    // one before it did not work.

    // 1. Buffers whose submission has already retired but that nobody has
    //    drained yet. Free, in both senses.
    reclaim();
    if (stats_.bytesHeld + bytes <= budget_) {
        return true;
    }

    // 2. Wait for the device, then drain again. This is the first step that
    //    costs real time, which is exactly why `alloc` is a prepare-time call.
    native_->sync();
    completed_.store(submitted_, std::memory_order_release);
    reclaim();
    if (stats_.bytesHeld + bytes <= budget_) {
        return true;
    }

    // 3. Give the driver back everything the pool is holding for reuse. The
    //    pool exists to avoid this; being out of memory is when it stops being
    //    the right trade.
    trim();
    if (stats_.bytesHeld + bytes <= budget_) {
        return true;
    }

    // 4. Ask whoever owns us. The pool does not know what an image cache is and
    //    must not; it says how much it needs and is told whether anything was
    //    given up. Once -- a loop here is a stall of unbounded length.
    if (onPressure_ && onPressure_(bytes)) {
        reclaim();
        trim();
        if (stats_.bytesHeld + bytes <= budget_) {
            return true;
        }
    }
    return false;
}

BufferId PooledDevice::alloc(size_t bytes) {
    if (bytes == 0) {
        return kInvalidBuffer;
    }
    const int    bucket = bucketFor(bytes);
    const size_t size = bucketBytes(bucket);
    if (size < bytes) {
        return kInvalidBuffer;   // past the biggest bucket; nothing to serve it
    }

    reclaim();

    uint32_t index = 0;
    if (!freeByBucket_[bucket].empty()) {
        index = freeByBucket_[bucket].back();
        freeByBucket_[bucket].pop_back();
        ++stats_.hits;
    } else {
        if (!makeRoom(size)) {
            return kInvalidBuffer;
        }
        const BufferId native = native_->alloc(size);
        if (native == kInvalidBuffer) {
            return kInvalidBuffer;
        }
        ++stats_.misses;
        stats_.bytesHeld += size;

        if (!recycledSlots_.empty()) {
            index = recycledSlots_.back();
            recycledSlots_.pop_back();
        } else {
            index = static_cast<uint32_t>(slots_.size());
            slots_.emplace_back();
        }
        slots_[index].native = native;
        slots_[index].bucketBytes = size;
    }

    Slot& slot = slots_[index];
    // Bumped on the way out, not on the way in: a handle that was released
    // holds the old generation, and the next thing to occupy the slot must not
    // answer to it.
    ++slot.generation;
    slot.live = true;
    stats_.bytesInUse += slot.bucketBytes;
    ++stats_.liveBuffers;

    return (static_cast<BufferId>(slot.generation) << kBufferSlotBits) | index;
}

void PooledDevice::release(BufferId id) {
    Slot* slot = resolve(id);
    if (slot == nullptr) {
        return;   // stale or never ours; releasing twice is not a crash
    }
    const uint32_t index = bufferSlot(id);
    slot->live = false;
    stats_.bytesInUse -= slot->bucketBytes;
    --stats_.liveBuffers;

    // Not back on the free list yet.
    //
    // Dispatches are asynchronous, so at this moment the device may still be
    // reading this buffer for work queued earlier. Handing it straight back
    // would let the next allocation write over a frame that is still being
    // rendered -- rarely, and only under load, which is the worst way for a bug
    // to behave. It waits for the submission it was last live during.
    //
    // Written here rather than left to cudaMallocAsync so that the rule is the
    // same on both backends and can be tested with neither.
    retiring_.push_back(Retiring{index, submitted_});
    stats_.bytesPending += slot->bucketBytes;
}

void PooledDevice::upload(BufferId id, const void* src, size_t bytes) {
    if (const Slot* slot = resolve(id); slot != nullptr) {
        native_->upload(slot->native, src, bytes);
    }
}

void PooledDevice::download(void* dst, BufferId id, size_t bytes) {
    if (const Slot* slot = resolve(id); slot != nullptr) {
        native_->download(dst, slot->native, bytes);
    }
}

KernelId PooledDevice::load(std::string_view name) { return native_->load(name); }

void PooledDevice::dispatch(KernelId kernel, Grid grid, const void* args,
                            size_t bytes) {
    // The handles in the blob are ours, and the backend has never heard of
    // them.
    //
    // Everywhere else the decorator translates -- upload, download, release all
    // resolve a slot and hand the native handle down -- and this was the one
    // place that forwarded the caller's bytes untouched. The backend then read
    // a pooled handle, which is a slot in the low half and a generation in the
    // high half, as an index into its own table: a wild pointer in
    // __constant__ memory and CUDA_ERROR_ILLEGAL_ADDRESS on the first read.
    //
    // Translated in place into a member so the bytes outlive the call, and a
    // blob that does not parse is passed through rather than mangled -- a
    // backend may one day take something that is not an Args.
    const void* forward = args;
    size_t      forwardBytes = bytes;
    if (const Args::View view = Args::read(args, bytes);
        view.valid && view.bufferCount > 0) {
        translated_.assign(static_cast<const unsigned char*>(args),
                           static_cast<const unsigned char*>(args) + bytes);
        auto* handles = reinterpret_cast<BufferId*>(
            translated_.data() + 2 * sizeof(uint32_t));
        for (uint32_t i = 0; i < view.bufferCount; ++i) {
            handles[i] = nativeHandle(handles[i]);
        }
        forward = translated_.data();
        forwardBytes = translated_.size();
    }

    // The counter moves first. A buffer released after this call belongs to
    // this submission and not the one before it, which is the difference
    // between waiting for the work that touched it and waiting for the work
    // that did not.
    ++submitted_;
    native_->dispatch(kernel, grid, forward, forwardBytes);
}

void PooledDevice::sync() {
    native_->sync();
    // A backend with completion handlers publishes this itself; doing it here
    // too costs nothing and keeps a backend that has none -- or a test -- from
    // holding every retiring buffer forever.
    completed_.store(submitted_, std::memory_order_release);
    reclaim();
}

void PooledDevice::waitFor(Submission at) {
    reclaim();
    if (retired(at)) {
        return;
    }
    // Long enough for work that is nearly done to land, short enough not to
    // burn a core on a backend that will never publish anything.
    for (int spins = 0; spins < 10000; ++spins) {
        if (reporter_ != nullptr) {
            notifyCompleted(reporter_->completedSubmissions());
        }
        if (retired(at)) {
            return;
        }
        std::this_thread::yield();
    }
    // Nothing arrived. Either the device is genuinely busy or this backend has
    // no completion handler; both are answered the same way, bluntly.
    sync();
}

void PooledDevice::trim() {
    reclaim();
    for (std::vector<uint32_t>& bucket : freeByBucket_) {
        for (const uint32_t index : bucket) {
            Slot& slot = slots_[index];
            native_->release(slot.native);
            stats_.bytesHeld -= slot.bucketBytes;
            slot.native = kInvalidBuffer;
            slot.bucketBytes = 0;
            recycledSlots_.push_back(index);
        }
        bucket.clear();
    }
}

}   // namespace gpe
