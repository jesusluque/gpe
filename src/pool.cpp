// Copyright (c) 2026 gpe contributors.
#include "pool.h"

#include "gpe/args.h"
#include "hostring.h"

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

/// The largest. 2^40 is a terabyte; no single image allocation approaches it.
constexpr int kMaxBucketShift = 40;
/// Steps within one doubling: 1, 1.25, 1.5, 1.75.
///
/// WHY NOT POWERS OF TWO
///
/// Because of what a plate is. A float32 RGBA image is width x height x 16,
/// and cinema widths sit just above a power of two where television widths sit
/// just below: 3840 wide is 126.6 MiB and takes the 128 MiB class, while 4096
/// wide is 135.0 MiB and takes 256. Measured on an L4 over a six-node chain,
/// the same graph held 2432 MiB for 2405 asked at UHD -- and 4864 MiB for 2565
/// at 4K DCI. Ninety per cent overhead, on the format a feature is finished in,
/// and invisible in every other figure: the cache reports what it believes it
/// holds, not what the device gave up for it.
///
/// Quarters cost nothing to compute and no bookkeeping worth the name: the
/// table is four times as long, which is 116 entries of a vector of vectors.
/// What they buy is that no size is ever charged more than a quarter over, and
/// the two DCI formats land within twenty per cent instead of ninety.
constexpr int kStepsPerDouble = 4;
constexpr int kBucketCount =
    (kMaxBucketShift - kMinBucketShift) * kStepsPerDouble + 1;

}   // namespace

int PooledDevice::bucketFor(size_t bytes) noexcept {
    if (bytes <= kMinBucketBytes) {
        return 0;
    }
    // The doubling this falls in, then the quarter within it. bit_width(n-1)-1
    // is the exponent at or below `bytes`.
    const int shift = std::bit_width(bytes - 1) - 1;
    const size_t floor = size_t{1} << shift;
    const size_t step = floor / kStepsPerDouble;
    // How many quarter-steps above `floor`, rounded up. `bytes` is above
    // `floor` by construction and at most one whole doubling above it.
    const int within =
        static_cast<int>((bytes - floor + step - 1) / step);
    const int bucket = (shift - kMinBucketShift) * kStepsPerDouble + within;
    return std::min(std::max(bucket, 0), kBucketCount - 1);
}

size_t PooledDevice::bucketBytes(int bucket) noexcept {
    const int shift = bucket / kStepsPerDouble + kMinBucketShift;
    const int within = bucket % kStepsPerDouble;
    const size_t floor = size_t{1} << shift;
    return floor + (floor / kStepsPerDouble) * static_cast<size_t>(within);
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

void PooledDevice::checkThread(const char* where) const {
    const std::thread::id here = std::this_thread::get_id();
    if (owner_ == std::thread::id{}) {
        owner_ = here;
        return;
    }
    if (owner_ == here || complainedAboutThread_) {
        return;
    }
    complainedAboutThread_ = true;
    std::fprintf(stderr,
                 "gpe: %s came from a second thread. The tables are locked, so "
                 "this is safe -- but submissions are ordered, and a dispatch "
                 "issued from elsewhere lands in a sequence the arenas are "
                 "counting on.\n",
                 where);
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
    const std::lock_guard<std::recursive_mutex> held(guard_);
    reclaimLocked();
}

void PooledDevice::reclaimLocked() {
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
    reclaimLocked();
    if (stats_.bytesHeld + bytes <= budget_) {
        return true;
    }

    // 2. Wait for the device, then drain again. This is the first step that
    //    costs real time, which is exactly why `alloc` is a prepare-time call.
    native_->sync();
    completed_.store(submitted_, std::memory_order_release);
    reclaimLocked();
    if (stats_.bytesHeld + bytes <= budget_) {
        return true;
    }

    // 3. Give the driver back everything the pool is holding for reuse. The
    //    pool exists to avoid this; being out of memory is when it stops being
    //    the right trade.
    trimLocked();
    if (stats_.bytesHeld + bytes <= budget_) {
        return true;
    }

    // 4. Ask whoever owns us. The pool does not know what an image cache is and
    //    must not; it says how much it needs and is told whether anything was
    //    given up. Once -- a loop here is a stall of unbounded length.
    if (onPressure_ && onPressure_(bytes)) {
        reclaimLocked();
        trimLocked();
        if (stats_.bytesHeld + bytes <= budget_) {
            return true;
        }
    }
    return false;
}

BufferId PooledDevice::alloc(size_t bytes) {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    if (bytes == 0) {
        return kInvalidBuffer;
    }
    const int    bucket = bucketFor(bytes);
    const size_t size = bucketBytes(bucket);
    if (size < bytes) {
        return kInvalidBuffer;   // past the biggest bucket; nothing to serve it
    }

    reclaimLocked();

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
    slot.askedBytes = bytes;
    stats_.bytesInUse += slot.bucketBytes;
    stats_.bytesAsked += bytes;
    ++stats_.liveBuffers;

    return (static_cast<BufferId>(slot.generation) << kBufferSlotBits) | index;
}

void PooledDevice::release(BufferId id) {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    Slot* slot = resolve(id);
    if (slot == nullptr) {
        return;   // stale or never ours; releasing twice is not a crash
    }
    const uint32_t index = bufferSlot(id);
    slot->live = false;
    --stats_.liveBuffers;

    if (slot->borrowed) {
        // Give up the claim and let the backend forget the pointer. Nothing is
        // freed, nothing goes on a free list, and the slot is recycled at once
        // -- there is no pending submission to wait for, because the buffer is
        // not ours to hand out again whatever the device is doing with it.
        native_->release(slot->native);
        slot->borrowed = false;
        slot->native = kInvalidBuffer;
        slot->bucketBytes = 0;
        ++slot->generation;
        recycledSlots_.push_back(index);
        return;
    }
    stats_.bytesInUse -= slot->bucketBytes;
    stats_.bytesAsked -= slot->askedBytes;

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
    if (slot->shared) {
        // Never recycled: it is the caller's memory, and a free list would hand
        // the pages they are writing into to somebody else.
        stats_.bytesHeld -= slot->bucketBytes;
        native_->release(slot->native);
        slot->native = kInvalidBuffer;
        slot->bucketBytes = 0;
        slot->shared = false;
        recycledSlots_.push_back(index);
        return;
    }
    retiring_.push_back(Retiring{index, submitted_});
    stats_.bytesPending += slot->bucketBytes;
}

void PooledDevice::upload(BufferId id, const void* src, size_t bytes) {
    // `resolve` reads the slot table, which another thread may be growing.
    const std::lock_guard<std::recursive_mutex> held(guard_);
    if (const Slot* slot = resolve(id); slot != nullptr) {
        native_->upload(slot->native, src, bytes);
    }
}

void PooledDevice::download(void* dst, BufferId id, size_t bytes) {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    if (const Slot* slot = resolve(id); slot != nullptr) {
        native_->download(dst, slot->native, bytes);
    }
}

bool PooledDevice::fill(BufferId id, uint8_t byte, size_t bytes) {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    const Slot* slot = resolve(id);
    return slot != nullptr && native_->fill(slot->native, byte, bytes);
}

KernelId PooledDevice::load(std::string_view name) { return native_->load(name); }

void PooledDevice::dispatch(KernelId kernel, Grid grid, const void* args,
                            size_t bytes) {
    // Locked, like everything else that touches the slot table.
    //
    // This used to be the one call that did not, on the grounds that dispatch
    // belongs to a single thread. That was true of dispatching and never true
    // of allocating: images are made on render threads, by design, and
    // `allocShared` grows `slots_` under the lock while this read it without
    // one. When the vector reallocated mid-read, `nativeHandle` returned
    // rubbish, the backend dispatched over a wild buffer, and the frame came
    // back exactly as it went in -- cleared. Intermittently, in proportion to
    // how much was being allocated, which is to say during playback.
    //
    // The cost is one uncontended lock per dispatch. What it buys is a picture.
    const std::lock_guard<std::recursive_mutex> held(guard_);

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
    checkThread("dispatch");
    ++submitted_;
    native_->dispatch(kernel, grid, forward, forwardBytes);
}

void PooledDevice::sync() {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    native_->sync();
    // A backend with completion handlers publishes this itself; doing it here
    // too costs nothing and keeps a backend that has none -- or a test -- from
    // holding every retiring buffer forever.
    completed_.store(submitted_, std::memory_order_release);
    reclaimLocked();
}

void PooledDevice::flush() {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    native_->flush();
}

void PooledDevice::waitFor(Submission at) {
    reclaim();
    if (retired(at)) {
        return;
    }
    // A backend holding the submission back in a batch would never report it:
    // hand it over first, or every wait is a spin and a full sync.
    flush();
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

void* PooledDevice::allocShared(size_t bytes, BufferId& out) {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    out = kInvalidBuffer;
    auto* staging = dynamic_cast<HostStaging*>(native_.get());
    if (staging == nullptr || bytes == 0) {
        return nullptr;
    }
    BufferId nativeId = kInvalidBuffer;
    void*    memory = staging->allocShared(bytes, nativeId);
    if (memory == nullptr || nativeId == kInvalidBuffer) {
        return nullptr;
    }

    // Adopted, not pooled. It never goes on a free list and is never handed to
    // anybody else: the caller owns this memory for as long as it holds the
    // pointer, and recycling it under them would be recycling the buffer they
    // are still writing their next frame into.
    uint32_t index = 0;
    if (!recycledSlots_.empty()) {
        index = recycledSlots_.back();
        recycledSlots_.pop_back();
    } else {
        index = static_cast<uint32_t>(slots_.size());
        slots_.emplace_back();
    }
    Slot& slot = slots_[index];
    slot.native = nativeId;
    slot.bucketBytes = bytes;
    slot.shared = true;
    ++slot.generation;
    slot.live = true;
    stats_.bytesHeld += bytes;
    stats_.bytesInUse += bytes;
    ++stats_.liveBuffers;

    out = (static_cast<BufferId>(slot.generation) << kBufferSlotBits) | index;
    return memory;
}

void PooledDevice::memory(size_t& total, size_t& available) const {
    native_->memory(total, available);
}

BufferId PooledDevice::adopt(uint64_t devicePtr, size_t bytes) {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    const BufferId native = native_->adopt(devicePtr, bytes);
    if (native == kInvalidBuffer) {
        return kInvalidBuffer;
    }
    // Its own slot, never from the free lists and never returned to one. An
    // adopted buffer has no reuse in it: the memory goes back to whoever owns
    // it, and a slot that offered it again would hand out a frame the decoder
    // has since overwritten.
    uint32_t index = 0;
    if (!recycledSlots_.empty()) {
        index = recycledSlots_.back();
        recycledSlots_.pop_back();
    } else {
        index = static_cast<uint32_t>(slots_.size());
        slots_.emplace_back();
    }
    Slot& slot = slots_[index];
    slot.native = native;
    slot.bucketBytes = bytes;
    slot.borrowed = true;
    ++slot.generation;
    slot.live = true;
    ++stats_.liveBuffers;
    // Deliberately not counted in `bytesInUse` or `bytesHeld`: this is not the
    // pool's memory and a budget that included it would be measuring somebody
    // else's allocations against our own ceiling.
    return (static_cast<BufferId>(slot.generation) << kBufferSlotBits) | index;
}

uint64_t PooledDevice::devicePointer(BufferId id) const {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    const Slot* slot = resolve(id);
    return slot != nullptr ? native_->devicePointer(slot->native) : 0;
}

uint64_t PooledDevice::stream() const { return native_->stream(); }

uint64_t PooledDevice::backendBuffer(BufferId id) const {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    const Slot* slot = resolve(id);
    return slot != nullptr ? native_->backendBuffer(slot->native) : 0;
}

uint64_t PooledDevice::backendDevice() const { return native_->backendDevice(); }
uint64_t PooledDevice::backendQueue() const { return native_->backendQueue(); }

void PooledDevice::downloadAsync(BufferId id, void* dst, size_t bytes,
                                 std::function<void(bool)> done) {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    const Slot* slot = resolve(id);
    if (slot == nullptr) {
        if (done) {
            done(false);
        }
        return;
    }
    native_->downloadAsync(slot->native, dst, bytes, std::move(done));
}

void* PooledDevice::allocHost(size_t bytes) {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    auto* staging = dynamic_cast<HostStaging*>(native_.get());
    return staging != nullptr && bytes != 0 ? staging->allocHost(bytes) : nullptr;
}

void PooledDevice::freeHost(void* memory) {
    if (memory == nullptr) {
        return;
    }
    const std::lock_guard<std::recursive_mutex> held(guard_);
    if (auto* staging = dynamic_cast<HostStaging*>(native_.get())) {
        staging->freeHost(memory);
    }
}

void PooledDevice::trim() {
    const std::lock_guard<std::recursive_mutex> held(guard_);
    trimLocked();
}

void PooledDevice::trimLocked() {
    reclaimLocked();
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
