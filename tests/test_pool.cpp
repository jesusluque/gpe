// The pool, on a device made of malloc.
//
// No GPU anywhere in this file. The pool is the piece where a mistake is
// cheapest to make and dearest to find -- a buffer handed out twice shows up as
// a wrong pixel under load and nowhere else -- so it is tested against an
// allocator that can be inspected, counted and made to fail on demand.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "pool.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

/// A `Device` that allocates with malloc and counts everything.
///
/// It is a Device rather than a narrower allocator interface because that is
/// what `PooledDevice` wraps -- testing against the real shape means the test
/// exercises the real decorator, not a parallel one that happens to compile.
class FakeDevice final : public gpe::Device {
public:
    [[nodiscard]] Backend backend() const override { return Backend::Metal; }

    [[nodiscard]] gpe::BufferId alloc(size_t bytes) override {
        if (fail) {
            return gpe::kInvalidBuffer;
        }
        ++allocCalls;
        const gpe::BufferId id = nextId++;
        blocks[id] = std::malloc(bytes);
        sizes[id] = bytes;
        return id;
    }
    void release(gpe::BufferId id) override {
        ++releaseCalls;
        std::free(blocks[id]);
        blocks.erase(id);
        sizes.erase(id);
    }
    void upload(gpe::BufferId id, const void* src, size_t bytes) override {
        std::memcpy(blocks[id], src, bytes);
    }
    void download(void* dst, gpe::BufferId id, size_t bytes) override {
        std::memcpy(dst, blocks[id], bytes);
    }
    [[nodiscard]] gpe::KernelId load(std::string_view) override { return 1; }
    void dispatch(gpe::KernelId, gpe::Grid, const void*, size_t) override {
        ++dispatches;
    }
    void sync() override { ++syncs; }

    [[nodiscard]] void* pointerOf(gpe::BufferId id) { return blocks[id]; }

    gpe::BufferId                     nextId = 1;
    std::map<gpe::BufferId, void*>    blocks;
    std::map<gpe::BufferId, size_t>   sizes;
    int  allocCalls = 0;
    int  releaseCalls = 0;
    int  dispatches = 0;
    int  syncs = 0;
    bool fail = false;
};

constexpr size_t kGiB = size_t{1} << 30;
// 1920*1080*4 channels*4 bytes = 33,177,600 bytes = 31.64 MiB, which lands in
// the 32 MiB bucket with 1.1% left over. Image dimensions sit close under
// powers of two, so the rounding this scheme is accused of costing barely
// costs anything at the sizes that actually occur.
constexpr size_t kHdBytes = 1920ull * 1080ull * 16ull;
constexpr size_t kHdBucket = size_t{1} << 25;   // 32 MiB

}   // namespace

int main() {
    using namespace gpe;

    // --- reuse: the criterion the brief names ------------------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        PooledDevice pool(std::move(fake), 4 * kGiB);

        const BufferId a = pool.alloc(kHdBytes);
        check(a != kInvalidBuffer, "an HD plate allocates");
        void* firstPointer = raw->pointerOf(pool.nativeHandle(a));
        check(raw->allocCalls == 1, "the first alloc reaches the driver");

        // Queued work may still be reading it, so releasing does not free it.
        pool.dispatch(1, Grid{16, 16, 1}, nullptr, 0);
        pool.release(a);
        pool.reclaim();
        check(pool.stats().bytesPending == kHdBucket,
              "a buffer released while its submission is in flight is not reusable");
        check(pool.stats().bytesInUse == 0, "but it is no longer in use");

        pool.sync();   // retires everything
        check(pool.stats().bytesPending == 0, "and a completed submission frees it");

        const BufferId b = pool.alloc(kHdBytes);
        check(raw->allocCalls == 1, "which the free list then serves");
        check(raw->pointerOf(pool.nativeHandle(b)) == firstPointer,
              "and it is the same native pointer");
        pool.release(b);
        pool.sync();
    }

    // --- released with nothing queued is reusable at once ------------------
    {
        // The other half of the rule, and the one worth pinning down: the wait
        // is for work that might be reading the buffer, not a wait on principle.
        // With nothing dispatched there is no such work, and making the caller
        // sync to get their own memory back would be a stall for nothing.
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        PooledDevice pool(std::move(fake), 4 * kGiB);

        const BufferId a = pool.alloc(kHdBytes);
        pool.release(a);
        const BufferId b = pool.alloc(kHdBytes);
        check(raw->allocCalls == 1, "no dispatch, no wait");
        pool.release(b);
        pool.sync();
    }

    // --- generations: a released handle resolves to nothing ----------------
    {
        auto fake = std::make_unique<FakeDevice>();
        PooledDevice pool(std::move(fake), 4 * kGiB);

        const BufferId a = pool.alloc(kHdBytes);
        pool.release(a);
        pool.sync();
        const BufferId b = pool.alloc(kHdBytes);

        check(bufferSlot(a) == bufferSlot(b), "the slot really was reused");
        check(a != b, "but the handle is not the same one");
        check(pool.nativeHandle(a) == kInvalidBuffer,
              "the old handle resolves to nothing");
        check(pool.nativeHandle(b) != kInvalidBuffer, "the new one resolves");

        // And using it does nothing rather than something wrong. This is the
        // whole reason the generation is checked in release builds too.
        std::vector<float> pattern(4, 7.0f);
        pool.upload(a, pattern.data(), sizeof(float) * 4);
        std::vector<float> readBack(4, 0.0f);
        pool.download(readBack.data(), b, sizeof(float) * 4);
        check(readBack[0] == 0.0f,
              "a write through a stale handle does not reach the live buffer");

        pool.release(b);
        pool.sync();
    }

    // --- deferred recycling is per submission, not global -------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        PooledDevice pool(std::move(fake), 4 * kGiB);

        const BufferId a = pool.alloc(kHdBytes);
        pool.dispatch(1, Grid{16, 16, 1}, nullptr, 0);   // submission 1
        pool.release(a);                                  // retires after 1
        const Submission afterFirst = pool.submission();
        pool.dispatch(1, Grid{16, 16, 1}, nullptr, 0);   // submission 2

        // Only the first has finished.
        pool.notifyCompleted(afterFirst);
        pool.reclaim();
        const BufferId b = pool.alloc(kHdBytes);
        check(raw->allocCalls == 1,
              "a buffer whose submission has retired comes back without the driver");
        pool.release(b);
        pool.sync();
    }

    // --- buckets round up, and the rounding is what gets reused ------------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        PooledDevice pool(std::move(fake), 4 * kGiB);

        const BufferId a = pool.alloc(kHdBytes);   // 31.64 MiB -> 32 MiB bucket
        check(raw->sizes[pool.nativeHandle(a)] == kHdBucket,
              "an HD plate lands in the 32 MiB bucket");
        pool.release(a);
        pool.sync();

        // Anything else in the same bucket reuses it. This is the payoff for
        // the rounding: a 30 MiB plate and a 31.64 MiB plate share a buffer.
        const BufferId b = pool.alloc(30ull * 1024 * 1024);
        check(raw->allocCalls == 1, "a different size in the same bucket reuses");
        pool.release(b);

        pool.sync();
        // A size in the next bucket up does not.
        const BufferId c = pool.alloc(40ull * 1024 * 1024);
        check(raw->allocCalls == 2, "a size in the next bucket does not reuse");
        pool.release(c);
        pool.sync();
    }

    // --- the budget is ours, not the driver's ------------------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        // Room for exactly two HD buckets.
        PooledDevice pool(std::move(fake), 2 * kHdBucket);

        const BufferId a = pool.alloc(kHdBytes);
        const BufferId b = pool.alloc(kHdBytes);
        check(a != kInvalidBuffer && b != kInvalidBuffer, "two fit");

        // The third does not, and the pool says so rather than letting the
        // system page. On unified memory there is no clean failure to detect,
        // which is the whole reason the ceiling is ours.
        const BufferId c = pool.alloc(kHdBytes);
        check(c == kInvalidBuffer, "the third is refused");
        check(raw->allocCalls == 2, "and the driver was never asked for it");

        // Freeing one makes room again -- through the ladder, without the
        // caller having to know any of it happened.
        pool.release(a);
        const BufferId d = pool.alloc(kHdBytes);
        check(d != kInvalidBuffer, "a release makes room for the next");
        pool.release(b);
        pool.release(d);
        pool.sync();
    }

    // --- the pressure callback is the last resort --------------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        PooledDevice* poolPtr = nullptr;
        int asked = 0;
        std::vector<BufferId> cache;

        PooledDevice pool(std::move(fake), 2 * kHdBucket,
                          [&](size_t) {
                              ++asked;
                              if (cache.empty()) {
                                  return false;
                              }
                              poolPtr->release(cache.back());
                              cache.pop_back();
                              return true;
                          });
        poolPtr = &pool;

        cache.push_back(pool.alloc(kHdBytes));
        const BufferId keep = pool.alloc(kHdBytes);
        check(cache.front() != kInvalidBuffer && keep != kInvalidBuffer,
              "the budget is full");

        const BufferId wanted = pool.alloc(kHdBytes);
        check(asked == 1, "the owner was asked once");
        check(wanted != kInvalidBuffer,
              "and giving something up was enough to serve the allocation");

        // Asked once, not in a loop: a loop here is a stall of unbounded
        // length, which on a frame path is worse than failing.
        const BufferId hopeless = pool.alloc(kHdBytes);
        check(hopeless == kInvalidBuffer, "with nothing left to give, it fails");
        check(asked == 2, "asked once more, and only once");

        pool.release(keep);
        pool.release(wanted);
        pool.sync();
    }

    // --- trim gives it all back --------------------------------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        PooledDevice pool(std::move(fake), 4 * kGiB);

        std::vector<BufferId> held;
        for (int i = 0; i < 4; ++i) {
            held.push_back(pool.alloc(kHdBytes));
        }
        for (const BufferId id : held) {
            pool.release(id);
        }
        pool.sync();
        check(raw->blocks.size() == 4, "four buffers are being kept for reuse");
        check(pool.stats().bytesHeld == 4 * kHdBucket, "and counted");

        pool.trim();
        check(raw->blocks.empty(), "trim hands every free buffer back");
        check(pool.stats().bytesHeld == 0, "and the count goes with them");

        // The slots come back too, rather than the table growing forever.
        const BufferId again = pool.alloc(kHdBytes);
        check(bufferSlot(again) <= 4, "a trimmed slot is reused, not appended");
        pool.release(again);
        pool.sync();
    }

    // --- a driver that refuses -------------------------------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        PooledDevice pool(std::move(fake), 4 * kGiB);
        raw->fail = true;
        check(pool.alloc(kHdBytes) == kInvalidBuffer,
              "a driver that refuses gives an invalid handle, not a crash");
        check(pool.stats().bytesHeld == 0, "and nothing is counted as held");
    }

    // --- zero, and sizes past the last bucket -------------------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        PooledDevice pool(std::move(fake), 4 * kGiB);
        check(pool.alloc(0) == kInvalidBuffer, "zero bytes is not a buffer");
        check(pool.nativeHandle(kInvalidBuffer) == kInvalidBuffer,
              "the invalid handle resolves to nothing");
        check(pool.nativeHandle(0xdeadbeefull << 32) == kInvalidBuffer,
              "and so does a handle that was never issued");
    }

    if (failures == 0) {
        std::puts("test_pool: ok");
    }
    return failures == 0 ? 0 : 1;
}
