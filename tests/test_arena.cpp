// The frame allocator: three slots, no driver call, no way to fail.
//
// The properties worth pinning are not "it returns a buffer". They are that
// nothing during a frame reaches the driver, that a slot is never reused before
// the work that read it has finished, and that the pipeline really is three
// deep -- because each of those is invisible until it is wrong under load.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>

#include "arena.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

/// Counts every call, and never completes anything by itself: completion is
/// published by the test, which is how a real backend's handler behaves and the
/// only way to drive the three-slot wait deliberately.
class FakeDevice final : public gpe::Device {
public:
    [[nodiscard]] Backend backend() const override { return Backend::Metal; }
    [[nodiscard]] gpe::BufferId alloc(size_t bytes) override {
        ++allocCalls;
        const gpe::BufferId id = nextId++;
        blocks[id] = std::malloc(bytes);
        return id;
    }
    void release(gpe::BufferId id) override {
        std::free(blocks[id]);
        blocks.erase(id);
    }
    void upload(gpe::BufferId, const void*, size_t) override {}
    void download(void*, gpe::BufferId, size_t) override {}
    [[nodiscard]] gpe::KernelId load(std::string_view) override { return 1; }
    void dispatch(gpe::KernelId, gpe::Grid, const void*, size_t) override {}
    void sync() override { ++syncs; }

    gpe::BufferId                  nextId = 1;
    std::map<gpe::BufferId, void*> blocks;
    int allocCalls = 0;
    int syncs = 0;
};

constexpr size_t kGiB = size_t{1} << 30;

}   // namespace

int main() {
    using namespace gpe;

    // --- prepare reserves for every slot; frame() never allocates -----------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        PooledDevice pool(std::move(fake), 4 * kGiB);
        FrameArenas arenas(pool);

        check(arenas.prepare(1920, 1080, 3), "a chain of three images prepares");
        check(raw->allocCalls == 3 * FrameArenas::kSlots,
              "one set of three per slot, taken up front");
        check(arenas.capacity() == 3, "and the slot's capacity is three");

        const int afterPrepare = raw->allocCalls;
        for (uint64_t f = 0; f < 12; ++f) {
            arenas.begin(f);
            for (int i = 0; i < 3; ++i) {
                const Image image = arenas.image();
                check(image.buf != kInvalidBuffer, "an image comes back");
                check(image.w == 1920 && image.h == 1080, "at the right size");
                check(image.stride == 1920, "unpadded");
            }
            pool.dispatch(1, Grid{120, 68, 1}, nullptr, 0);
            arenas.end();
            // The pipeline is keeping up, so everything queued has retired by
            // the time this slot comes round again.
            pool.notifyCompleted(pool.submission());
        }
        check(raw->allocCalls == afterPrepare,
              "twelve frames and not one driver allocation");
        check(raw->syncs == 0, "and not one synchronisation");
        check(arenas.stalls() == 0, "nor a stall");
    }

    // --- three slots really are three --------------------------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        PooledDevice pool(std::move(fake), 4 * kGiB);
        FrameArenas arenas(pool);
        check(arenas.prepare(64, 64, 1), "a one-image chain prepares");

        std::set<BufferId> seen;
        BufferId first = kInvalidBuffer;
        for (uint64_t f = 0; f < FrameArenas::kSlots; ++f) {
            arenas.begin(f);
            const BufferId id = arenas.image().buf;
            if (f == 0) {
                first = id;
            }
            seen.insert(id);
            pool.dispatch(1, Grid{4, 4, 1}, nullptr, 0);
            arenas.end();
            pool.notifyCompleted(pool.submission());
        }
        check(seen.size() == FrameArenas::kSlots,
              "three consecutive frames get three different buffers");

        // And the fourth comes back to the first, which is what makes it a
        // ring rather than a queue that grows.
        arenas.begin(FrameArenas::kSlots);
        check(arenas.image().buf == first, "the fourth frame reuses the first");
        arenas.end();
    }

    // --- a slot is not reused before its work has finished ------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        PooledDevice pool(std::move(fake), 4 * kGiB);
        FrameArenas arenas(pool);
        check(arenas.prepare(64, 64, 1), "prepared");

        // Three frames queued and nothing reported as finished.
        for (uint64_t f = 0; f < 3; ++f) {
            arenas.begin(f);
            (void)arenas.image();
            pool.dispatch(1, Grid{4, 4, 1}, nullptr, 0);
            arenas.end();
        }
        check(arenas.stalls() == 0, "three frames deep, still no stall");
        check(raw->syncs == 0, "and still no synchronisation");

        // The fourth lands on slot 0, whose work has not been reported. It has
        // to wait. With nothing publishing completion the wait falls back to a
        // sync, which is the blunt answer and the one that must happen rather
        // than the frame going ahead over a buffer still being read.
        arenas.begin(3);
        check(arenas.stalls() == 1, "the fourth frame stalls on slot zero");
        check(raw->syncs == 1, "and with no handler it falls back to a sync");
        arenas.end();
    }

    // --- prepare is all or nothing -----------------------------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        FakeDevice* raw = fake.get();
        // Room for four HD buckets; three slots of two images needs six.
        PooledDevice pool(std::move(fake), 4 * (size_t{1} << 25));
        FrameArenas arenas(pool);

        check(!arenas.prepare(1920, 1080, 2), "a chain that does not fit is refused");
        check(arenas.capacity() == 0, "nothing is left half-reserved");
        check(pool.stats().bytesInUse == 0, "and nothing is left held");
        check(raw->blocks.size() <= 4, "the pool kept what it took, no more");

        // Which is the point of the split: this failed while the chain was
        // being built, not at a frame boundary during playback.
        check(arenas.prepare(1920, 1080, 1), "a chain that does fit prepares");
    }

    // --- asking for more than was declared is loud, not silent --------------
    {
        auto fake = std::make_unique<FakeDevice>();
        PooledDevice pool(std::move(fake), 4 * kGiB);
        FrameArenas arenas(pool);
        check(arenas.prepare(64, 64, 1), "prepared for one");

        arenas.begin(0);
        check(arenas.image().buf != kInvalidBuffer, "the first is served");
        std::fprintf(stderr, "--- one deliberate warning follows ---\n");
        check(arenas.image().buf == kInvalidBuffer,
              "the second is refused rather than invented");
        check(arenas.used() == 1, "and the count does not run past capacity");
        arenas.end();
    }

    // --- re-preparing replaces rather than accumulates ----------------------
    {
        auto fake = std::make_unique<FakeDevice>();
        PooledDevice pool(std::move(fake), 4 * kGiB);
        FrameArenas arenas(pool);

        check(arenas.prepare(1920, 1080, 2), "prepared for two");
        const size_t afterFirst = pool.stats().bytesInUse;
        check(arenas.prepare(1920, 1080, 2), "prepared again, same shape");
        check(pool.stats().bytesInUse == afterFirst,
              "a chain that changes shape re-prepares, it does not accumulate");
    }

    if (failures == 0) {
        std::puts("test_arena: ok");
    }
    return failures == 0 ? 0 : 1;
}
