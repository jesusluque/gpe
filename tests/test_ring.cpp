// The queue between two stages.
//
// The properties that matter are the ones that only show up under load: that it
// is really bounded, that a full queue pushes back rather than growing, and
// that closing it drains rather than dropping. A ring that works when it is
// never full is a ring nobody has tested.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "ring.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

}   // namespace

int main() {
    using namespace gpe;

    // --- bounded, and it says so ------------------------------------------
    {
        Ring<int> ring(3);
        check(ring.capacity() == 3, "the capacity is what was asked for");
        check(ring.tryPush(1) && ring.tryPush(2) && ring.tryPush(3),
              "three fit");
        check(ring.size() == 3, "and three are in it");
        check(!ring.tryPush(4), "the fourth is refused rather than queued");
        check(ring.size() == 3, "and did not grow");

        check(ring.pop().value() == 1, "first in, first out");
        check(ring.tryPush(4), "and there is room again");
        ring.close();
    }

    // --- closing drains, it does not drop ----------------------------------
    {
        // A consumer has to be able to finish what was queued before it learns
        // to stop, or the last frames of a playback disappear.
        Ring<int> ring(4);
        check(ring.tryPush(10) && ring.tryPush(20), "two queued");
        ring.close();
        check(!ring.tryPush(30), "nothing new is accepted after closing");
        check(ring.pop().value() == 10, "but what was queued still comes out");
        check(ring.pop().value() == 20, "all of it");
        check(!ring.pop().has_value(), "and then it reports the end");
    }

    // --- a full ring makes the producer wait -------------------------------
    {
        // The whole point of bounding it: a producer that outruns the consumer
        // is slowed to the consumer's rate instead of eating memory.
        Ring<int> ring(2);
        std::atomic<int> produced{0};
        std::atomic<bool> blocked{false};

        std::thread producer([&] {
            for (int i = 0; i < 8; ++i) {
                if (i == 2) {
                    blocked.store(true);   // about to fill it
                }
                if (!ring.push(i)) {
                    return;
                }
                produced.fetch_add(1);
            }
        });

        // Give it every chance to run ahead. If it were unbounded it would be
        // finished by now.
        while (!blocked.load()) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        check(produced.load() <= 3,
              "a producer with a full queue is stopped, not buffered");

        int seen = 0;
        while (seen < 8) {
            if (!ring.pop().has_value()) {
                break;
            }
            ++seen;
        }
        producer.join();
        check(seen == 8, "and everything gets through once the consumer drains");
        ring.close();
    }

    // --- two threads, every item once, in order ----------------------------
    {
        constexpr int kCount = 50000;
        Ring<int>     ring(3);
        std::vector<int> got;
        got.reserve(kCount);

        std::thread consumer([&] {
            while (true) {
                std::optional<int> value = ring.pop();
                if (!value.has_value()) {
                    return;
                }
                got.push_back(*value);
            }
        });
        for (int i = 0; i < kCount; ++i) {
            check(ring.push(i) || i == kCount, "");
        }
        ring.close();
        consumer.join();

        check(got.size() == static_cast<size_t>(kCount),
              "every item crossed exactly once");
        bool ordered = true;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != static_cast<int>(i)) {
                ordered = false;
                break;
            }
        }
        check(ordered, "and in the order it was sent");
        check(ring.pushed() == ring.popped(), "the counters agree");
    }

    // --- close wakes a blocked consumer ------------------------------------
    {
        // Otherwise shutting a pipeline down deadlocks on whichever stage
        // happened to be waiting.
        Ring<int>         ring(2);
        std::atomic<bool> returned{false};
        std::thread waiter([&] {
            (void)ring.pop();
            returned.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        check(!returned.load(), "the consumer is waiting");
        ring.close();
        waiter.join();
        check(returned.load(), "and closing releases it");
    }

    if (failures == 0) {
        std::puts("test_ring: ok");
    }
    return failures == 0 ? 0 : 1;
}
