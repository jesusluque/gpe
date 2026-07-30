// The drop policy: what is shown, and what is said, when a frame is late.
#include <cstdio>
#include <string>

#include "drop.h"

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

    std::fprintf(stderr, "--- four deliberate startup lines follow ---\n");

    // The default leaves the graded picture alone and still reports the drop.
    // Those two together are the whole reason it is the default.
    {
        DropPolicy policy;
        check(policy.signal() == DropSignal::RepeatMarked, "the default repeats");
        check(!policy.marksThePicture(), "and does not write into the picture");
        check(policy.reportsOutOfBand(), "but does report the drop");
    }

    // The only mode that loses the information entirely, which is why it has to
    // be asked for by name.
    {
        DropPolicy policy(DropSignal::RepeatSilent);
        check(!policy.marksThePicture(), "silent repeats leave the picture");
        check(!policy.reportsOutOfBand(), "and say nothing anywhere");
    }

    // Destructive, deliberately: a measurement with no side channel to read.
    {
        DropPolicy policy(DropSignal::ColorBar);
        check(policy.marksThePicture(), "a colour bar is painted in");
        check(policy.reportsOutOfBand(), "and still reported out of band");
        check(kDropBarColour[0] == 1.0f && kDropBarColour[1] == 0.0f &&
                  kDropBarColour[2] == 1.0f,
              "flat magenta: no graded frame is this by accident");
    }

    // Black works, is never the default, and says why on the way in.
    {
        DropPolicy policy(DropSignal::Black);
        check(policy.marksThePicture(), "black is painted in too");
        check(DropPolicy{}.signal() != DropSignal::Black,
              "and is never what you get without asking");
    }

    // A run of one is a hiccup; a run of thirty is a player that is not keeping
    // up. Counting them apart is what lets somebody tell the difference.
    {
        DropPolicy policy;
        policy.onFrame();
        check(policy.onDrop() == 1, "one drop is a run of one");
        check(policy.onDrop() == 2, "two in a row is a run of two");
        check(policy.onDrop() == 3, "three in a row is a run of three");
        policy.onFrame();
        check(policy.repeatRun() == 0, "a frame that arrives ends the run");
        check(policy.drops() == 3, "and the total is still three");
        check(policy.frames() == 2, "with two frames delivered");
    }

    if (failures == 0) {
        std::puts("test_drop: ok");
    }
    return failures == 0 ? 0 : 1;
}
