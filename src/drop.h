// Copyright (c) 2026 gpe contributors.
//
// What the player shows when a frame does not arrive in time.
//
// The clock wins: a late frame is not waited for, because waiting makes the
// next one late too and the timeline slips against the sound. What is shown
// instead is a policy, and it is a policy rather than a constant because the
// right answer differs between a colourist checking a grade and a machine
// measuring conformance.
//
// THE SIGNAL BELONGS OUTSIDE THE PICTURE
//
// Wherever there is somewhere else to put it, the fact that a frame was
// repeated is reported without touching the pixels: a non-destructive burn-in
// alongside the timecode and frame number, and on an SDI output a per-frame
// flag in VANC correlated with the RP188 timecode. A downstream tool can then
// count drops exactly, and the picture that was graded is the picture that was
// shown.
//
// Only when a destructive mark is genuinely wanted -- an automated measurement
// with no side channel to read -- does a mode write into the image, and then it
// is a flat colour bar rather than black. Both are equally easy for a detector
// to find, and black is ambiguous with two things that happen legitimately: a
// signal that has dropped out, and a fade the material itself contains. Magenta
// and green appear in no graded picture by accident.
#pragma once

#include <cstdint>
#include <cstdio>

namespace gpe {

/// What to show in place of a frame that missed its deadline.
enum class DropSignal {
    /// The previous frame again, with the repeat reported out of band. The
    /// default: nothing is written into the picture and nothing is hidden.
    RepeatMarked,
    /// The previous frame again, reported nowhere. For a demonstration where a
    /// burn-in would be in the way, and it is the only mode that loses the
    /// information entirely.
    RepeatSilent,
    /// A flat colour bar over the picture. Destructive, and chosen over black
    /// for exactly that reason -- it cannot be confused with signal loss.
    ColorBar,
    /// Black. Never the default; see below.
    Black,
};

/// The colour a ColorBar drop is painted, premultiplied linear RGBA.
///
/// Magenta: no graded picture contains a frame of flat magenta by accident, and
/// it survives a chroma subsample that would blur a subtler marker away.
inline constexpr float kDropBarColour[4] = {1.0f, 0.0f, 1.0f, 1.0f};

/// Counts drops and says what to show.
///
/// Deliberately not a policy the render code branches on: the render code asks
/// what to do and does it, and every decision about *why* lives here.
class DropPolicy {
public:
    explicit DropPolicy(DropSignal signal = DropSignal::RepeatMarked)
        : signal_(signal) {
        // Logged at startup, always. A player behaving unexpectedly is a
        // player somebody has to be able to ask "what mode is this in", and
        // the answer belongs in the log rather than in a memory of what was
        // configured last week.
        std::fprintf(stderr, "gpe: drop signal = %s\n", name(signal));
        if (signal == DropSignal::Black) {
            // Black is ambiguous with a signal that has dropped out and with a
            // fade the material contains, so choosing it leaves a trace that
            // says so rather than a line that looks like every other setting.
            std::fprintf(stderr,
                         "gpe: WARNING -- black drops cannot be told apart "
                         "from signal loss or from a fade in the material. "
                         "ColorBar measures the same thing without the "
                         "ambiguity.\n");
        }
    }

    [[nodiscard]] DropSignal signal() const noexcept { return signal_; }

    /// A frame arrived in time.
    void onFrame() noexcept {
        ++frames_;
        repeatRun_ = 0;
    }

    /// A frame did not. Returns how many times in a row now.
    uint64_t onDrop() noexcept {
        ++drops_;
        return ++repeatRun_;
    }

    /// True when the drop has to be painted into the picture. False for the two
    /// repeat modes, where the picture is left exactly as it was graded.
    [[nodiscard]] bool marksThePicture() const noexcept {
        return signal_ == DropSignal::ColorBar || signal_ == DropSignal::Black;
    }

    /// True when the drop is reported out of band -- burn-in, VANC, log.
    [[nodiscard]] bool reportsOutOfBand() const noexcept {
        return signal_ != DropSignal::RepeatSilent;
    }

    [[nodiscard]] uint64_t frames() const noexcept { return frames_; }
    [[nodiscard]] uint64_t drops() const noexcept { return drops_; }
    /// How many of the last frames in a row were repeats. A run of one is a
    /// hiccup; a run of thirty is a player that is not keeping up at all, and
    /// the two want different answers from whoever is watching.
    [[nodiscard]] uint64_t repeatRun() const noexcept { return repeatRun_; }

    [[nodiscard]] static const char* name(DropSignal signal) noexcept {
        switch (signal) {
            case DropSignal::RepeatMarked:
                return "RepeatMarked";
            case DropSignal::RepeatSilent:
                return "RepeatSilent";
            case DropSignal::ColorBar:
                return "ColorBar";
            case DropSignal::Black:
                return "Black";
        }
        return "?";
    }

private:
    DropSignal signal_ = DropSignal::RepeatMarked;
    uint64_t   frames_ = 0;
    uint64_t   drops_ = 0;
    uint64_t   repeatRun_ = 0;
};

}   // namespace gpe
