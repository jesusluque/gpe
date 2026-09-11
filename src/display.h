// Copyright (c) 2026 gpe contributors.
//
// The display pass, phase 1: run it on the device, download the small result.
//
// Double buffered, because QRhi exposes no hook for saying "the GPU has
// finished with this texture" -- so the engine cannot know when a presenter is
// done reading. Writing into N+1 while N is being presented removes the
// question: by the time a buffer comes round again a full frame has passed and
// the presenter has long since uploaded it.
#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "gpe/presenter.h"
#include "pool.h"

namespace gpe {

class DisplayPass {
public:
    explicit DisplayPass(PooledDevice& device) : device_(&device) {}
    ~DisplayPass();

    /// How a finished picture gets back to the presenter.
    ///
    /// `Wait`: the frame just dispatched, downloaded synchronously. Its cost is
    /// a wait on the device inside every presented frame -- the one thing the
    /// pool's rules forbid inside a frame.
    ///
    /// `Late`: the newest frame whose picture is already back, and never a
    /// wait on the normal path; a frame of latency. What OpenRV does for SDI
    /// output with a ring of PBOs, read one frame behind. On a unified-memory
    /// device there is no readback at all: the pass writes a buffer the CPU
    /// can already read, and the presenter reads it once the pool has seen
    /// that work retire. On a discrete card it is `downloadAsync`, which on a
    /// backend without an asynchronous one (CUDA today) is the same wait as
    /// `Wait`, a frame later.
    enum class Readback { Wait, Late };

    /// Before `prepare`, which allocates for the mode. `useShared` false reads
    /// back with downloadAsync even where memory is unified: the discrete
    /// card's path, on a machine that would otherwise never take it.
    void setReadback(Readback mode, bool useShared = true) noexcept {
        mode_ = mode;
        useShared_ = useShared;
    }
    [[nodiscard]] Readback readback() const noexcept { return mode_; }

    /// `Late` only: hands the presenter the newest picture that has arrived,
    /// without dispatching anything. For a viewer that stops presenting -- a
    /// pause -- and still wants the last frame it asked for on screen.
    void drain(Presenter& presenter);

    /// `Late` only: pictures that arrived but were never shown, because a
    /// newer one arrived before the presenter was next handed one.
    [[nodiscard]] uint64_t overtaken() const noexcept { return overtaken_; }

    DisplayPass(const DisplayPass&) = delete;
    DisplayPass& operator=(const DisplayPass&) = delete;

    /// Reserves the device buffers and the host side for a target of at most
    /// `maxWidth` x `maxHeight`. Prepare-time; the only place this can fail.
    ///
    /// The maximum rather than the current size, so that resizing a window
    /// during playback does not allocate. A window bigger than what was
    /// reserved is clamped rather than refused -- a viewer that goes black
    /// because somebody dragged a corner is worse than one that shows a
    /// slightly smaller picture for a moment.
    [[nodiscard]] bool prepare(int maxWidth, int maxHeight);

    /// The baked display transform: `size`^3 RGBA samples, red fastest,
    /// covering the linear range [min, max].
    ///
    /// Uploaded once, here, rather than per frame -- it changes when the
    /// display or the view changes, which is when somebody clicks a menu, and
    /// not sixty times a second. `size` under two removes it and the pass falls
    /// back to an sRGB encode.
    ///
    /// The baking is the host's job, not this class's: OCIO belongs to whoever
    /// owns the config, and an engine that linked it would be an engine with an
    /// opinion about colour management.
    [[nodiscard]] bool setLut(const float* rgba, int size, float min, float max);

    /// Transforms, downsamples and packs `source` into the presenter's size,
    /// downloads it, and hands it over.
    ///
    /// With `Readback::Wait` the download is synchronous, and that is the cost
    /// this phase pays: about 8 MB at a 4K widget rather than the 33 MB a
    /// full-size readback would be. `Readback::Late` removes the wait.
    void present(const Image& source, Presenter& presenter,
                 const DisplayControls& controls, uint64_t frame,
                 const DisplayView& view = {});

    /// The same, with a second picture on the right of the divider.
    ///
    /// One dispatch rather than two: the viewer's own path draws twice with a
    /// scissor because the display transform is OCIO's shader and not its to
    /// add a uniform to, and a kernel we own has no such constraint. It also
    /// means the two sides cannot disagree about anything, being the same code
    /// with a different index.
    void presentWipe(const Image& left, const Image& right,
                     Presenter& presenter, const DisplayControls& controls,
                     uint64_t frame, const DisplayView& view = {});

    /// How many frames have been handed over, and how big the last one was.
    [[nodiscard]] uint64_t presented() const noexcept { return presented_; }
    [[nodiscard]] size_t lastBytes() const noexcept { return lastBytes_; }

private:
    enum State : int { kIdle = 0, kPending = 1, kReady = 2, kFailed = 3 };

    struct Target {
        BufferId buffer = kInvalidBuffer;
        // Late readback only.
        unsigned char*             shared = nullptr;   ///< host view of `buffer` (unified memory)
        std::vector<unsigned char> host;               ///< where downloadAsync lands otherwise
        std::atomic<int>           state{kIdle};
        uint64_t                   at = 0;             ///< the dispatch's submission (shared)
        uint64_t                   order = 0;
        uint64_t                   frame = 0;
        int                        width = 0;
        int                        height = 0;
    };

    /// Two for `Wait`, and the reason is in the file header. Three for `Late`:
    /// one being written, one travelling back, one on screen.
    static constexpr int kBuffers = 3;

    void releaseTargets();
    [[nodiscard]] int usableTargets() const noexcept { return mode_ == Readback::Late ? 3 : 2; }
    [[nodiscard]] int idleTarget();
    void dispatchInto(Target& target, const Image& source, const Image& right,
                      const DisplayControls& controls, int width, int height,
                      const DisplayView& view);

    PooledDevice*              device_ = nullptr;
    KernelId                   kernel_ = kInvalidKernel;
    Readback                   mode_ = Readback::Wait;
    bool                       useShared_ = true;
    Target                     targets_[kBuffers];
    int                        at_ = 0;
    uint64_t                   ordered_ = 0;
    uint64_t                   shownOrder_ = 0;
    uint64_t                   overtaken_ = 0;
    int                        maxWidth_ = 0;
    int                        maxHeight_ = 0;
    std::vector<unsigned char> host_;
    BufferId                   lut_ = kInvalidBuffer;
    int                        lutSize_ = 0;
    float                      lutMin_ = 0.0f;
    float                      lutMax_ = 1.0f;
    uint64_t                   presented_ = 0;
    size_t                     lastBytes_ = 0;
};

}   // namespace gpe
