// Copyright (c) 2026 gpe contributors.
//
// The vocabulary of the engine. Deliberately tiny: everything that crosses the
// backend boundary is here, and nothing here knows what a backend is.
#pragma once

#include <cstddef>
#include <cstdint>

namespace gpe {

/// A device allocation, opaque on purpose.
///
/// An integer rather than a pointer because a CUDA allocation and a
/// MTL::Buffer* are not the same kind of thing and never will be. The handle
/// costs a lookup and buys a client that compiles unchanged on both.
using BufferId = uint32_t;

/// A compiled kernel, looked up by name.
using KernelId = uint32_t;

/// Zero is nothing, for both. Returned by `alloc` and `load` when they fail.
///
/// A sentinel rather than an error type because these two are the only calls
/// that can fail in a way the caller can act on, and the interface is fixed at
/// eight virtuals -- there is no room for an out-parameter and no appetite for
/// a ninth method to read the last error from.
inline constexpr BufferId kInvalidBuffer = 0;
inline constexpr KernelId kInvalidKernel = 0;

/// The one internal format: float32 RGBA, linear, premultiplied.
///
/// Not a variant, not a template parameter. Every buffer in the engine is this,
/// and conversion happens at the edges where a file format or a display forces
/// it. Kernels can therefore assume a stride in whole pixels and never branch
/// on depth.
inline constexpr int kChannels = 4;
inline constexpr size_t kBytesPerPixel = sizeof(float) * kChannels;

/// A picture living in a device buffer.
///
/// `stride` is in pixels, not bytes: at one fixed format the two differ by a
/// constant, and pixels are what the kernel indexes by.
struct Image {
    BufferId buf = kInvalidBuffer;
    int      w = 0;
    int      h = 0;
    int      stride = 0;

    [[nodiscard]] constexpr size_t bytes() const noexcept {
        return static_cast<size_t>(stride) * static_cast<size_t>(h) *
               kBytesPerPixel;
    }
};

/// A dispatch size, in threads -- not in groups.
///
/// Threads rather than groups because the two backends disagree about what a
/// group is and agree about what a thread is. Each backend divides by its own
/// group size and handles the remainder, so a client that asks for 1920x1080
/// gets 1920x1080 threads whichever machine it is on.
struct Grid {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

}   // namespace gpe
