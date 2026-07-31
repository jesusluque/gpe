// Copyright (c) 2026 gpe contributors.
//
// Where a finished frame goes, and the seam that keeps Qt out of the engine.
//
// Nothing below this header knows what a window is. `Presenter` is our own
// interface, and the QRhi implementation lives on the other side of it in
// viewer/ -- which is not tidiness, it is insulation: QRhi offers no source or
// binary compatibility guarantee between Qt versions, so the Qt version is
// pinned in the build and updating it is treated as a change with risk. When
// that update breaks something, what it can break is one file.
//
// DIRECTION
//
// QRhi owns the display texture and gpe writes into it. Not the other way
// round: wrapping a gpe buffer with QRhi's createFrom() would make the engine's
// allocator responsible for something Qt's renderer schedules, and the two have
// no shared vocabulary for saying when either has finished with it.
//
// PHASE 1 IS A DOWNLOAD
//
// This interface takes bytes, because the first implementation is a readback:
// the display pass runs on the device -- transform, downsample, pack -- and
// what crosses to the CPU is the widget-sized 8-bit picture, about 8 MB at 4K
// widget size rather than the 33 MB of a full-size float plate.
//
// Interop comes later, is an optimisation, and is measured before it is kept:
// on macOS by adopting QRhi's MTLDevice so that compute and render share one
// device and there is no cross-synchronisation at all, and on Windows through
// D3D11 with cudaGraphicsD3D11RegisterResource -- which has to check at startup
// that CUDA and D3D11 are on the same physical GPU, because on a laptop with an
// integrated and a discrete chip they routinely are not. Both go behind this
// same interface, which is why it takes a frame rather than a texture.
#pragma once

#include <cstddef>
#include <cstdint>

namespace gpe {

class Presenter {
public:
    virtual ~Presenter() = default;

    /// The size the display pass should produce, in pixels.
    ///
    /// Asked every frame rather than set once: a window is resized while it is
    /// playing, and a pass that produced last size's picture would either
    /// stretch or tear.
    virtual void targetSize(int& width, int& height) const = 0;

    /// One frame, packed RGBA8, `width * height` pixels with no row padding.
    ///
    /// The bytes belong to the caller and are valid only for the duration of
    /// this call. A presenter that needs them afterwards copies them -- which
    /// is what uploading to a texture does anyway.
    virtual void present(const void* rgba8, int width, int height,
                         uint64_t frame) = 0;
};

/// The viewer's controls, passed to the display pass.
///
/// Exposure before the transfer function and gamma after it, which is the order
/// that makes them mean what a colourist expects.
struct DisplayControls {
    float exposure = 0.0f;   ///< stops
    float gamma = 1.0f;      ///< display-referred, after the transform

    /// Which channels reach the screen. The numbering is the viewer's own, so
    /// that the engine and the interface cannot drift into meaning different
    /// things by the same number.
    enum class Channels : uint32_t {
        Rgb = 0,
        Red = 1,
        Green = 2,
        Blue = 3,
        Alpha = 4,
        Luminance = 5,
    };
    Channels channels = Channels::Rgb;

    /// A checkerboard behind non-opaque pixels rather than black. Black is
    /// ambiguous: a matte that is genuinely black and a matte that is missing
    /// look the same.
    bool  checkerboard = true;
    float checkerSize = 16.0f;

    /// The divider of an A/B wipe, 0..1 across the output. Only read when the
    /// pass is given a second image.
    float wipeAt = 0.5f;
};

}   // namespace gpe
