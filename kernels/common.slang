// Copyright (c) 2026 gpe contributors.
//
// What every kernel needs and nothing else.
//
// There is one internal format -- float32 RGBA, linear, premultiplied -- so a
// buffer is a flat array of float4 and a kernel never branches on depth or
// channel count. Filtering that a texture unit would have done in hardware is
// written here instead, because a CUDA array and a MTLTexture disagree about
// formats and sampling and a buffer of float4 does not.
module common;

/// Where a pixel lives in a strided buffer.
///
/// `stride` is in pixels, matching gpe::Image, so this is the one place the
/// unit is ever multiplied out.
public uint pixelIndex(uint2 xy, uint stride) {
    return xy.y * stride + xy.x;
}

/// A dispatch is rounded up to whole thread groups, so the last group runs off
/// the edge of the picture. Every kernel starts by asking this.
public bool inside(uint2 xy, uint2 size) {
    return xy.x < size.x && xy.y < size.y;
}

/// The size of a thread group, in one place.
///
/// 16x16 is 256 threads: a full warp multiple on NVIDIA and a comfortable
/// simdgroup multiple on Apple, which is the largest number that is a good
/// answer on both without measuring per-kernel.
public static const uint kGroupWidth = 16;
public static const uint kGroupHeight = 16;
