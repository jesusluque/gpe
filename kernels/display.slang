// Copyright (c) 2026 gpe contributors.
//
// The display pass: everything that turns a working image into something a
// window can show, done on the device, so that what crosses to the CPU is the
// small buffer and not the big one.
//
// A 1920x1080 float4 plate is 33 MB. The same picture at a widget's size in
// RGBA8 is about 2 MB, and at 4K widget size about 8 MB. Downsampling and
// packing *before* the download is the difference between a transfer that fits
// in a frame and one that does not -- and it is the whole reason this pass
// exists rather than the viewer doing the work after a full-size readback.
//
// THE DISPLAY TRANSFORM ARRIVES AS A LUT
//
// A real display transform is a view, a display and whatever LUT the show uses,
// and none of that is a formula -- OCIO builds it at run time from a config
// that may have been written this morning. These kernels are compiled at build
// time, so the transform cannot be compiled in.
//
// So it is baked: the host runs OCIO over an identity cube once, when the
// display or the view changes, and hands the result over as a 3D lattice this
// samples trilinearly. That is what every player that is not itself a shader
// compiler does, and the accuracy is a property of the grid size rather than of
// this file -- 33 on a side is the usual answer and 65 is the careful one.
//
// With no lattice bound the pass falls back to an sRGB encode, which is a
// defensible picture rather than a black one and is what a machine with no
// config gets.
import common;

struct DisplayParams {
    uint  srcWidth;
    uint  srcHeight;
    uint  srcStride;
    uint  dstWidth;
    uint  dstHeight;
    uint  dstStride;
    float exposure;    // stops, applied in linear
    float gamma;       // after the transfer function, as a viewer control
    /// Samples on a side of the baked lattice, or 0 for the sRGB fallback.
    uint  lutSize;
    /// The linear range the lattice covers. Values outside it are clamped to
    /// the edge, which is what a display transform does anyway -- there is no
    /// more display past white.
    float lutMin;
    float lutMax;
    uint  pad0;
};

StructuredBuffer<float4>     src;
RWStructuredBuffer<uint>     dst;   // packed RGBA8, one uint per pixel
/// The baked transform, `lutSize`^3 entries, red fastest. Always bound: a
/// kernel with an unbound buffer is a kernel that reads a null pointer, so the
/// host binds a single black sample when there is no lattice and `lutSize`
/// says not to read it.
StructuredBuffer<float4>     lut;
ConstantBuffer<DisplayParams> params;

/// Linear to sRGB. The piecewise curve, not the 2.2 approximation: the toe
/// matters in the shadows, which is where a viewer is looked at hardest.
float encodeSrgb(float linear) {
    float v = max(linear, 0.0);
    return v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

uint packRgba8(float4 colour) {
    // Rounded, not truncated. Truncating loses half a code value everywhere and
    // shows up as a picture that is consistently a shade dark.
    uint r = uint(clamp(colour.x, 0.0, 1.0) * 255.0 + 0.5);
    uint g = uint(clamp(colour.y, 0.0, 1.0) * 255.0 + 0.5);
    uint b = uint(clamp(colour.z, 0.0, 1.0) * 255.0 + 0.5);
    uint a = uint(clamp(colour.w, 0.0, 1.0) * 255.0 + 0.5);
    return r | (g << 8) | (b << 16) | (a << 24);
}

/// Trilinear from the lattice. Eight samples and seven lerps, which is the
/// cheapest interpolation that has no visible banding on a gradient -- nearest
/// on a 33-cube shows the cube's own structure in a sky.
float3 sampleLut(float3 colour) {
    const float size = float(params.lutSize);
    const float span = max(params.lutMax - params.lutMin, 1e-6);
    float3 t = clamp((colour - params.lutMin) / span, 0.0, 1.0) * (size - 1.0);

    const int3 i0 = int3(floor(t));
    const int3 i1 = min(i0 + int3(1, 1, 1), int3(int(size) - 1, int(size) - 1, int(size) - 1));
    const float3 f = t - float3(i0);

    const int n = int(size);
    // Red fastest, which is the order OCIO writes a cube in and the order the
    // host bakes it in. Getting this the other way round swaps red and blue and
    // looks, infuriatingly, almost right.
    const int b000 = (i0.z * n + i0.y) * n + i0.x;
    const int b100 = (i0.z * n + i0.y) * n + i1.x;
    const int b010 = (i0.z * n + i1.y) * n + i0.x;
    const int b110 = (i0.z * n + i1.y) * n + i1.x;
    const int b001 = (i1.z * n + i0.y) * n + i0.x;
    const int b101 = (i1.z * n + i0.y) * n + i1.x;
    const int b011 = (i1.z * n + i1.y) * n + i0.x;
    const int b111 = (i1.z * n + i1.y) * n + i1.x;

    const float3 c00 = lerp(lut[b000].xyz, lut[b100].xyz, f.x);
    const float3 c10 = lerp(lut[b010].xyz, lut[b110].xyz, f.x);
    const float3 c01 = lerp(lut[b001].xyz, lut[b101].xyz, f.x);
    const float3 c11 = lerp(lut[b011].xyz, lut[b111].xyz, f.x);
    return lerp(lerp(c00, c10, f.y), lerp(c01, c11, f.y), f.z);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void displayMain(uint3 tid: SV_DispatchThreadID) {
    const uint2 xy = tid.xy;
    if (!inside(xy, uint2(params.dstWidth, params.dstHeight))) {
        return;
    }

    // A box average over the source pixels this output pixel covers.
    //
    // Not a point sample: a viewer showing a 4K plate in a 960-pixel widget
    // point-sampled is a viewer full of aliasing, and every fine detail in the
    // picture -- hair, fabric, noise -- turns into something that crawls when
    // the shot moves. A box is the cheapest filter that does not do that, and
    // at these ratios it is close enough to a proper one that the difference is
    // not what anybody is judging.
    const uint x0 = xy.x * params.srcWidth / params.dstWidth;
    const uint x1 = max(x0 + 1u, (xy.x + 1u) * params.srcWidth / params.dstWidth);
    const uint y0 = xy.y * params.srcHeight / params.dstHeight;
    const uint y1 = max(y0 + 1u, (xy.y + 1u) * params.srcHeight / params.dstHeight);

    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    uint   count = 0;
    for (uint y = y0; y < y1 && y < params.srcHeight; ++y) {
        for (uint x = x0; x < x1 && x < params.srcWidth; ++x) {
            sum = sum + src[pixelIndex(uint2(x, y), params.srcStride)];
            count = count + 1;
        }
    }
    float4 colour = count > 0 ? sum / float(count) : float4(0.0, 0.0, 0.0, 0.0);

    // Exposure is a linear scale and belongs before the transfer function;
    // gamma is a display-referred tweak and belongs after. Swapping them is
    // visibly wrong in the shadows.
    colour = float4(colour.xyz * exp2(params.exposure), colour.w);

    // The baked transform if there is one, the sRGB curve if there is not.
    float3 encoded = params.lutSize >= 2
                         ? sampleLut(colour.xyz)
                         : float3(encodeSrgb(colour.x), encodeSrgb(colour.y),
                                  encodeSrgb(colour.z));
    if (params.gamma != 1.0) {
        encoded = pow(max(encoded, float3(0.0, 0.0, 0.0)),
                      float3(1.0 / params.gamma, 1.0 / params.gamma,
                             1.0 / params.gamma));
    }

    dst[pixelIndex(xy, params.dstStride)] =
        packRgba8(float4(encoded.x, encoded.y, encoded.z, colour.w));
}
