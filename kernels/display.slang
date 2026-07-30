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
// WHAT IS NOT HERE YET
//
// The display transform is an sRGB encode, and it should be an OCIO display
// transform: a real one carries the view, the display and whatever LUT the
// show uses, and none of that is a formula. Replacing this function is where
// that goes, and the shape of the pass does not change when it does -- the
// input is still scene-linear float4, the output is still packed 8-bit at the
// widget's size.
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
};

StructuredBuffer<float4>     src;
RWStructuredBuffer<uint>     dst;   // packed RGBA8, one uint per pixel
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

    float3 encoded = float3(encodeSrgb(colour.x), encodeSrgb(colour.y),
                            encodeSrgb(colour.z));
    if (params.gamma != 1.0) {
        encoded = pow(max(encoded, float3(0.0, 0.0, 0.0)),
                      float3(1.0 / params.gamma, 1.0 / params.gamma,
                             1.0 / params.gamma));
    }

    dst[pixelIndex(xy, params.dstStride)] =
        packRgba8(float4(encoded.x, encoded.y, encoded.z, colour.w));
}
