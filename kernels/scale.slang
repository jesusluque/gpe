// Copyright (c) 2026 gpe contributors.
//
// out = in * factor.
//
// The whole point of this kernel is to be uninteresting. It is here to prove
// the road exists -- Slang compiles it, the build embeds it, the backend loads
// it, the dispatch reaches it and the numbers come back doubled -- and the day
// a real operation misbehaves, this is what says whether the road or the
// operation is at fault.
import common;

/// One dispatch's parameters, copied by `dispatch` and matched byte for byte by
/// `gpe::ScaleArgs` on the host. Four-byte members in declaration order: no
/// padding to disagree about, on either backend.
struct ScaleParams {
    uint  width;
    uint  height;
    uint  stride;   // in pixels
    float factor;
};

StructuredBuffer<float4>   src;
RWStructuredBuffer<float4> dst;
ConstantBuffer<ScaleParams> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void scaleMain(uint3 tid: SV_DispatchThreadID) {
    const uint2 xy = tid.xy;
    if (!inside(xy, uint2(params.width, params.height))) {
        return;
    }
    const uint at = pixelIndex(xy, params.stride);
    // Premultiplied throughout, so alpha scales with the colour. Scaling only
    // RGB would quietly unpremultiply the image, which is the kind of thing
    // that shows up three operations later as a dark edge.
    dst[at] = src[at] * params.factor;
}
