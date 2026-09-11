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
// compiler does. The accuracy is a property of the grid size -- 33 on a side is
// the usual answer and 65 is the careful one -- and of how the grid is spread:
// evenly over a scene-linear range it wastes nearly every cell on highlights,
// so a log2 shaper spreads it in stops (lutShaper; the host bakes through
// DisplayPass::lutInput, which writes the same shaper on its side).
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
    /// 0 rgb, 1 r, 2 g, 3 b, 4 a, 5 luminance. The same numbering the viewer's
    /// own shader uses, so the two cannot drift into meaning different things.
    uint  channels;
    /// Non-zero to put a checkerboard behind non-opaque pixels rather than
    /// black. Black is ambiguous: a matte that is genuinely black and a matte
    /// that is missing look identical.
    uint  checkerboard;
    float checkerSize;
    /// The other side of a wipe. Zero for none.
    uint  wipeEnabled;
    /// Where the divider is, 0..1 across the output.
    float wipeAt;
    /// The second source, which need not be the same size as the first.
    uint  srcBWidth;
    uint  srcBHeight;
    uint  srcBStride;
    /// Where the output sits in what the checkerboard is anchored to. Added
    /// before the cell is worked out, so the pattern stays put while the
    /// picture moves across it.
    float checkerOriginX;
    float checkerOriginY;
    /// The source coordinate under the target's first pixel, and how many
    /// source pixels a target pixel spans. Above one this averages a box;
    /// below one it interpolates, because that is what magnifying a picture
    /// should look like and what the viewer's own sampler does.
    float viewOriginX;
    float viewOriginY;
    float viewPerPixelX;
    float viewPerPixelY;
    /// How a linear value finds its place in the lattice: 0 spreads
    /// [lutMin, lutMax] evenly, 1 spreads log2 of it evenly (both ends then
    /// linear values above zero). Scene-linear 0..16 on an even 33-cube puts
    /// every value under 0.5 -- all of the shadows and the mid-tones -- in the
    /// first cell; in stops each cell is under a stop wide wherever it lands.
    uint  lutShaper;
    /// Non-zero to add triangular noise of one code value before the picture
    /// is rounded to 8 bits, so a smooth gradient reads as smooth rather than
    /// as bands a code value apart.
    uint  dither;
};

StructuredBuffer<float4>     src;
/// The other side of a wipe. Always bound -- a kernel with an unbound buffer
/// reads a null pointer -- and ignored when `wipeEnabled` is zero, in which
/// case the host binds a single sample rather than a picture.
StructuredBuffer<float4>     srcB;
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

/// Triangular noise in (-1, 1), fixed per pixel: a still frame does not
/// shimmer, and the same pixel reads the same in every frame it is paused on.
float triangularNoise(uint2 xy) {
    uint h = xy.x * 0x8da6b343u ^ xy.y * 0xd8163841u;
    h = (h ^ (h >> 15)) * 0x2c1b3c6du;
    h = (h ^ (h >> 12)) * 0x297a2d39u;
    const float a = float(h & 0xFFFFu) / 65535.0;
    const float b = float((h >> 16) & 0xFFFFu) / 65535.0;
    return a + b - 1.0;
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
    float3 unit;
    if (params.lutShaper == 1u) {
        const float lo = log2(max(params.lutMin, 1e-10));
        const float hi = log2(max(params.lutMax, params.lutMin * 2.0));
        unit = (log2(max(colour, float3(params.lutMin))) - lo) / (hi - lo);
    } else {
        const float span = max(params.lutMax - params.lutMin, 1e-6);
        unit = (colour - params.lutMin) / span;
    }
    float3 t = clamp(unit, 0.0, 1.0) * (size - 1.0);

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

    // Which side of the divider this pixel is on.
    //
    // One dispatch that chooses per pixel, where the viewer's own path is two
    // draws with a scissor. That was not a preference there: the display
    // transform is OCIO's shader and not ours to add a uniform to, so the only
    // way to show two pictures through it was to run it twice. A kernel we own
    // has no such constraint, and one dispatch also means the two sides cannot
    // disagree about anything -- they are the same code with a different index.
    const bool useB = params.wipeEnabled != 0u &&
                      float(xy.x) >= params.wipeAt * float(params.dstWidth);
    const uint sw = useB ? params.srcBWidth : params.srcWidth;
    const uint sh = useB ? params.srcBHeight : params.srcHeight;
    const uint ss = useB ? params.srcBStride : params.srcStride;

    // A box average over the source pixels this output pixel covers.
    //
    // Not a point sample: a viewer showing a 4K plate in a 960-pixel widget
    // point-sampled is a viewer full of aliasing, and every fine detail in the
    // picture -- hair, fabric, noise -- turns into something that crawls when
    // the shot moves. A box is the cheapest filter that does not do that, and
    // at these ratios it is close enough to a proper one that the difference is
    // not what anybody is judging.
    const float leftEdge = params.viewOriginX + float(xy.x) * params.viewPerPixelX;
    const float lowerEdge = params.viewOriginY + float(xy.y) * params.viewPerPixelY;
    const float rightEdge = leftEdge + params.viewPerPixelX;
    const float upperEdge = lowerEdge + params.viewPerPixelY;

    float4 colour;
    if (params.viewPerPixelX <= 1.0 && params.viewPerPixelY <= 1.0) {
        // Magnifying: the nearest source pixel, not a blend of its neighbours.
        //
        // Above 1:1 a compositor's viewer shows the pixels. That is what people
        // zoom in to look at, and it is what this host's own sampler does --
        // GL_NEAREST on both its textures. Interpolating here instead made
        // every frame the GPU path drew visibly softer than the occasional one
        // handed back to the ordinary path, so a playback alternated between
        // blurred letters and clean ones.
        const int ix = clamp(int(floor(leftEdge + params.viewPerPixelX * 0.5)),
                             0, int(sw) - 1);
        const int iy = clamp(int(floor(lowerEdge + params.viewPerPixelY * 0.5)),
                             0, int(sh) - 1);
        colour = useB ? srcB[pixelIndex(uint2(ix, iy), ss)]
                      : src[pixelIndex(uint2(ix, iy), ss)];
    } else {
        // Minifying: a box average over the source pixels this output pixel
        // covers. Not a point sample -- a 4K plate in a 960-pixel widget point
        // sampled is a viewer full of aliasing, and every fine detail in it,
        // hair, fabric, noise, turns into something that crawls when the shot
        // moves.
        const int x0 = max(int(floor(leftEdge)), 0);
        const int y0 = max(int(floor(lowerEdge)), 0);
        const int x1 = min(max(int(ceil(rightEdge)), x0 + 1), int(sw));
        const int y1 = min(max(int(ceil(upperEdge)), y0 + 1), int(sh));
        float4 sum = float4(0.0, 0.0, 0.0, 0.0);
        uint   count = 0;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                sum = sum + (useB ? srcB[pixelIndex(uint2(x, y), ss)]
                                  : src[pixelIndex(uint2(x, y), ss)]);
                count = count + 1;
            }
        }
        colour = count > 0 ? sum / float(count) : float4(0.0, 0.0, 0.0, 0.0);
    }

    // Exposure is a linear scale and belongs before the transfer function;
    // gamma is a display-referred tweak and belongs after. Swapping them is
    // visibly wrong in the shadows.
    colour = float4(colour.xyz * exp2(params.exposure), colour.w);

    // Channel isolation after exposure and before the transform, which is where
    // the viewer's own shader does it. Looking at the red channel means looking
    // at red through the display transform, not at a linear number: a matte
    // judged in linear is judged against the wrong curve.
    if (params.channels == 1u) {
        colour = float4(colour.x, colour.x, colour.x, 1.0);
    } else if (params.channels == 2u) {
        colour = float4(colour.y, colour.y, colour.y, 1.0);
    } else if (params.channels == 3u) {
        colour = float4(colour.z, colour.z, colour.z, 1.0);
    } else if (params.channels == 4u) {
        colour = float4(colour.w, colour.w, colour.w, 1.0);
    } else if (params.channels == 5u) {
        // Rec.709 luminance, which is what the working space's primaries are.
        float luma = dot(colour.xyz, float3(0.2126, 0.7152, 0.0722));
        colour = float4(luma, luma, luma, 1.0);
    }

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

    // And the checkerboard last, over the transformed picture rather than
    // under it -- a checker that went through the display transform would
    // change shade with the view, and it is chrome rather than image.
    if (params.checkerboard != 0u && colour.w < 1.0) {
        const float size = max(params.checkerSize, 1.0);
        const float2 cell = floor((float2(xy) + float2(params.checkerOriginX,
                                                      params.checkerOriginY)) / size);
        const float odd = fmod(cell.x + cell.y, 2.0);
        const float3 checker = lerp(float3(0.18, 0.18, 0.18),
                                    float3(0.28, 0.28, 0.28), odd);
        encoded = lerp(checker, encoded, colour.w);
    }

    if (params.dither != 0u) {
        encoded = encoded + float3(triangularNoise(xy) / 255.0);
    }
    dst[pixelIndex(xy, params.dstStride)] =
        packRgba8(float4(encoded.x, encoded.y, encoded.z, colour.w));
}
