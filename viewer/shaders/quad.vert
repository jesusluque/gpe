#version 440
// A full-screen triangle, not a quad.
//
// Three vertices instead of six, and no diagonal seam down the middle where
// two triangles meet -- along that seam the GPU shades some pixels twice and
// the interpolation is discontinuous, which on a picture with fine detail is a
// visible line. The triangle is bigger than the screen and gets clipped.
layout(location = 0) out vec2 vTexCoord;
void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vTexCoord = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
