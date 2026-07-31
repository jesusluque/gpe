#version 440
// Straight through. Everything about how the picture should look -- exposure,
// the transfer function, the downsample -- happened in the display pass on the
// engine's side, where it is one kernel that both backends share rather than
// two shaders that have to be kept saying the same thing.
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 fragColor;
layout(binding = 0) uniform sampler2D uPicture;
void main() {
    fragColor = texture(uPicture, vTexCoord);
}
