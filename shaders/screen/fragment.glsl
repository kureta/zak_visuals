#version 460

uniform sampler2D texture0;
out vec4 fragColor;
in vec2 uv;

void main() {
    fragColor = texture2D(texture0, uv);
}
