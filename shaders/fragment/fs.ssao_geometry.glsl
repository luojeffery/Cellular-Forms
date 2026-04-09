#version 460 core
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec3 gAlbedo;

uniform sampler2D texture_diffuse1;
uniform bool useColor;
uniform vec3 color;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;
in vec3 CellColor;
uniform bool useSSBO;

void main()
{
    gPosition = FragPos;
    gNormal = normalize(Normal);
    if (useSSBO)
        gAlbedo.rgb = CellColor;
    else if (useColor)
        gAlbedo.rgb = color;
    else
        gAlbedo.rgb = texture(texture_diffuse1, TexCoords).rgb;
}