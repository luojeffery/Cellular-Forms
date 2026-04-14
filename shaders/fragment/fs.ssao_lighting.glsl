#version 460 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D ssao;
uniform bool enableSSAO;
struct Light {
    vec3 Position;
    vec3 Color;

    float Linear;
    float Quadratic;
};
uniform Light light;
uniform bool enablePhong;

void main()
{
    vec3 FragPos = texture(gPosition, TexCoords).rgb;
    if (FragPos.z == 0.0) discard;

    vec3 Normal = texture(gNormal, TexCoords).rgb;
    vec3 Diffuse = texture(gAlbedo, TexCoords).rgb;
    
    float AmbientOcclusion = 1.0;
    if (enableSSAO)
        AmbientOcclusion = texture(ssao, TexCoords).r;

    float ambientIntensity = 1.0;
    vec3 color = Diffuse * ambientIntensity * AmbientOcclusion;
    
    if (enablePhong) {
        vec3 up = vec3(0.0, 1.0, 0.0);
        float hemiDiffuse = max(dot(Normal, up), 0.0) * 0.15;
        color += Diffuse * hemiDiffuse * light.Color * AmbientOcclusion;
    }

    FragColor = vec4(color, 1.0);
}
