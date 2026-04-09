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
    vec3 Normal = texture(gNormal, TexCoords).rgb;
    vec3 Diffuse = texture(gAlbedo, TexCoords).rgb;
    float AmbientOcclusion = 1;
    if (enableSSAO)
        AmbientOcclusion = texture(ssao, TexCoords).r;

    float aoLift = mix(0.05, 1.0, AmbientOcclusion);
    vec3 ambient = 0.55 * Diffuse * aoLift;
    vec3 lighting = ambient;
    if (enablePhong) {
        vec3 viewDir  = normalize(-FragPos); // assuming camera is at origin
        // diffuse
        vec3 lightDir = normalize(light.Position - FragPos);
        vec3 diffuse = 1.25 * max(dot(Normal, lightDir), 0.0) * Diffuse * light.Color;
        // specular (Blinn-Phong)
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(Normal, halfwayDir), 0.0), 8.0);
        vec3 specular = 0.35 * light.Color * spec;
        // attenuation
        float distance = length(light.Position - FragPos);
        float attenuation = 1.0 / (1.0 + light.Linear * distance + light.Quadratic * distance * distance);
        diffuse *= attenuation;
        specular *= attenuation;
        lighting += diffuse + specular;
        // Let AO influence direct light a bit so occlusion is perceptible beyond ambient only.
        lighting *= mix(0.4, 1.0, AmbientOcclusion);
        lighting *= 1.2;
    }

    FragColor = vec4(lighting, 1.0);
}
