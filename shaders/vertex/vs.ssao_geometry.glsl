#version 460 core

struct Cell {
    vec3 position;       // 16 bytes (vec3 + padding)
    float foodLevel;     // 4  (included in above 16)
    vec3 voxelCoord;     // 16 bytes
    float radius;        // 4
    int linkStartIndex;  // 4
    int linkCount;       // 4
    int flatVoxelIndex;  // 4
    int isActive;        // 4 (to make total = 64)
};



layout(std430, binding = 0) buffer CellBuffer {
    Cell cells[];
};

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec2 TexCoords;
out vec3 Normal;
out vec3 CellColor;

uniform bool invertedNormals;
uniform bool useSSBO;
uniform vec3 cellColorCentroid;
uniform float cellColorMaxDistance;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform bool enableClip;
uniform vec4 clipPlane;  // (nx, ny, nz, d) in world space: dot(n, P) + d >= 0 => keep

void main()
{
    vec3 finalPosition;
    CellColor = vec3(1.0);
    Cell cell = cells[gl_InstanceID];
    if (useSSBO) {
        if (cell.isActive == 0) {
            finalPosition = vec3(0);
        }
        else {
            vec3 instancePos = cell.position;
            float radius = cell.radius;

            // Per-instance cross-section: if the cell CENTER is on the
            // clipped side of the plane, collapse the entire sphere to a
            // degenerate point so no partial cuts are visible.
            vec3 cellWorldCenter = (model * vec4(instancePos + vec3(0, 5, 0), 1.0)).xyz;
            if (enableClip && dot(clipPlane.xyz, cellWorldCenter) + clipPlane.w < 0.0) {
                finalPosition = vec3(0);
            } else {
                float normalizedDistance = clamp(length(instancePos - cellColorCentroid) / max(cellColorMaxDistance, 1e-4), 0.0, 1.0);
                vec3 innerColor = vec3(1.0, 1.0, 0.0);
                vec3 outerColor = vec3(0.0, 0.55, 1.0);
                CellColor = mix(innerColor, outerColor, normalizedDistance);
                finalPosition = aPos * radius + instancePos + vec3(0, 5, 0);
            }
        }
    }
    else {
        finalPosition = aPos;
    }
    vec4 worldPos = model * vec4(finalPosition, 1.0);
    vec4 viewPos = view * worldPos;
    FragPos = viewPos.xyz;
    TexCoords = aTexCoords;

    mat3 normalMatrix = transpose(inverse(mat3(view * model)));
    Normal = normalMatrix * (invertedNormals ? -aNormal : aNormal);

    gl_Position = projection * viewPos;
}