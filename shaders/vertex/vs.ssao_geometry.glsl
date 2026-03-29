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

uniform bool invertedNormals;
uniform bool useSSBO;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    vec3 finalPosition;
    Cell cell = cells[gl_InstanceID];
    if (useSSBO) {
        if (cell.isActive == 0) {
            finalPosition = vec3(0);
        }
        else {
            vec3 instancePos = cell.position;
            float radius = cell.radius;
            finalPosition = aPos * radius + instancePos + vec3(0, 5, 0);
        }
    }
    else {
        finalPosition = aPos;
    }
    vec4 viewPos = view * model * vec4(finalPosition, 1.0);
    FragPos = viewPos.xyz;
    TexCoords = aTexCoords;

    mat3 normalMatrix = transpose(inverse(mat3(view * model)));
    Normal = normalMatrix * (invertedNormals ? -aNormal : aNormal);

    gl_Position = projection * viewPos;
}