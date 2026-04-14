#version 460
layout(local_size_x = 256) in;

struct Cell {
    vec3 position;
    float foodLevel;
    vec3 voxelCoord;
    float radius;
    int linkStartIndex;
    int linkCount;
    int flatVoxelIndex;
    int isActive;
};

layout(std430, binding = 0) buffer CellBuffer {
    Cell cells[];
};

// Snapshot buffer: flat vec4 array (xyz = position, w unused).
// Using vec4 for std430 alignment simplicity.
layout(std430, binding = 8) buffer PositionSnapshot {
    vec4 posSnapshot[];
};

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cells.length()) return;

    if (cells[id].isActive != 0) {
        posSnapshot[id] = vec4(cells[id].position, 0.0);
    } else {
        posSnapshot[id] = vec4(0.0);
    }
}
