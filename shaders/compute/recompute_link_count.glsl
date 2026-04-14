#version 460
layout(local_size_x = 256) in;

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
layout(std430, binding = 1) buffer LinkBuffer {
    uint links[];
};

const uint EMPTY = 0xFFFFFFFFu;

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) {
        return;
    }

    // Count non-EMPTY links
    int linkBase = cells[id].linkStartIndex;
    int count = 0;

    for (int i = 0; i < 8; i++) {
        if (links[linkBase + i] != EMPTY) {
            count++;
        }
    }

    // Write the recomputed count
    cells[id].linkCount = count;
}
