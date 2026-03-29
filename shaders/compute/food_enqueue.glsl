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
layout(std430, binding = 5) buffer GlobalCounts {
    uint numActiveCells;
    uint divisionQueueCount;
};
layout(std430, binding = 7) buffer DivisionQueue {
    uint divisionQueue[];
};

uniform int numCells;
uniform int foodThreshold;
uniform int maxDivisionQueue;

uint hash(uint x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

float randFloat(uint seed) {
    return float(hash(seed) & 0xFFFFu) / float(0xFFFFu); // in [0.0, 1.0)
}

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    // Add food increment
    float r = randFloat(id + uint(gl_WorkGroupID.x) * 1234u);
    float increment = 1.0 + 9.0 * r;

    if (numActiveCells < numCells) {
        cells[id].foodLevel += increment;
    }

    // Enqueue if ready to divide
    if (cells[id].foodLevel >= foodThreshold && numActiveCells < numCells) {
        uint queueIndex = atomicAdd(divisionQueueCount, 1u);
        if (queueIndex < maxDivisionQueue) {
            divisionQueue[queueIndex] = id;
            cells[id].foodLevel = 0.0; // Reset food so we don't enqueue again next frame
        } else {
            // Queue full, undo the increment
            atomicAdd(divisionQueueCount, -1u);
        }
    }
}
