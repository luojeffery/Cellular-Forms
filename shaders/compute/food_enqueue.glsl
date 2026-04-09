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
uniform float foodPerFrame;
uniform float foodThreshold;
uniform int maxDivisionQueue;

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    if (numActiveCells >= numCells) {
        return;
    }

    // Uniform feeding: every active cell gets the same amount each frame.
    cells[id].foodLevel += foodPerFrame;

    // Enqueue if ready to divide.
    if (cells[id].foodLevel >= foodThreshold) {
        uint queueIndex = atomicAdd(divisionQueueCount, 1u);
        if (queueIndex < maxDivisionQueue) {
            divisionQueue[queueIndex] = id;
            cells[id].foodLevel = 0.0; // Reset so each parent divides at most once per growth step.
        } else {
            atomicAdd(divisionQueueCount, -1u);
        }
    }
}
