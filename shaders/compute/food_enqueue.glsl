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
    uint numActiveCells;      // Monotonic high-water mark (next unallocated slot)
    uint divisionQueueCount;
    uint trueActiveCount;     // Actual number of active cells this frame
};
layout(std430, binding = 7) buffer DivisionQueue {
    uint divisionQueue[];
};

uniform int numCells;
uniform float foodPerFrame;
uniform float foodThreshold;
uniform int maxDivisionQueue;
uniform uint frameCounter;

// Simple hash for per-cell randomization.
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

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    // Count this active cell for the true active count this frame.
    // trueActiveCount is reset to 0 by CPU each frame before this dispatch.
    atomicAdd(trueActiveCount, 1u);

    // Note: numActiveCells is a monotonic high-water mark (never decremented).
    // Capacity check happens in process_division_queue's CAS allocator, not here.
    // Cells keep accumulating food even near capacity — division simply won't
    // succeed if no slots remain.

    // Random per-cell food: base rate * random multiplier in [0.5, 1.5].
    // This staggers division timing so cells don't all divide simultaneously,
    // which would overwhelm the CAS allocation and cause most divisions to fail.
    uint seed = hash(id * 0x9E3779B9u + frameCounter * 0x517CC1B7u);
    float randomMul = 0.5 + float(seed & 0xFFFFu) / float(0xFFFFu);
    cells[id].foodLevel += foodPerFrame * randomMul;

    // Gate: only cells with enough links to split cleanly may divide.
    // Cells with < 3 neighbors can't form a meaningful cleavage plane.
    if (cells[id].linkCount < 3) return;

    // Enqueue if ready to divide. Do NOT reset food here — food is only
    // reset by process_division_queue on SUCCESSFUL division. This way,
    // cells whose division fails (capacity, CAS contention) keep their
    // food and retry next frame instead of losing progress.
    if (cells[id].foodLevel >= foodThreshold) {
        uint queueIndex = atomicAdd(divisionQueueCount, 1u);
        if (queueIndex < uint(maxDivisionQueue)) {
            divisionQueue[queueIndex] = id;
        } else {
            atomicAdd(divisionQueueCount, 0xFFFFFFFFu); // -1 as uint
        }
    }
}
