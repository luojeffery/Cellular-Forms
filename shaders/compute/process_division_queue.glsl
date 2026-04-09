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
layout(std430, binding = 5) buffer GlobalCounts {
    uint numActiveCells;
    uint divisionQueueCount;
};
layout(std430, binding = 7) buffer DivisionQueue {
    uint divisionQueue[];
};

uniform int numCells;

const uint EMPTY = 0xFFFFFFFFu;

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

vec3 safeNormalize(vec3 v, vec3 fallback) {
    float len2 = dot(v, v);
    if (len2 > 1e-8) {
        return v * inversesqrt(len2);
    }
    return fallback;
}

void main() {
    uint queueIndex = gl_GlobalInvocationID.x;
    
    // Check if this thread should process a division
    if (queueIndex >= divisionQueueCount) return;
    
    uint parentId = divisionQueue[queueIndex];
    if (parentId >= cells.length() || cells[parentId].isActive == 0) return;

    // Allocate new cell index
    uint newIndex = atomicAdd(numActiveCells, 1u);
    if (newIndex >= numCells) {
        // Out of bounds, undo
        atomicAdd(numActiveCells, -1u);
        return;
    }

    // Initialize daughter cell
    cells[newIndex].isActive = 1;
    float rx = randFloat(parentId * 7u + 0u) * 2.0 - 1.0;
    float ry = randFloat(parentId * 7u + 1u) * 2.0 - 1.0;
    float rz = randFloat(parentId * 7u + 2u) * 2.0 - 1.0;
    float tx = randFloat(parentId * 7u + 3u) * 2.0 - 1.0;
    float ty = randFloat(parentId * 7u + 4u) * 2.0 - 1.0;
    float tz = randFloat(parentId * 7u + 5u) * 2.0 - 1.0;

    int parentBase = cells[parentId].linkStartIndex;
    vec3 parentPos = cells[parentId].position;

    // Estimate outward surface normal from local neighborhood to keep growth on the shell.
    vec3 normalSum = vec3(0.0);
    for (int i = 0; i < 6; i++) {
        uint neighbor = links[parentBase + i];
        if (neighbor == EMPTY || neighbor >= cells.length() || cells[neighbor].isActive == 0) {
            continue;
        }
        normalSum += safeNormalize(parentPos - cells[neighbor].position, vec3(0.0, 1.0, 0.0));
    }

    vec3 radialFallback = safeNormalize(parentPos, vec3(0.0, 1.0, 0.0));
    vec3 outward = safeNormalize(normalSum, radialFallback);
    vec3 randomVec = safeNormalize(vec3(rx, ry, rz), radialFallback);

    // Tangential jitter helps break symmetry without sending daughter deep into the volume.
    vec3 tangent = randomVec - dot(randomVec, outward) * outward;
    tangent = safeNormalize(tangent, vec3(1.0, 0.0, 0.0));
    float tangentSign = (randFloat(parentId * 7u + 6u) < 0.5) ? -1.0 : 1.0;

    vec3 splitOffset = 0.08 * outward + 0.03 * tangentSign * tangent;
    cells[newIndex].position = parentPos + splitOffset;
    cells[newIndex].radius = 0.3;
    cells[newIndex].voxelCoord = cells[parentId].voxelCoord;
    cells[newIndex].flatVoxelIndex = cells[parentId].flatVoxelIndex;
    cells[newIndex].linkStartIndex = int(newIndex * 6);
    cells[newIndex].linkCount = 0; // Will be recomputed later
    cells[newIndex].foodLevel = 0.0;
    
    // Initialize daughter's link slots to EMPTY
    for (int i = 0; i < 6; ++i) {
        links[cells[newIndex].linkStartIndex + i] = EMPTY;
    }

    // Step 1: Sever half of parent's links and assign to daughter
    int daughterBase = cells[newIndex].linkStartIndex;
    int linksToSever = cells[parentId].linkCount / 2;
    uint neighbors[6]; // Max 6 neighbors
    int neighborCount = 0;

    for (int i = 0; i < 6; i++) {
        uint neighbor = links[parentBase + i];
        if (neighbor == EMPTY || neighborCount >= linksToSever) {
            // Leave daughter slot empty or already severed enough
            continue;
        }
        
        // Sever from parent and assign to daughter
        links[parentBase + i] = EMPTY;
        links[daughterBase + neighborCount] = neighbor;
        neighbors[neighborCount++] = neighbor;
    }

    // Step 2: Add parent-daughter link (daughter -> parent)
    for (int i = 0; i < 6; i++) {
        if (links[daughterBase + i] == EMPTY) {
            links[daughterBase + i] = parentId;
            break;
        }
    }

    // Step 3: Add parent-daughter link (parent -> daughter)
    for (int i = 0; i < 6; i++) {
        if (links[parentBase + i] == EMPTY) {
            links[parentBase + i] = newIndex;
            break;
        }
    }

    // Step 4: Update neighbors: replace parent link with daughter link
    for (int i = 0; i < neighborCount; i++) {
        uint neighbor = neighbors[i];
        int neighborBase = cells[neighbor].linkStartIndex;

        for (int j = 0; j < 6; j++) {
            if (links[neighborBase + j] == parentId) {
                links[neighborBase + j] = newIndex;
                break;
            }
        }
    }

    // Note: linkCount is NOT updated here - it will be recomputed in a separate pass
}
