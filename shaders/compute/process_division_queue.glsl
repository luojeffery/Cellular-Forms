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
uniform float linkRestLength;

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

// The link rest length must match the value passed to the simulate shader (linkRestLength uniform).

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

    int parentBase = cells[parentId].linkStartIndex;
    vec3 parentPos = cells[parentId].position;
    vec3 radialFallback = safeNormalize(parentPos, vec3(0.0, 1.0, 0.0));

    // Gather active neighbors and their relative positions for normal estimation.
    vec3 rel[6];
    int relCount = 0;
    vec3 neighborPosSum = vec3(0.0);
    for (int i = 0; i < 6; i++) {
        uint neighbor = links[parentBase + i];
        if (neighbor == EMPTY || neighbor >= cells.length() || cells[neighbor].isActive == 0) continue;
        rel[relCount] = cells[neighbor].position - parentPos;
        neighborPosSum += cells[neighbor].position;
        relCount++;
    }

    // Estimate the outward surface normal using cross products of neighbor-pair vectors.
    // This is more robust than the simple sum-of-directions approach.
    vec3 outward;
    if (relCount >= 2) {
        vec3 normalSum = vec3(0.0);
        for (int i = 0; i < relCount; ++i) {
            for (int j = i + 1; j < relCount; ++j) {
                vec3 cp = cross(rel[i], rel[j]);
                float cpLen2 = dot(cp, cp);
                if (cpLen2 > 1e-10) {
                    normalSum += cp * inversesqrt(cpLen2);
                }
            }
        }
        if (dot(normalSum, normalSum) > 1e-8) {
            // Orient consistent with the radial direction (outward from origin).
            if (dot(normalSum, radialFallback) < 0.0) normalSum = -normalSum;
            outward = safeNormalize(normalSum, radialFallback);
        } else {
            outward = safeNormalize(parentPos - neighborPosSum / float(relCount), radialFallback);
        }
    } else if (relCount > 0) {
        outward = safeNormalize(parentPos - neighborPosSum / float(relCount), radialFallback);
    } else {
        outward = radialFallback;
    }

    // Choose a random TANGENTIAL split direction (perpendicular to the surface normal).
    // Splitting tangentially keeps both daughter and parent on the surface shell.
    float rx = randFloat(parentId * 7u + 0u) * 2.0 - 1.0;
    float ry = randFloat(parentId * 7u + 1u) * 2.0 - 1.0;
    float rz = randFloat(parentId * 7u + 2u) * 2.0 - 1.0;
    vec3 randomVec = safeNormalize(vec3(rx, ry, rz), radialFallback);
    // Project onto the tangent plane of the surface and renormalize.
    vec3 splitDir = randomVec - dot(randomVec, outward) * outward;
    splitDir = safeNormalize(splitDir, cross(outward, radialFallback));

    // Place the daughter half a rest-length away in the tangential split direction.
    // This gives the physics a clear, surface-aligned starting state.
    cells[newIndex].position = parentPos + (linkRestLength * 0.5) * splitDir;
    cells[newIndex].isActive = 1;
    cells[newIndex].radius = 0.3;
    cells[newIndex].voxelCoord = cells[parentId].voxelCoord;
    cells[newIndex].flatVoxelIndex = cells[parentId].flatVoxelIndex;
    cells[newIndex].linkStartIndex = int(newIndex * 6);
    cells[newIndex].linkCount = 0; // Will be recomputed later
    cells[newIndex].foodLevel = 0.0;

    // Initialize daughter's link slots to EMPTY
    int daughterBase = cells[newIndex].linkStartIndex;
    for (int i = 0; i < 6; ++i) {
        links[daughterBase + i] = EMPTY;
    }

    // Divide parent's links spatially: neighbors on the positive splitDir side transfer to
    // the daughter cell, preserving a locally planar (manifold) surface topology.
    uint transferredNeighbors[6];
    int transferredCount = 0;
    int daughterLinkCount = 0;

    for (int i = 0; i < 6; i++) {
        uint neighbor = links[parentBase + i];
        if (neighbor == EMPTY || neighbor >= cells.length() || cells[neighbor].isActive == 0) continue;

        // Determine which side of the split plane this neighbor falls on.
        vec3 toNeighbor = cells[neighbor].position - parentPos;
        float side = dot(toNeighbor, splitDir);

        // Positive-side neighbors go to daughter (reserve one slot for the parent-daughter link).
        if (side > 0.0 && daughterLinkCount < 5) {
            links[parentBase + i] = EMPTY;
            links[daughterBase + daughterLinkCount] = neighbor;
            daughterLinkCount++;
            transferredNeighbors[transferredCount++] = neighbor;
        }
        // Negative-side (and zero) neighbors stay with the parent.
    }

    // Add bidirectional parent-daughter link.
    for (int i = 0; i < 6; i++) {
        if (links[daughterBase + i] == EMPTY) { links[daughterBase + i] = parentId; break; }
    }
    for (int i = 0; i < 6; i++) {
        if (links[parentBase + i] == EMPTY) { links[parentBase + i] = newIndex; break; }
    }

    // Update transferred neighbors: replace their link to parent with a link to daughter.
    for (int i = 0; i < transferredCount; i++) {
        uint neighbor = transferredNeighbors[i];
        int neighborBase = cells[neighbor].linkStartIndex;
        for (int j = 0; j < 6; j++) {
            if (links[neighborBase + j] == parentId) {
                links[neighborBase + j] = newIndex;
                break;
            }
        }
    }

    // Note: linkCount is NOT updated here - it will be recomputed in a separate pass.
}
