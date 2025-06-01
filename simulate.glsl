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
layout(std430, binding = 2) buffer CellCountPerVoxel {
    uint cellCountPerVoxel[];
};
layout(std430, binding = 3) buffer StartIndexPerVoxel {
    uint startIndexPerVoxel[];
};
layout(std430, binding = 4) buffer FlatVoxelCellIDs {
    uint flatVoxelCellIDs[];
};

layout(std430, binding = 5) buffer GlobalCounts {
    uint numActiveCells;
};

uniform float linkRestLength;
uniform float springFactor;
uniform float planarFactor;
uniform float bulgeFactor;
uniform float repulsionFactor;
uniform float repulsionRadius;
uniform vec3 gridResolution;
uniform float timeStep;
uniform float voxelSize;
uniform int numCells;
uniform int foodThreshold;

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


void main() {
    uint id = gl_GlobalInvocationID.x;


    if (id >= cells.length() || cells[id].isActive == 0) return;
    if (cells[id].linkCount == 0) {
        for (int i = 0; i < 6; ++i) {
            links[cells[id].linkStartIndex + i] = EMPTY;
        }
    }
    float r = randFloat(id + uint(gl_WorkGroupID.x) * 1234u);
    float increment = 1.0 + 9.0 * r;

    if (numActiveCells < numCells) {
        cells[id].foodLevel += increment;
    }

    uint newIndex;
    // cell division
    if (cells[id].foodLevel >= foodThreshold && numActiveCells < numCells) {
        // when a cell divides, set the daughter cell at the same position as where it is right now.
        // the daughter cell will form links to the same neighbors as the parent cell.
        // the parent cell will continue to move as a result of calculations from the previously linked cells.
        // we will delete the links after moving this cell. we delete the first three links (just set to EMPTY)
        newIndex = atomicAdd(numActiveCells, 1);
        if (newIndex >= numCells) {
            // If we exceed the number of cells, we stop processing this cell
            atomicAdd(numActiveCells, -1);
            cells[id].foodLevel = 0;
        }
        else {
            cells[newIndex].isActive = 1;
            float rx = randFloat(id * 3u + 0u) * 2.0 - 1.0;
            float ry = randFloat(id * 3u + 1u) * 2.0 - 1.0;
            float rz = randFloat(id * 3u + 2u) * 2.0 - 1.0;

            vec3 randomOffset = 0.1 * normalize(vec3(rx, ry, rz));
            cells[newIndex].position = cells[id].position + randomOffset;
            cells[newIndex].radius = 0.1;
            cells[newIndex].voxelCoord = cells[id].voxelCoord;
            cells[newIndex].flatVoxelIndex = cells[id].flatVoxelIndex;
            cells[newIndex].linkStartIndex = int(newIndex * 6);
            cells[newIndex].linkCount = 0;
            for (int i = 0; i < 6; ++i) {
                links[cells[newIndex].linkStartIndex + i] = EMPTY;
            }
        }
    }

    vec3 P = cells[id].position;
    int n = 6;

    // --- DIRECT FORCES (spring, planar, bulge) ---
    vec3 springTarget = vec3(0.0);
    vec3 planarTarget = vec3(0.0);
    vec3 normal = vec3(0.0);
    float bulgeDist = 0.0;


    for (int i = 0; i < n; ++i) {
        if (i == -1)
            continue;
        uint neighborIndex = links[cells[id].linkStartIndex + i];
        vec3 L = cells[neighborIndex].position;

        planarTarget += L;
        vec3 dir = normalize(P - L);
        springTarget += L + linkRestLength * dir;
    }

    if (cells[id].linkCount > 0) {
        planarTarget /= float(cells[id].linkCount);
        springTarget /= float(cells[id].linkCount);
    }

    // Approximate normal
    if (cells[id].linkCount >= 2) {
        vec3 A = cells[links[cells[id].linkStartIndex]].position;
        vec3 B = cells[links[cells[id].linkStartIndex + 1]].position;
        normal = normalize(cross(normalize(A - P), normalize(B - P)));
    } else {
        normal = vec3(0.0, 1.0, 0.0); // fallback
    }

    for (int i = 0; i < n; ++i) {
        if (i == -1)
            continue;
        uint neighborIndex = links[cells[id].linkStartIndex + i];
        vec3 L = cells[neighborIndex].position;

        float dotNr = dot(L - P, normal);
        float LLen = length(L - P);
        bulgeDist += sqrt(max(0.0, linkRestLength * linkRestLength - LLen * LLen + dotNr * dotNr)) + dotNr;
    }

    if (cells[id].linkCount > 0) {
        bulgeDist /= float(cells[id].linkCount);
    }

    vec3 bulgeTarget = P + bulgeDist * normal;

    vec3 totalTarget;
    if (repulsionFactor != 0) {
        uint startIndex = startIndexPerVoxel[cells[id].flatVoxelIndex];
        uint count = cellCountPerVoxel[cells[id].flatVoxelIndex];
        vec3 collisionOffset = vec3(0.0);
        uint indirectCells[50];
        uint indirectCount = 0;

        for (uint i = startIndex; i < startIndex + count; i++) {
            uint candidate = flatVoxelCellIDs[i];
            if (candidate == id) continue;

            bool isDirect = false;
            for (uint j = 0; j < cells[id].linkCount; j++) {
                if (links[cells[id].linkStartIndex + j] == candidate) {
                    isDirect = true;
                    break;
                }
            }

            if (!isDirect) {
                indirectCells[indirectCount++] = candidate;
            }
        }
        for (uint i = 0; i < indirectCells.length(); i++) {
            vec3 diff = cells[id].position - cells[indirectCells[i]].position;
            float dist2 = dot(diff, diff);
            float roiSquared = repulsionRadius * repulsionRadius;
            if (dist2 < roiSquared && dist2 > 0.5)
                collisionOffset += repulsionFactor * ((roiSquared - dist2) / roiSquared) * normalize(diff);
        }

        totalTarget = P
        + springFactor * (springTarget - P)
        + planarFactor * (planarTarget - P)
        + bulgeFactor * (bulgeTarget - P)
        + collisionOffset;
    }
    else {
        totalTarget = P
        + springFactor * (springTarget - P)
        + planarFactor * (planarTarget - P)
        + bulgeFactor * (bulgeTarget - P);
    }

    cells[id].position = mix(P, totalTarget, timeStep);

    // after moving recalculate voxel coord
    cells[id].voxelCoord = floor(cells[id].position / voxelSize);
    ivec3 voxel_offset = ivec3(gridResolution) / 2;
    ivec3 shiftedVoxel = ivec3(cells[id].voxelCoord) + voxel_offset;
    cells[id].flatVoxelIndex = int(shiftedVoxel.x +
                              shiftedVoxel.y * gridResolution.x +
                              shiftedVoxel.z * gridResolution.x * gridResolution.y);

        // post cell division, update links
    if (cells[id].foodLevel >= foodThreshold && numActiveCells < numCells) {
        if (newIndex >= numCells) {
            atomicAdd(numActiveCells, -1); // Undo if out of bounds
        } else {
            cells[id].foodLevel = 0;

            // Step 1: Sever half of parent's links and assign to daughter
            int linksToSever = cells[id].linkCount / 2;
            uint neighbors[3];
            int neighborCount = 0;

            int parentBase = cells[id].linkStartIndex;
            int daughterBase = cells[newIndex].linkStartIndex;

            for (int i = 0; i < 6; i++) {
                uint neighbor = links[parentBase + i];
                if (neighbor == EMPTY || neighborCount >= linksToSever) {
                    atomicExchange(links[daughterBase + i], EMPTY);
                    continue;
                }
                atomicExchange(links[parentBase + i], EMPTY);
                // Write neighbor to daughter's link list atomically
                atomicExchange(links[daughterBase + i], neighbor);

                neighbors[neighborCount++] = neighbor;
                atomicAdd(cells[newIndex].linkCount, 1);
                atomicAdd(cells[id].linkCount, -1);
            }

            // Step 2: Daughter -> parent
            for (int i = 0; i < 6; i++) {
                if (atomicCompSwap(links[daughterBase + i], EMPTY, id) == EMPTY) {
                    atomicAdd(cells[newIndex].linkCount, 1);
                    break;
                }
            }

            // Step 3: Parent -> daughter
            for (int i = 0; i < 6; i++) {
                if (atomicCompSwap(links[parentBase + i], EMPTY, newIndex) == EMPTY) {
                    atomicAdd(cells[id].linkCount, 1);
                    break;
                }
            }

            // Step 4: Neighbor -> daughter (replace parent link with daughter)
            for (int i = 0; i < neighborCount; i++) {
                uint neighbor = neighbors[i];
                int neighborBase = cells[neighbor].linkStartIndex;

                for (int j = 0; j < 6; j++) {
                    if (atomicCompSwap(links[neighborBase + j], id, newIndex) == id) {
                        break;
                    }
                }
            }

        }
    }

    //todo: you need to heal unused links
    int voxID = cells[id].flatVoxelIndex;
    uint startID = startIndexPerVoxel[voxID];
    int selfLinkStart = cells[id].linkStartIndex;
    for (uint idx = startID; idx < startID + int(cellCountPerVoxel[voxID]); idx++) {
        uint otherID = flatVoxelCellIDs[idx];
        if (otherID == id) continue;
        if (cells[otherID].isActive == 0) continue; // Skip inactive cells
        int otherLinkStart = cells[otherID].linkStartIndex;

        bool alreadyLinked = false;
        for (int j = 0; j < 6; ++j) {
            if (links[selfLinkStart + j] == int(otherID)) {
                alreadyLinked = true;
                break;
            }
        }

        if (alreadyLinked) continue;

        // Find empty spot in self
        int selfEmpty = -1;
        for (int j = 0; j < 6; ++j) {
            if (links[selfLinkStart + j] == EMPTY) {
                selfEmpty = selfLinkStart + j;
                break;
            }
        }

        // Find empty spot in other
        int otherEmpty = -1;
        for (int j = 0; j < 6; ++j) {
            if (links[otherLinkStart + j] == EMPTY) {
                otherEmpty = otherLinkStart + j;
                break;
            }
        }

        // Only link if both have space
        if (selfEmpty != -1 && otherEmpty != -1) {
            atomicExchange(links[selfEmpty], int(otherID));
            atomicExchange(links[otherEmpty], int(id));
            atomicAdd(cells[id].linkCount, 1);
            atomicAdd(cells[otherID].linkCount, 1);
        }
    }
}

