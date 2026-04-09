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
    uint divisionQueueCount;
};

const uint EMPTY = 0xFFFFFFFFu;
const int MAX_NEIGHBORS = 6;
// Normals must agree within ~49.5° (cos⁻¹ 0.65) to be considered the same surface patch.
const float MIN_NORMAL_ALIGNMENT = 0.65;
// Keep radial (normal-direction) separation tight so cells on opposite faces of a thin
// shell cannot heal a link across the shell interior.
const float MAX_RADIAL_SEPARATION = 0.25;

uniform vec3 gridResolution;
uniform float healingRadius;
uniform int targetNeighbors;

vec3 safeNormalize(vec3 v, vec3 fallback) {
    float len2 = dot(v, v);
    if (len2 > 1e-8) {
        return v * inversesqrt(len2);
    }
    return fallback;
}

bool has_link(int base, uint otherID) {
    for (int j = 0; j < MAX_NEIGHBORS; ++j) {
        if (links[base + j] == otherID) {
            return true;
        }
    }
    return false;
}

int find_empty_slot(int base) {
    for (int j = 0; j < MAX_NEIGHBORS; ++j) {
        if (links[base + j] == EMPTY) {
            return base + j;
        }
    }
    return -1;
}

vec3 estimate_local_normal(uint cellID) {
    vec3 P = cells[cellID].position;
    int base = cells[cellID].linkStartIndex;
    vec3 radialFallback = safeNormalize(P, vec3(0.0, 1.0, 0.0));

    vec3 rel[MAX_NEIGHBORS];
    int relCount = 0;
    vec3 neighborPosSum = vec3(0.0);

    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        uint n = links[base + i];
        if (n == EMPTY || n >= cells.length() || cells[n].isActive == 0) {
            continue;
        }
        vec3 L = cells[n].position;
        rel[relCount++] = L - P;
        neighborPosSum += L;
    }

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
            if (dot(normalSum, radialFallback) < 0.0) {
                normalSum = -normalSum;
            }
            return safeNormalize(normalSum, radialFallback);
        }
    }

    if (relCount > 0) {
        vec3 avg = neighborPosSum / float(relCount);
        return safeNormalize(P - avg, radialFallback);
    }
    return radialFallback;
}

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;
    if (cells[id].linkCount >= targetNeighbors) return;

    int selfLinkStart = cells[id].linkStartIndex;
    if (find_empty_slot(selfLinkStart) == -1) return;

    vec3 selfPos = cells[id].position;
    vec3 selfNormal = estimate_local_normal(id);

    float healingRadius2 = healingRadius * healingRadius;
    uint bestOtherID = EMPTY;
    float bestScore = 1e30;

    uint activeCount = min(numActiveCells, uint(cells.length()));
    for (uint otherID = 0u; otherID < activeCount; ++otherID) {
        if (otherID == id || otherID >= cells.length()) continue;
        if (cells[otherID].isActive == 0) continue;
        if (cells[otherID].linkCount >= targetNeighbors) continue;
        if (find_empty_slot(cells[otherID].linkStartIndex) == -1) continue;

        vec3 otherNormal = estimate_local_normal(otherID);
        if (dot(selfNormal, otherNormal) < MIN_NORMAL_ALIGNMENT) continue;

        vec3 delta = cells[otherID].position - selfPos;
        float radialSeparation = abs(dot(delta, selfNormal));
        if (radialSeparation > MAX_RADIAL_SEPARATION) continue;

        vec3 tangentialDelta = delta - dot(delta, selfNormal) * selfNormal;
        float tangentialDist2 = dot(tangentialDelta, tangentialDelta);
        if (tangentialDist2 > healingRadius2) continue;

        int otherLinkStart = cells[otherID].linkStartIndex;
        if (has_link(selfLinkStart, otherID) || has_link(otherLinkStart, id)) continue;

        float score = tangentialDist2 + 0.25 * radialSeparation * radialSeparation;
        if (score < bestScore) {
            bestScore = score;
            bestOtherID = otherID;
        }
    }

    if (bestOtherID == EMPTY) {
        return;
    }

    int otherLinkStart = cells[bestOtherID].linkStartIndex;
    if (has_link(selfLinkStart, bestOtherID) || has_link(otherLinkStart, id)) {
        return;
    }

    int selfEmpty = find_empty_slot(selfLinkStart);
    int otherEmpty = find_empty_slot(otherLinkStart);
    if (selfEmpty == -1 || otherEmpty == -1) {
        return;
    }

    if (atomicCompSwap(links[selfEmpty], EMPTY, bestOtherID) != EMPTY) {
        return;
    }
    if (atomicCompSwap(links[otherEmpty], EMPTY, id) != EMPTY) {
        links[selfEmpty] = EMPTY;
    }
}
