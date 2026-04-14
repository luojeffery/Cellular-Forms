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
    uint numActiveCells;      // Monotonic high-water mark (next unallocated slot). Never decremented.
    uint divisionQueueCount;
    uint trueActiveCount;     // Actual number of active cells this frame (written by food_enqueue)
};
layout(std430, binding = 7) buffer DivisionQueue {
    uint divisionQueue[];
};
layout(std430, binding = 9) buffer DivisionClaims {
    coherent uint divideClaim[];
};

uniform int numCells;
uniform float linkRestLength;
uniform uint frameCounter;

const uint EMPTY = 0xFFFFFFFFu;
const int MAX_NEIGHBORS = 8;
const float PI = 3.14159265;
const float TWO_PI = 6.28318530;

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
    return float(hash(seed) & 0xFFFFu) / float(0xFFFFu);
}

vec3 safeNormalize(vec3 v, vec3 fallback) {
    float len2 = dot(v, v);
    if (len2 > 1e-8) {
        return v * inversesqrt(len2);
    }
    return fallback;
}

// Count non-EMPTY link slots for a cell.
int countLinks(int base) {
    int c = 0;
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        if (links[base + i] != EMPTY) c++;
    }
    return c;
}

// Check if a link to target already exists.
bool hasLink(int base, uint target) {
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        if (links[base + i] == target) return true;
    }
    return false;
}

// Atomic remove: scan for target and atomicCompSwap it to EMPTY.
// Returns true if successfully removed.
bool atomicRemoveLink(int base, uint target) {
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        if (atomicCompSwap(links[base + i], target, EMPTY) == target) {
            return true;
        }
    }
    return false;
}

// Atomic add: find an EMPTY slot and atomicCompSwap it to target.
// Returns true if successfully added.
bool atomicAddLink(int base, uint target) {
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        if (atomicCompSwap(links[base + i], EMPTY, target) == EMPTY) {
            return true;
        }
    }
    return false;
}

// Wrap angle difference to [-pi, pi].
float wrapAngle(float a) {
    if (a > PI) a -= TWO_PI;
    if (a < -PI) a += TWO_PI;
    return a;
}

// ---------------------------------------------------------------
// Neighborhood claim system.
//
// Before mutating ANY link, a dividing thread must claim exclusive
// access to the parent cell AND every one of its neighbors.  Claims
// are acquired in ascending cell-ID order (canonical ordering) to
// prevent two adjacent divisions from deadlocking / mutual-failing.
//
// Token = queueIndex + 1 (so 0 stays "unclaimed").
// On any claim failure the thread releases everything and returns
// false — the division simply retries next frame.
// ---------------------------------------------------------------
const uint MAX_CLAIM_SLOTS = 9u; // parent + up to 8 neighbors

bool claimNeighborhood(uint parentId, int parentBase, uint token,
                       out uint claimed[MAX_CLAIM_SLOTS], out uint claimedCount)
{
    // Gather IDs to claim: parent + all active neighbors.
    uint toSort[MAX_CLAIM_SLOTS];
    uint sortCount = 0u;
    toSort[sortCount++] = parentId;
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        uint n = links[parentBase + i];
        if (n != EMPTY && n < cells.length() && cells[n].isActive != 0) {
            toSort[sortCount++] = n;
        }
    }

    // Insertion sort by cell ID (ascending) — at most 7 elements.
    for (uint i = 1u; i < sortCount; ++i) {
        uint key = toSort[i];
        int j = int(i) - 1;
        while (j >= 0 && toSort[j] > key) {
            toSort[j + 1] = toSort[j];
            j--;
        }
        toSort[j + 1] = key;
    }

    // Attempt to claim each cell in sorted order.
    claimedCount = 0u;
    for (uint i = 0u; i < sortCount; ++i) {
        uint prev = atomicCompSwap(divideClaim[toSort[i]], 0u, token);
        if (prev != 0u && prev != token) {
            // Conflict — release everything we claimed so far and fail.
            for (uint r = 0u; r < claimedCount; ++r) {
                atomicCompSwap(divideClaim[claimed[r]], token, 0u);
            }
            claimedCount = 0u;
            return false;
        }
        // Either we just claimed it (prev==0) or we already own it (prev==token; shouldn't happen with sorted unique IDs, but safe).
        claimed[claimedCount++] = toSort[i];
    }

    memoryBarrierBuffer();
    return true;
}

void releaseNeighborhood(uint claimed[MAX_CLAIM_SLOTS], uint claimedCount, uint token) {
    memoryBarrierBuffer();
    for (uint i = 0u; i < claimedCount; ++i) {
        atomicCompSwap(divideClaim[claimed[i]], token, 0u);
    }
}

void main() {
    uint queueIndex = gl_GlobalInvocationID.x;
    if (queueIndex >= divisionQueueCount) return;

    uint parentId = divisionQueue[queueIndex];
    if (parentId >= cells.length() || cells[parentId].isActive == 0) return;

    int parentBase = cells[parentId].linkStartIndex;
    vec3 parentPos = cells[parentId].position;

    // ---------------------------------------------------------------
    // Step 0: Claim exclusive access to parent + all its neighbors.
    //         If any cell is already claimed by another dividing
    //         thread, bail out — this division retries next frame.
    // ---------------------------------------------------------------
    uint token = queueIndex + 1u; // non-zero unique token per thread
    uint claimed[MAX_CLAIM_SLOTS];
    uint claimedCount = 0u;
    if (!claimNeighborhood(parentId, parentBase, token, claimed, claimedCount)) {
        return; // neighborhood busy — skip, will retry next frame
    }

    // ---------------------------------------------------------------
    // Step 1: Gather active neighbors and estimate surface normal.
    //         ALL READ-ONLY — no allocation yet.
    // ---------------------------------------------------------------
    uint neighborIDs[MAX_NEIGHBORS];
    vec3 neighborRel[MAX_NEIGHBORS];
    int neighborCount = 0;

    for (int i = 0; i < MAX_NEIGHBORS; i++) {
        uint n = links[parentBase + i];
        if (n == EMPTY || n >= cells.length() || cells[n].isActive == 0) continue;
        neighborIDs[neighborCount] = n;
        neighborRel[neighborCount] = cells[n].position - parentPos;
        neighborCount++;
    }

    if (neighborCount < 3) {
        releaseNeighborhood(claimed, claimedCount, token);
        return;
    }

    // Estimate outward surface normal via pairwise cross products.
    vec3 normalSum = vec3(0.0);
    for (int i = 0; i < neighborCount; ++i) {
        for (int j = i + 1; j < neighborCount; ++j) {
            vec3 cp = cross(neighborRel[i], neighborRel[j]);
            float cpLen2 = dot(cp, cp);
            if (cpLen2 > 1e-10) {
                normalSum += cp * inversesqrt(cpLen2);
            }
        }
    }

    // Local outward hint from neighbor centroid (works for folded/branching forms).
    vec3 neighborCentroid = vec3(0.0);
    for (int i = 0; i < neighborCount; ++i) {
        neighborCentroid += neighborRel[i];
    }
    neighborCentroid /= float(neighborCount);
    vec3 localOutwardHint = safeNormalize(-neighborCentroid, vec3(0.0, 1.0, 0.0));

    vec3 outward;
    if (dot(normalSum, normalSum) > 1e-8) {
        if (dot(normalSum, localOutwardHint) < 0.0) normalSum = -normalSum;
        outward = safeNormalize(normalSum, localOutwardHint);
    } else {
        outward = localOutwardHint;
    }

    // ---------------------------------------------------------------
    // Step 2: Build tangent frame and sort neighbors by angle.
    // ---------------------------------------------------------------
    vec3 arbitrary = abs(outward.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(outward, arbitrary));
    vec3 B = cross(outward, T);

    float angles[MAX_NEIGHBORS];
    for (int i = 0; i < neighborCount; ++i) {
        float u = dot(neighborRel[i], T);
        float v = dot(neighborRel[i], B);
        angles[i] = atan(v, u);
    }

    // Sort neighbors by angle (insertion sort; n <= 8).
    int order[MAX_NEIGHBORS];
    for (int i = 0; i < neighborCount; ++i) order[i] = i;
    for (int i = 1; i < neighborCount; ++i) {
        int key = order[i];
        float keyAngle = angles[key];
        int j = i - 1;
        while (j >= 0 && angles[order[j]] > keyAngle) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    // ---------------------------------------------------------------
    // Step 3: Choose cleavage plane via true angular opposition.
    // ---------------------------------------------------------------
    float cleavageAngle = (randFloat(parentId * 7u + 3u + frameCounter * 131u) * 2.0 - 1.0) * PI;

    // Find the neighbor closest to cleavageAngle.
    int cleavageRingIdx0 = 0;
    float bestAbsDiff0 = 1e30;
    for (int i = 0; i < neighborCount; ++i) {
        float diff = abs(wrapAngle(angles[order[i]] - cleavageAngle));
        if (diff < bestAbsDiff0) {
            bestAbsDiff0 = diff;
            cleavageRingIdx0 = i;
        }
    }

    // Find the neighbor closest to the opposite angle.
    float oppositeAngle = wrapAngle(cleavageAngle + PI);
    int cleavageRingIdx1 = (cleavageRingIdx0 + 1) % neighborCount;
    float bestAbsDiff1 = 1e30;
    for (int i = 0; i < neighborCount; ++i) {
        if (i == cleavageRingIdx0) continue;
        float diff = abs(wrapAngle(angles[order[i]] - oppositeAngle));
        if (diff < bestAbsDiff1) {
            bestAbsDiff1 = diff;
            cleavageRingIdx1 = i;
        }
    }

    // ---------------------------------------------------------------
    // Step 3b: Choose the SHORTER arc for transfer to daughter.
    // ---------------------------------------------------------------
    int arcA = 0;
    {
        int idx = (cleavageRingIdx0 + 1) % neighborCount;
        while (idx != cleavageRingIdx1) {
            arcA++;
            idx = (idx + 1) % neighborCount;
        }
    }
    int arcB = neighborCount - 2 - arcA;

    if (arcB < arcA) {
        int tmp = cleavageRingIdx0;
        cleavageRingIdx0 = cleavageRingIdx1;
        cleavageRingIdx1 = tmp;
    }

    int transferCount = min(arcA, arcB);

    uint cleavageNeighbor0 = neighborIDs[order[cleavageRingIdx0]];
    uint cleavageNeighbor1 = neighborIDs[order[cleavageRingIdx1]];

    // ---------------------------------------------------------------
    // Step 4: Full capacity check — ALL read-only, before allocation.
    // ---------------------------------------------------------------
    int daughterNeeded = transferCount + 3;
    if (daughterNeeded > MAX_NEIGHBORS) {
        releaseNeighborhood(claimed, claimedCount, token);
        return;
    }

    int parentFinal = neighborCount - transferCount + 1;
    if (parentFinal > MAX_NEIGHBORS) {
        releaseNeighborhood(claimed, claimedCount, token);
        return;
    }

    int cn0Base = cells[cleavageNeighbor0].linkStartIndex;
    int cn1Base = cells[cleavageNeighbor1].linkStartIndex;
    if (countLinks(cn0Base) >= MAX_NEIGHBORS || countLinks(cn1Base) >= MAX_NEIGHBORS) {
        releaseNeighborhood(claimed, claimedCount, token);
        return;
    }

    // ---------------------------------------------------------------
    // Step 5: Allocate daughter cell via CAS bump.
    //
    // numActiveCells is a monotonic high-water mark — never decremented.
    // On failure the reserved slot is simply left inactive (leaked).
    // This avoids the corruption that would occur if we decremented
    // while a higher-index slot was already allocated by another thread.
    // ---------------------------------------------------------------
    uint daughterId;
    {
        uint old = numActiveCells;
        bool allocated = false;
        for (int attempt = 0; attempt < 8; ++attempt) {
            if (old >= uint(numCells)) break; // full
            uint swapped = atomicCompSwap(numActiveCells, old, old + 1u);
            if (swapped == old) {
                daughterId = old;
                allocated = true;
                break;
            }
            old = swapped; // retry with updated value
        }
        if (!allocated) {
            releaseNeighborhood(claimed, claimedCount, token);
            return;
        }
    }

    // ---------------------------------------------------------------
    // Step 6: Initialize daughter cell.
    // ---------------------------------------------------------------
    vec3 cleavageDir = safeNormalize(
        cells[cleavageNeighbor0].position - cells[cleavageNeighbor1].position,
        T
    );
    vec3 splitDir = safeNormalize(cross(outward, cleavageDir), T);

    int daughterBase = int(daughterId * uint(MAX_NEIGHBORS));
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        links[daughterBase + i] = EMPTY;
    }

    cells[daughterId].position = parentPos + (linkRestLength * 0.5) * splitDir + (linkRestLength * 0.25) * outward;
    cells[daughterId].isActive = 0;  // Stay inactive until division fully succeeds
    cells[daughterId].radius = 0.3;
    cells[daughterId].voxelCoord = cells[parentId].voxelCoord;
    cells[daughterId].flatVoxelIndex = cells[parentId].flatVoxelIndex;
    cells[daughterId].linkStartIndex = daughterBase;
    cells[daughterId].linkCount = 0;
    cells[daughterId].foodLevel = 0.0;

    // ---------------------------------------------------------------
    // Step 7: Transfer shorter-arc neighbors from parent to daughter.
    //
    // All link mutations use atomicCompSwap. We track every successful
    // transfer so that on ANY failure we can fully reverse earlier
    // transfers, restoring the original topology. Without this rollback,
    // a mid-loop failure would permanently disconnect transferred
    // neighbors from both parent and daughter.
    // ---------------------------------------------------------------
    uint transferredNeighbors[MAX_NEIGHBORS]; // IDs of successfully transferred neighbors
    int transferredCount = 0;
    bool divisionFailed = false;
    {
        int idx = (cleavageRingIdx0 + 1) % neighborCount;
        while (idx != cleavageRingIdx1) {
            uint neighborId = neighborIDs[order[idx]];
            int neighborBase = cells[neighborId].linkStartIndex;

            // Transfer: neighbor drops parent link, gains daughter link.
            bool removedFromNeighbor = atomicRemoveLink(neighborBase, parentId);
            if (!removedFromNeighbor) {
                divisionFailed = true;
                break;
            }
            if (!atomicAddLink(neighborBase, daughterId)) {
                atomicAddLink(neighborBase, parentId);
                divisionFailed = true;
                break;
            }

            // Transfer: parent drops neighbor link, daughter gains it.
            bool removedFromParent = atomicRemoveLink(parentBase, neighborId);
            if (!removedFromParent) {
                // Undo the neighbor side we just swapped.
                atomicRemoveLink(neighborBase, daughterId);
                atomicAddLink(neighborBase, parentId);
                divisionFailed = true;
                break;
            }
            if (!atomicAddLink(daughterBase, neighborId)) {
                // Undo parent side — re-add parent→neighbor.
                atomicAddLink(parentBase, neighborId);
                // Undo neighbor side — swap daughter back to parent.
                atomicRemoveLink(neighborBase, daughterId);
                atomicAddLink(neighborBase, parentId);
                divisionFailed = true;
                break;
            }

            // This neighbor fully transferred.
            transferredNeighbors[transferredCount] = neighborId;
            transferredCount++;

            idx = (idx + 1) % neighborCount;
        }
    }

    // ---------------------------------------------------------------
    // Rollback helper: reverse ALL successful transfers so far.
    // For each transferred neighbor: undo daughter↔neighbor, restore parent↔neighbor.
    // ---------------------------------------------------------------
    if (divisionFailed) {
        for (int r = 0; r < transferredCount; ++r) {
            uint nid = transferredNeighbors[r];
            int nBase = cells[nid].linkStartIndex;
            // Undo neighbor→daughter, restore neighbor→parent.
            atomicRemoveLink(nBase, daughterId);
            atomicAddLink(nBase, parentId);
            // Undo daughter→neighbor, restore parent→neighbor.
            atomicRemoveLink(daughterBase, nid);
            atomicAddLink(parentBase, nid);
        }
        cells[daughterId].isActive = 0;
        releaseNeighborhood(claimed, claimedCount, token);
        return;
    }

    // ---------------------------------------------------------------
    // Step 8: Dual-link cleavage neighbors to BOTH parent AND daughter.
    // ---------------------------------------------------------------
    bool cn0ok = atomicAddLink(cn0Base, daughterId);
    bool d_cn0ok = atomicAddLink(daughterBase, cleavageNeighbor0);

    bool cn1ok = atomicAddLink(cn1Base, daughterId);
    bool d_cn1ok = atomicAddLink(daughterBase, cleavageNeighbor1);

    // ---------------------------------------------------------------
    // Step 9: Create parent ↔ daughter bidirectional link.
    // ---------------------------------------------------------------
    bool p_dok = atomicAddLink(parentBase, daughterId);
    bool d_pok = atomicAddLink(daughterBase, parentId);

    // If any critical link failed, roll back ALL transfers + step 8/9
    // partial links, then deactivate daughter.
    if (!cn0ok || !d_cn0ok || !cn1ok || !d_cn1ok || !p_dok || !d_pok) {
        // Undo step 8/9 partial links (remove only what was added).
        if (cn0ok)   atomicRemoveLink(cn0Base, daughterId);
        if (d_cn0ok) atomicRemoveLink(daughterBase, cleavageNeighbor0);
        if (cn1ok)   atomicRemoveLink(cn1Base, daughterId);
        if (d_cn1ok) atomicRemoveLink(daughterBase, cleavageNeighbor1);
        if (p_dok)   atomicRemoveLink(parentBase, daughterId);
        if (d_pok)   atomicRemoveLink(daughterBase, parentId);

        // Undo ALL step 7 transfers.
        for (int r = 0; r < transferredCount; ++r) {
            uint nid = transferredNeighbors[r];
            int nBase = cells[nid].linkStartIndex;
            atomicRemoveLink(nBase, daughterId);
            atomicAddLink(nBase, parentId);
            atomicRemoveLink(daughterBase, nid);
            atomicAddLink(parentBase, nid);
        }
        cells[daughterId].isActive = 0;
        releaseNeighborhood(claimed, claimedCount, token);
        return;
    }

    // Division fully succeeded — activate daughter and reset parent food.
    cells[daughterId].isActive = 1;
    cells[parentId].foodLevel = 0.0;

    // Release neighborhood claims now that all mutations are complete.
    releaseNeighborhood(claimed, claimedCount, token);

    // sanitize_links + recompute_link_count run immediately after this shader.
}
