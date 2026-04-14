#version 460
layout(local_size_x = 256) in;

// ---------------------------------------------------------------------------
// Topology-aware link healing.
//
// Instead of scanning ALL active cells by distance (which bridges across
// surface folds), this shader only forms a new link between two cells when:
//
//   1. They share at least one common linked neighbor (same surface patch).
//   2. They are adjacent in that shared neighbor's cyclic 1-ring
//      (i.e., the new link fills a triangulation gap, not a random shortcut).
//   3. Neither cell is already at MAX_NEIGHBORS.
//   4. They are not already linked to each other.
//
// This preserves the 2D manifold topology of the surface.
// ---------------------------------------------------------------------------

struct Cell {
    vec3 position;       // 16 bytes (vec3 + padding)
    float foodLevel;     // 4
    vec3 voxelCoord;     // 16 bytes
    float radius;        // 4
    int linkStartIndex;  // 4
    int linkCount;       // 4
    int flatVoxelIndex;  // 4
    int isActive;        // 4
};

layout(std430, binding = 0) buffer CellBuffer {
    Cell cells[];
};
layout(std430, binding = 1) buffer LinkBuffer {
    uint links[];
};
layout(std430, binding = 5) buffer GlobalCounts {
    uint numActiveCells;      // Monotonic high-water mark (next unallocated slot)
    uint divisionQueueCount;
    uint trueActiveCount;     // Actual number of active cells this frame
};

const uint EMPTY = 0xFFFFFFFFu;
const int MAX_NEIGHBORS = 8;

uniform int targetNeighbors;

// ---------- helpers --------------------------------------------------------

bool hasLink(int base, uint target) {
    for (int j = 0; j < MAX_NEIGHBORS; ++j) {
        if (links[base + j] == target) return true;
    }
    return false;
}

int findEmptySlot(int base) {
    for (int j = 0; j < MAX_NEIGHBORS; ++j) {
        if (links[base + j] == EMPTY) return base + j;
    }
    return -1;
}

int countLinks(int base) {
    int c = 0;
    for (int j = 0; j < MAX_NEIGHBORS; ++j) {
        if (links[base + j] != EMPTY) c++;
    }
    return c;
}

vec3 safeNormalize(vec3 v, vec3 fallback) {
    float len2 = dot(v, v);
    if (len2 > 1e-8) return v * inversesqrt(len2);
    return fallback;
}

// Build sorted 1-ring for a cell. Returns the count (up to MAX_NEIGHBORS).
// ring[] is filled with neighbor IDs in cyclic angular order.
int buildSortedRing(uint cellId, out uint ring[MAX_NEIGHBORS]) {
    vec3 P = cells[cellId].position;
    int base = cells[cellId].linkStartIndex;

    // Gather active neighbors.
    uint nbrs[MAX_NEIGHBORS];
    vec3 rels[MAX_NEIGHBORS];
    int n = 0;
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        uint nid = links[base + i];
        if (nid == EMPTY || nid >= cells.length() || cells[nid].isActive == 0) continue;
        nbrs[n] = nid;
        rels[n] = cells[nid].position - P;
        n++;
    }

    if (n < 2) {
        for (int i = 0; i < n; ++i) ring[i] = nbrs[i];
        return n;
    }

    // Estimate normal using local neighbor geometry (not global position).
    // Use (P - neighborCentroid) as outward hint — works even for folded forms.
    vec3 neighborCentroid = vec3(0.0);
    for (int i = 0; i < n; ++i) neighborCentroid += rels[i];
    neighborCentroid /= float(n);
    vec3 localOutwardHint = safeNormalize(-neighborCentroid, vec3(0.0, 1.0, 0.0));

    vec3 normalSum = vec3(0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            vec3 cp = cross(rels[i], rels[j]);
            float cpLen2 = dot(cp, cp);
            if (cpLen2 > 1e-10) normalSum += cp * inversesqrt(cpLen2);
        }
    }
    vec3 N;
    if (dot(normalSum, normalSum) > 1e-8) {
        if (dot(normalSum, localOutwardHint) < 0.0) normalSum = -normalSum;
        N = safeNormalize(normalSum, localOutwardHint);
    } else {
        N = localOutwardHint;
    }

    // Build tangent frame.
    vec3 arb = abs(N.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(N, arb));
    vec3 B = cross(N, T);

    // Compute angles and insertion-sort.
    float ang[MAX_NEIGHBORS];
    int order[MAX_NEIGHBORS];
    for (int i = 0; i < n; ++i) {
        ang[i] = atan(dot(rels[i], B), dot(rels[i], T));
        order[i] = i;
    }
    for (int i = 1; i < n; ++i) {
        int key = order[i];
        float keyAng = ang[key];
        int j = i - 1;
        while (j >= 0 && ang[order[j]] > keyAng) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    for (int i = 0; i < n; ++i) ring[i] = nbrs[order[i]];
    return n;
}

// ---------- main -----------------------------------------------------------

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cells.length() || cells[id].isActive == 0) return;

    int selfBase = cells[id].linkStartIndex;

    // Only heal cells that have room and are below target.
    if (countLinks(selfBase) >= targetNeighbors) return;
    if (findEmptySlot(selfBase) == -1) return;

    vec3 selfPos = cells[id].position;

    // For each of our linked neighbors, check that neighbor's 1-ring for
    // cells that are adjacent to us in the ring but not yet linked to us.
    // Those are the only safe candidates for a new link.

    uint bestCandidate = EMPTY;
    float bestDist2 = 1e30;

    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        uint mutualId = links[selfBase + i];
        if (mutualId == EMPTY || mutualId >= cells.length() || cells[mutualId].isActive == 0) continue;

        // Build the sorted 1-ring of this mutual neighbor.
        uint ring[MAX_NEIGHBORS];
        int ringCount = buildSortedRing(mutualId, ring);
        if (ringCount < 2) continue;

        // Find where 'id' (self) sits in the mutual neighbor's ring.
        int selfRingIdx = -1;
        for (int r = 0; r < ringCount; ++r) {
            if (ring[r] == id) { selfRingIdx = r; break; }
        }
        if (selfRingIdx == -1) continue; // shouldn't happen if links are bidirectional

        // The two cells adjacent to 'id' in this ring are candidates.
        // They share the mutual neighbor with us AND are ring-adjacent,
        // so a link to them fills a triangulation gap.
        uint prevInRing = ring[(selfRingIdx - 1 + ringCount) % ringCount];
        uint nextInRing = ring[(selfRingIdx + 1) % ringCount];

        uint candidates[2];
        candidates[0] = prevInRing;
        candidates[1] = nextInRing;

        for (int c = 0; c < 2; ++c) {
            uint candidateId = candidates[c];
            if (candidateId == id) continue;
            if (candidateId >= cells.length() || cells[candidateId].isActive == 0) continue;

            // Already linked?
            if (hasLink(selfBase, candidateId)) continue;

            // Candidate must have room.
            int candidateBase = cells[candidateId].linkStartIndex;
            if (countLinks(candidateBase) >= targetNeighbors) continue;
            if (findEmptySlot(candidateBase) == -1) continue;

            // Pick the closest valid candidate.
            vec3 delta = cells[candidateId].position - selfPos;
            float d2 = dot(delta, delta);
            if (d2 < bestDist2) {
                bestDist2 = d2;
                bestCandidate = candidateId;
            }
        }
    }

    if (bestCandidate == EMPTY) return;

    // Double-check before atomic writes.
    int candidateBase = cells[bestCandidate].linkStartIndex;
    if (hasLink(selfBase, bestCandidate) || hasLink(candidateBase, id)) return;

    int selfSlot = findEmptySlot(selfBase);
    int otherSlot = findEmptySlot(candidateBase);
    if (selfSlot == -1 || otherSlot == -1) return;

    // Atomic claim to avoid races with other threads.
    if (atomicCompSwap(links[selfSlot], EMPTY, bestCandidate) != EMPTY) return;
    if (atomicCompSwap(links[otherSlot], EMPTY, id) != EMPTY) {
        links[selfSlot] = EMPTY; // rollback
    }
}
