#version 460
layout(local_size_x = 256) in;

struct Cell {
    vec3 position;
    float foodLevel;
    int linkStartIndex;
    int linkCount;
    float radius;
    float padding;
};

layout(std430, binding = 0) buffer CellBuffer {
    Cell cells[];
};
layout(std430, binding = 1) buffer LinkBuffer {
    int links[];
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

uniform float linkRestLength;
uniform float springFactor;
uniform float planarFactor;
uniform float bulgeFactor;
uniform float repulsionFactor;
uniform float repulsionRadius;
uniform float voxelSize;
uniform ivec3 gridResolution;
uniform float timeStep;

uvec3 getVoxelCoord(vec3 pos) {
    return uvec3(floor(pos / voxelSize));
}

uint flattenVoxelIndex(uvec3 v) {
    return v.x + v.y * gridResolution.x + v.z * gridResolution.x * gridResolution.y;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cells.length()) return;

    vec3 P = cells[id].position;
    int n = cells[id].linkCount;

    // --- DIRECT FORCES (spring, planar, bulge) ---
    vec3 springTarget = vec3(0.0);
    vec3 planarTarget = vec3(0.0);
    vec3 normal = vec3(0.0);
    float bulgeDist = 0.0;

    for (int i = 0; i < n; ++i) {
        int neighborIndex = links[cells[id].linkStartIndex + i];
        vec3 L = cells[neighborIndex].position;

        planarTarget += L;
        vec3 dir = normalize(P - L);
        springTarget += L + linkRestLength * dir;
    }

    if (n > 0) {
        planarTarget /= float(n);
        springTarget /= float(n);
    }

    // Approximate normal
    if (n >= 2) {
        vec3 A = cells[links[cells[id].linkStartIndex]].position;
        vec3 B = cells[links[cells[id].linkStartIndex + 1]].position;
        normal = normalize(cross(normalize(A - P), normalize(B - P)));
    } else {
        normal = vec3(0.0, 1.0, 0.0); // fallback
    }

    for (int i = 0; i < n; ++i) {
        int neighborIndex = links[cells[id].linkStartIndex + i];
        vec3 L = cells[neighborIndex].position;

        float dotNr = dot(L - P, normal);
        float LLen = length(L - P);
        bulgeDist += sqrt(max(0.0, linkRestLength * linkRestLength - LLen * LLen + dotNr * dotNr + dotNr));
    }

    if (n > 0) {
        bulgeDist /= float(n);
    }

    vec3 bulgeTarget = P + bulgeDist * normal;

    // --- INDIRECT REPULSION FORCES ---
    vec3 repulsion = vec3(0.0);
    uvec3 myVoxel = getVoxelCoord(P);

    for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
    for (int dz = -1; dz <= 1; ++dz) {
        uvec3 neighborVoxel = myVoxel + uvec3(dx, dy, dz);
        if (any(greaterThanEqual(neighborVoxel, uvec3(gridResolution)))) continue;

        uint voxelIdx = flattenVoxelIndex(neighborVoxel);
        uint count = cellCountPerVoxel[voxelIdx];
        uint start = startIndexPerVoxel[voxelIdx];

        for (uint i = 0; i < count; ++i) {
            uint otherID = flatVoxelCellIDs[start + i];
            if (otherID == id) continue;

            vec3 otherPos = cells[otherID].position;
            vec3 offset = P - otherPos;
            float dist = length(offset);
            if (dist < repulsionRadius && dist > 0.0001) {
                float repulsionStrength = repulsionFactor * (1.0 - dist / repulsionRadius);
                repulsion += normalize(offset) * repulsionStrength;
            }
        }
    }}}

    // --- COMBINE AND INTEGRATE ---
    vec3 totalTarget = P
        + springFactor * (springTarget - P)
        + planarFactor * (planarTarget - P)
        + bulgeFactor * (bulgeTarget - P)
        + repulsion;

    cells[id].position = mix(P, totalTarget, timeStep); // damped integration
}