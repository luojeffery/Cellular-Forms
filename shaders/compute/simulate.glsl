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
    uint numActiveCells;      // Monotonic high-water mark
    uint divisionQueueCount;
    uint trueActiveCount;     // Actual active cells (counted by food_enqueue)
};
// Position snapshot from before this dispatch — ensures all neighbor reads
// see pre-update positions (synchronous Euler step, not Gauss-Seidel).
layout(std430, binding = 8) readonly buffer PositionSnapshot {
    vec4 posSnapshot[];
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

const uint EMPTY = 0xFFFFFFFFu;
const int MAX_NEIGHBORS = 8;

vec3 safeNormalize(vec3 v, vec3 fallback) {
    float len2 = dot(v, v);
    if (len2 > 1e-8) {
        return v * inversesqrt(len2);
    }
    return fallback;
}

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    vec3 P = cells[id].position;
    int linkBase = cells[id].linkStartIndex;

    // --- DIRECT FORCES (spring, planar, bulge) ---
    vec3 springTarget = vec3(0.0);
    vec3 planarTarget = vec3(0.0);
    vec3 normal = vec3(0.0);
    float bulgeDist = 0.0;
    int validCount = 0;

    uint neighborIDs[MAX_NEIGHBORS];
    vec3 neighborRel[MAX_NEIGHBORS];

    // Local outward hint: direction from neighbor centroid to cell.
    // Works for folded/branching forms (unlike global radial direction).
    vec3 neighborCentroidHint = vec3(0.0);
    
    for (int i = 0; i < MAX_NEIGHBORS; ++i) {
        uint neighborIndex = links[linkBase + i];
        if (neighborIndex == EMPTY || neighborIndex >= cells.length() || cells[neighborIndex].isActive == 0) continue;
        vec3 L = posSnapshot[neighborIndex].xyz;  // Read from snapshot, not live buffer
        vec3 rel = L - P;

        // Planar target: average of neighbor positions (paper formula)
        planarTarget += L;
        
        // Spring target: for each link, target position is neighbor + restLength * direction from neighbor to cell
        // Paper: springTarget = 1/n * sum(Lr + linkRestLength * normalize(P - Lr))
        vec3 dirFromNeighborToCell = safeNormalize(P - L, vec3(0.0, 1.0, 0.0));
        springTarget += L + linkRestLength * dirFromNeighborToCell;

        neighborIDs[validCount] = neighborIndex;
        neighborRel[validCount] = rel;
        neighborCentroidHint += rel;
        validCount++;
    }

    // If no valid neighbors, skip all force computation — keep cell at P.
    // This prevents isolated cells from drifting toward the origin because
    // springTarget/planarTarget default to zero.
    if (validCount == 0) {
        // Still update voxel coords for the cell's current position.
        cells[id].voxelCoord = floor(P / voxelSize);
        ivec3 voxel_offset = ivec3(gridResolution) / 2;
        ivec3 shiftedVoxel = ivec3(cells[id].voxelCoord) + voxel_offset;
        if (any(lessThan(shiftedVoxel, ivec3(0))) || any(greaterThanEqual(shiftedVoxel, ivec3(gridResolution)))) {
            cells[id].flatVoxelIndex = -1;
        } else {
            cells[id].flatVoxelIndex = int(shiftedVoxel.x +
                                      shiftedVoxel.y * gridResolution.x +
                                      shiftedVoxel.z * gridResolution.x * gridResolution.y);
        }
        return;
    }

    vec3 localOutwardHint;
    {
        planarTarget /= float(validCount);
        springTarget /= float(validCount);
        neighborCentroidHint /= float(validCount);
        localOutwardHint = safeNormalize(-neighborCentroidHint, vec3(0.0, 1.0, 0.0));
    }

    // Paper-style local normal from pairs of linked neighbors around P.
    vec3 normalSum = vec3(0.0);
    int normalPairCount = 0;
    for (int i = 0; i < validCount; ++i) {
        for (int j = i + 1; j < validCount; ++j) {
            vec3 cp = cross(neighborRel[i], neighborRel[j]);
            float cpLen2 = dot(cp, cp);
            if (cpLen2 > 1e-10) {
                normalSum += cp * inversesqrt(cpLen2);
                normalPairCount++;
            }
        }
    }

    if (normalPairCount > 0) {
        if (dot(normalSum, localOutwardHint) < 0.0) {
            normalSum = -normalSum;
        }
        normal = safeNormalize(normalSum, localOutwardHint);
    } else {
        // validCount > 0 guaranteed (we returned early if 0).
        normal = safeNormalize(P - planarTarget, localOutwardHint);
    }

    // Bulge target from intersections of P's normal with rest-length spheres around neighbor pairs.
    float bulgeAccum = 0.0;
    int bulgePairCount = 0;
    for (int i = 0; i < validCount; ++i) {
        float dotNi = dot(neighborRel[i], normal);
        float discI = linkRestLength * linkRestLength - dot(neighborRel[i], neighborRel[i]) + dotNi * dotNi;
        if (discI <= 0.0) {
            continue;
        }

        for (int j = i + 1; j < validCount; ++j) {
            float dotNj = dot(neighborRel[j], normal);
            float discJ = linkRestLength * linkRestLength - dot(neighborRel[j], neighborRel[j]) + dotNj * dotNj;
            if (discJ <= 0.0) {
                continue;
            }

            float t1 = dotNi + sqrt(discI);
            float t2 = dotNj + sqrt(discJ);
            bulgeAccum += 0.5 * (t1 + t2);
            bulgePairCount++;
        }
    }

    if (bulgePairCount > 0) {
        bulgeDist = bulgeAccum / float(bulgePairCount);
    } else if (validCount > 0) {
        // Fallback when there are too few valid pairs.
        for (int i = 0; i < validCount; ++i) {
            float dotNi = dot(neighborRel[i], normal);
            float discI = linkRestLength * linkRestLength - dot(neighborRel[i], neighborRel[i]) + dotNi * dotNi;
            if (discI > 0.0) {
                bulgeDist += dotNi + sqrt(discI);
            }
        }
        bulgeDist /= float(validCount);
    }

    // Negative bulge pulls cells inward and tends to fill the interior volume.
    bulgeDist = max(0.0, bulgeDist);

    vec3 bulgeTarget = P + bulgeDist * normal;

    vec3 totalTarget;
    if (repulsionFactor != 0) {
        vec3 collisionOffset = vec3(0.0);
        float roiSquared = repulsionRadius * repulsionRadius;

        // Global active-cell scan for repulsion (voxel-independent).
        // This avoids edge artifacts from finite voxel grids.
        uint activeCount = min(numActiveCells, uint(cells.length()));
        for (uint candidate = 0u; candidate < activeCount; ++candidate) {
            if (candidate == id || cells[candidate].isActive == 0) {
                continue;
            }

            bool isDirect = false;
            for (int j = 0; j < MAX_NEIGHBORS; ++j) {
                if (links[linkBase + j] == candidate) {
                    isDirect = true;
                    break;
                }
            }
            if (isDirect) {
                continue;
            }

            vec3 diff = P - posSnapshot[candidate].xyz;  // Read from snapshot
            float dist2 = dot(diff, diff);
            if (dist2 < roiSquared && dist2 > 1e-8) {
                collisionOffset += repulsionFactor * ((roiSquared - dist2) / roiSquared) * normalize(diff);
            }
        }

        // Planar force pulls toward average of neighbors, which on a sphere causes drift
        // Make planar force tangential to sphere surface to prevent radial drift
        vec3 planarOffset = planarTarget - P;
        if (dot(normal, normal) > 1e-8) {
            float normalComponent = dot(planarOffset, normal);
            vec3 tangentialPlanarOffset = planarOffset - normalComponent * normal;
            totalTarget = P
                + springFactor * (springTarget - P)
                + planarFactor * tangentialPlanarOffset  // Only tangential component
                + bulgeFactor * (bulgeTarget - P)
                + collisionOffset;
        } else {
            totalTarget = P
                + springFactor * (springTarget - P)
                + planarFactor * (planarTarget - P)
                + bulgeFactor * (bulgeTarget - P)
                + collisionOffset;
        }
    }
    else {
        // Planar force pulls toward average of neighbors, which on a sphere causes drift
        // Make planar force tangential to sphere surface to prevent radial drift
        vec3 planarOffset = planarTarget - P;
        if (dot(normal, normal) > 1e-8) {
            float normalComponent = dot(planarOffset, normal);
            vec3 tangentialPlanarOffset = planarOffset - normalComponent * normal;
            totalTarget = P
                + springFactor * (springTarget - P)
                + planarFactor * tangentialPlanarOffset  // Only tangential component
                + bulgeFactor * (bulgeTarget - P);
        } else {
            totalTarget = P
                + springFactor * (springTarget - P)
                + planarFactor * (planarTarget - P)
                + bulgeFactor * (bulgeTarget - P);
        }
    }

    cells[id].position = mix(P, totalTarget, timeStep);

    // after moving recalculate voxel coord
    cells[id].voxelCoord = floor(cells[id].position / voxelSize);
    ivec3 voxel_offset = ivec3(gridResolution) / 2;
    ivec3 shiftedVoxel = ivec3(cells[id].voxelCoord) + voxel_offset;
    if (any(lessThan(shiftedVoxel, ivec3(0))) || any(greaterThanEqual(shiftedVoxel, ivec3(gridResolution)))) {
        cells[id].flatVoxelIndex = -1;
    } else {
        cells[id].flatVoxelIndex = int(shiftedVoxel.x +
                                  shiftedVoxel.y * gridResolution.x +
                                  shiftedVoxel.z * gridResolution.x * gridResolution.y);
    }
}
