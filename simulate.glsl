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

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    vec3 P = cells[id].position;
    int linkCount = cells[id].linkCount;
    int linkBase = cells[id].linkStartIndex;

    // --- DIRECT FORCES (spring, planar, bulge) ---
    vec3 springTarget = vec3(0.0);
    vec3 planarTarget = vec3(0.0);
    vec3 normal = vec3(0.0);
    float bulgeDist = 0.0;
    int validCount = 0;

    for (int i = 0; i < 6; ++i) {
        if (i >= linkCount) continue;
        uint neighborIndex = links[linkBase + i];
        if (neighborIndex == EMPTY) continue;
        vec3 L = cells[neighborIndex].position;
        vec3 toNeighbor = L - P;
        float distToNeighbor = length(toNeighbor);

        // Planar target: average of neighbor positions (paper formula)
        planarTarget += L;
        
        // Spring target: for each link, target position is neighbor + restLength * direction from neighbor to cell
        // Paper: springTarget = 1/n * sum(Lr + linkRestLength * normalize(P - Lr))
        // Note: normalize(P - L) is direction FROM neighbor L TO cell P
        vec3 dirFromNeighborToCell = normalize(P - L);
        if (length(P - L) > 0.001) {  // Avoid division by zero
            springTarget += L + linkRestLength * dirFromNeighborToCell;
        }
        validCount++;
    }

    if (validCount > 0) {
        planarTarget /= float(validCount);
        springTarget /= float(validCount);
    }

    // For a sphere centered at origin, use position vector as normal
    // This ensures symmetric expansion and prevents bias
    normal = normalize(P);
    if (length(P) < 0.001) {
        normal = vec3(0.0, 1.0, 0.0); // Avoid division by zero at origin
    }

    validCount = 0;
    for (int i = 0; i < 6; ++i) {
        if (i >= linkCount) continue;
        uint neighborIndex = links[linkBase + i];
        if (neighborIndex == EMPTY) continue;
        vec3 L = cells[neighborIndex].position;

        float dotNr = dot(L - P, normal);
        float LLen = length(L - P);
        bulgeDist += sqrt(max(0.0, linkRestLength * linkRestLength - LLen * LLen + dotNr * dotNr)) + dotNr;
        validCount++;
    }

    if (validCount > 0) {
        bulgeDist /= float(validCount);
    }

    vec3 bulgeTarget = P + bulgeDist * normal;

    vec3 totalTarget;
    if (repulsionFactor != 0) {
        vec3 collisionOffset = vec3(0.0);
        uint indirectCells[100];  // Increased for neighbor voxels
        uint indirectCount = 0;

        // Check current voxel and all 26 neighboring voxels (3x3x3 - 1)
        ivec3 currentVoxel = ivec3(cells[id].voxelCoord) + ivec3(gridResolution) / 2;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    ivec3 neighborVoxel = currentVoxel + ivec3(dx, dy, dz);
                    neighborVoxel = clamp(neighborVoxel, ivec3(0), ivec3(gridResolution) - ivec3(1));
                    uint neighborFlatVoxel = uint(neighborVoxel.x + neighborVoxel.y * int(gridResolution.x) + neighborVoxel.z * int(gridResolution.x) * int(gridResolution.y));
                    
                    uint startIndex = startIndexPerVoxel[neighborFlatVoxel];
                    uint count = cellCountPerVoxel[neighborFlatVoxel];
                    
                    for (uint i = startIndex; i < startIndex + count && indirectCount < 100; i++) {
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
                            // Check if already added
                            bool alreadyAdded = false;
                            for (uint k = 0; k < indirectCount; k++) {
                                if (indirectCells[k] == candidate) {
                                    alreadyAdded = true;
                                    break;
                                }
                            }
                            if (!alreadyAdded) {
                                indirectCells[indirectCount++] = candidate;
                            }
                        }
                    }
                }
            }
        }
        
        for (uint i = 0; i < indirectCount; i++) {
            vec3 diff = cells[id].position - cells[indirectCells[i]].position;
            float dist2 = dot(diff, diff);
            float roiSquared = repulsionRadius * repulsionRadius;
            if (dist2 < roiSquared && dist2 > 0.5)
                collisionOffset += repulsionFactor * ((roiSquared - dist2) / roiSquared) * normalize(diff);
        }

        // Planar force pulls toward average of neighbors, which on a sphere causes drift
        // Make planar force tangential to sphere surface to prevent radial drift
        vec3 planarOffset = planarTarget - P;
        vec3 radialDir = normalize(P);
        if (length(P) > 0.001) {
            float radialComponent = dot(planarOffset, radialDir);
            vec3 tangentialPlanarOffset = planarOffset - radialComponent * radialDir;
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
        vec3 radialDir = normalize(P);
        if (length(P) > 0.001) {
            float radialComponent = dot(planarOffset, radialDir);
            vec3 tangentialPlanarOffset = planarOffset - radialComponent * radialDir;
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
    // Clamp to valid voxel grid bounds
    shiftedVoxel = clamp(shiftedVoxel, ivec3(0), ivec3(gridResolution) - ivec3(1));
    cells[id].flatVoxelIndex = int(shiftedVoxel.x +
                              shiftedVoxel.y * gridResolution.x +
                              shiftedVoxel.z * gridResolution.x * gridResolution.y);
}
