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

// Compute local surface normal from neighbors using cross products
// This is the correct approach per the Lomas paper - the normal should be
// perpendicular to the local surface defined by neighboring cells
vec3 computeSurfaceNormal(vec3 P, int linkBase, int linkCount) {
    // Collect valid neighbor positions
    vec3 neighbors[6];
    int numNeighbors = 0;
    
    for (int i = 0; i < 6 && i < linkCount; ++i) {
        uint neighborIndex = links[linkBase + i];
        if (neighborIndex != EMPTY && neighborIndex < cells.length()) {
            neighbors[numNeighbors++] = cells[neighborIndex].position;
        }
    }
    
    if (numNeighbors < 3) {
        // Not enough neighbors to compute a proper normal
        // Fall back to radial direction (works for sphere-like shapes)
        vec3 radial = normalize(P);
        return length(P) > 0.001 ? radial : vec3(0.0, 1.0, 0.0);
    }
    
    // Compute normal by averaging cross products of consecutive neighbor vectors
    // This gives us the normal to the local surface patch
    vec3 normal = vec3(0.0);
    vec3 center = vec3(0.0);
    
    // First compute centroid of neighbors
    for (int i = 0; i < numNeighbors; ++i) {
        center += neighbors[i];
    }
    center /= float(numNeighbors);
    
    // Compute normal using cross products of vectors from P to consecutive neighbors
    // We need to be careful about winding order
    for (int i = 0; i < numNeighbors; ++i) {
        vec3 v1 = neighbors[i] - P;
        vec3 v2 = neighbors[(i + 1) % numNeighbors] - P;
        normal += cross(v1, v2);
    }
    
    float normalLen = length(normal);
    if (normalLen < 0.001) {
        // Degenerate case - neighbors are collinear or P is at centroid
        vec3 radial = normalize(P);
        return length(P) > 0.001 ? radial : vec3(0.0, 1.0, 0.0);
    }
    
    normal = normalize(normal);
    
    // Ensure normal points outward (away from origin for initial sphere)
    // This heuristic works for convex shapes expanding from origin
    if (dot(normal, P) < 0.0) {
        normal = -normal;
    }
    
    return normal;
}

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    vec3 P = cells[id].position;
    int linkCount = cells[id].linkCount;
    int linkBase = cells[id].linkStartIndex;

    // Skip cells with no links - they can't participate in physics properly
    if (linkCount == 0) return;

    // --- DIRECT FORCES (spring, planar, bulge) ---
    vec3 springTarget = vec3(0.0);
    vec3 planarTarget = vec3(0.0);
    int validCount = 0;

    for (int i = 0; i < 6; ++i) {
        if (i >= linkCount) break;
        uint neighborIndex = links[linkBase + i];
        if (neighborIndex == EMPTY || neighborIndex >= cells.length()) continue;
        if (cells[neighborIndex].isActive == 0) continue;
        
        vec3 L = cells[neighborIndex].position;
        vec3 toNeighbor = L - P;
        float distToNeighbor = length(toNeighbor);

        // Planar target: average of neighbor positions (paper formula)
        planarTarget += L;
        
        // Spring target: for each link, target position is neighbor + restLength * direction from neighbor to cell
        // Paper: springTarget = 1/n * sum(Lr + linkRestLength * normalize(P - Lr))
        if (distToNeighbor > 0.0001) {
            vec3 dirFromNeighborToCell = (P - L) / distToNeighbor;
            springTarget += L + linkRestLength * dirFromNeighborToCell;
        } else {
            // Cells at same position - push apart in arbitrary direction
            springTarget += L + linkRestLength * vec3(1.0, 0.0, 0.0);
        }
        validCount++;
    }

    if (validCount == 0) return;
    
    planarTarget /= float(validCount);
    springTarget /= float(validCount);

    // Compute proper surface normal from neighbors (per Lomas paper)
    vec3 normal = computeSurfaceNormal(P, linkBase, linkCount);

    // Bulge force calculation (per Lomas paper)
    // For each neighbor, compute how far along the normal we need to move
    // to restore the link to rest length
    float bulgeDist = 0.0;
    int bulgeCount = 0;
    
    for (int i = 0; i < 6; ++i) {
        if (i >= linkCount) break;
        uint neighborIndex = links[linkBase + i];
        if (neighborIndex == EMPTY || neighborIndex >= cells.length()) continue;
        if (cells[neighborIndex].isActive == 0) continue;
        
        vec3 L = cells[neighborIndex].position;
        vec3 toNeighbor = L - P;
        
        // Project neighbor vector onto normal
        float dotNr = dot(toNeighbor, normal);
        float distSq = dot(toNeighbor, toNeighbor);
        
        // The bulge distance formula from the paper:
        // d = sqrt(restLength^2 - |L-P|^2 + (dot(L-P, n))^2) + dot(L-P, n)
        // This computes how far along the normal we need to move to make the
        // distance to the neighbor equal to restLength
        float underSqrt = linkRestLength * linkRestLength - distSq + dotNr * dotNr;
        if (underSqrt >= 0.0) {
            bulgeDist += sqrt(underSqrt) + dotNr;
        }
        bulgeCount++;
    }

    if (bulgeCount > 0) {
        bulgeDist /= float(bulgeCount);
    }

    vec3 bulgeTarget = P + bulgeDist * normal;

    // --- REPULSION FORCE (indirect neighbors) ---
    vec3 collisionOffset = vec3(0.0);
    
    if (repulsionFactor > 0.0) {
        // Check current voxel and all 26 neighboring voxels (3x3x3)
        ivec3 gridRes = ivec3(gridResolution);
        ivec3 voxelOffset = gridRes / 2;
        ivec3 currentVoxel = ivec3(floor(P / voxelSize)) + voxelOffset;
        currentVoxel = clamp(currentVoxel, ivec3(0), gridRes - ivec3(1));
        
        float roiSquared = repulsionRadius * repulsionRadius;
        
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    ivec3 neighborVoxel = currentVoxel + ivec3(dx, dy, dz);
                    
                    // Skip if outside grid
                    if (any(lessThan(neighborVoxel, ivec3(0))) || 
                        any(greaterThanEqual(neighborVoxel, gridRes))) {
                        continue;
                    }
                    
                    int neighborFlatVoxel = neighborVoxel.x + 
                                           neighborVoxel.y * gridRes.x + 
                                           neighborVoxel.z * gridRes.x * gridRes.y;
                    
                    uint startIndex = startIndexPerVoxel[neighborFlatVoxel];
                    uint count = cellCountPerVoxel[neighborFlatVoxel];
                    
                    for (uint i = startIndex; i < startIndex + count; i++) {
                        uint candidate = flatVoxelCellIDs[i];
                        if (candidate == id) continue;
                        if (candidate >= cells.length()) continue;
                        if (cells[candidate].isActive == 0) continue;

                        // Check if this is a direct neighbor (linked cell)
                        bool isDirect = false;
                        for (int j = 0; j < linkCount && j < 6; j++) {
                            if (links[linkBase + j] == candidate) {
                                isDirect = true;
                                break;
                            }
                        }
                        
                        // Only apply repulsion to non-linked cells
                        if (!isDirect) {
                            vec3 diff = P - cells[candidate].position;
                            float dist2 = dot(diff, diff);
                            
                            // Apply repulsion if within range
                            // Removed the dist2 > 0.5 check - we want repulsion even at very close range
                            if (dist2 < roiSquared && dist2 > 0.0001) {
                                float dist = sqrt(dist2);
                                // Smooth falloff: stronger when closer
                                float strength = (roiSquared - dist2) / roiSquared;
                                collisionOffset += repulsionFactor * strength * (diff / dist);
                            }
                        }
                    }
                }
            }
        }
    }

    // Combine all forces
    // The paper uses direct position updates (no momentum) assuming highly viscous medium
    vec3 springForce = springFactor * (springTarget - P);
    vec3 planarForce = planarFactor * (planarTarget - P);
    vec3 bulgeForce = bulgeFactor * (bulgeTarget - P);
    
    vec3 totalForce = springForce + planarForce + bulgeForce + collisionOffset;
    
    // Update position with timestep
    cells[id].position = P + totalForce * timeStep;

    // After moving, recalculate voxel coordinates
    vec3 newPos = cells[id].position;
    cells[id].voxelCoord = floor(newPos / voxelSize);
    ivec3 voxel_offset = ivec3(gridResolution) / 2;
    ivec3 shiftedVoxel = ivec3(cells[id].voxelCoord) + voxel_offset;
    shiftedVoxel = clamp(shiftedVoxel, ivec3(0), ivec3(gridResolution) - ivec3(1));
    cells[id].flatVoxelIndex = int(shiftedVoxel.x +
                              shiftedVoxel.y * gridResolution.x +
                              shiftedVoxel.z * gridResolution.x * gridResolution.y);
}
