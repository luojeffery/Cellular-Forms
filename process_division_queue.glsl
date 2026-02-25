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
uniform float voxelSize;
uniform vec3 gridResolution;

const uint EMPTY = 0xFFFFFFFFu;
const int MAX_LINKS = 6;

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

vec3 randVec3(uint seed) {
    float rx = randFloat(seed * 3u + 0u) * 2.0 - 1.0;
    float ry = randFloat(seed * 3u + 1u) * 2.0 - 1.0;
    float rz = randFloat(seed * 3u + 2u) * 2.0 - 1.0;
    return normalize(vec3(rx, ry, rz));
}

void main() {
    uint queueIndex = gl_GlobalInvocationID.x;
    
    if (queueIndex >= divisionQueueCount) return;
    
    uint parentId = divisionQueue[queueIndex];
    if (parentId >= cells.length() || cells[parentId].isActive == 0) return;

    // Allocate new cell index
    uint daughterId = atomicAdd(numActiveCells, 1u);
    if (daughterId >= numCells) {
        atomicAdd(numActiveCells, -1u);
        return;
    }

    vec3 parentPos = cells[parentId].position;
    int parentBase = cells[parentId].linkStartIndex;
    int daughterBase = int(daughterId * MAX_LINKS);
    
    // Collect parent's current neighbors and their positions
    uint parentNeighbors[MAX_LINKS];
    vec3 neighborPositions[MAX_LINKS];
    int numParentNeighbors = 0;
    
    for (int i = 0; i < MAX_LINKS; i++) {
        uint neighbor = links[parentBase + i];
        if (neighbor != EMPTY && neighbor < cells.length() && cells[neighbor].isActive == 1) {
            parentNeighbors[numParentNeighbors] = neighbor;
            neighborPositions[numParentNeighbors] = cells[neighbor].position;
            numParentNeighbors++;
        }
    }
    
    // Choose division direction based on neighbors
    // The daughter should be placed between the parent and some of its neighbors
    // This helps maintain surface topology
    vec3 divisionDir;
    if (numParentNeighbors >= 2) {
        // Use direction perpendicular to the first two neighbors
        vec3 v1 = normalize(neighborPositions[0] - parentPos);
        vec3 v2 = normalize(neighborPositions[1] - parentPos);
        divisionDir = normalize(cross(v1, v2));
        
        // Add some randomness to avoid degenerate cases
        divisionDir = normalize(divisionDir + 0.3 * randVec3(parentId + queueIndex));
    } else {
        // Not enough neighbors, use random direction
        divisionDir = randVec3(parentId + queueIndex);
    }
    
    // Position daughter slightly offset from parent
    float offsetDist = 0.15;  // Small offset, will be corrected by spring forces
    vec3 daughterPos = parentPos + offsetDist * divisionDir;
    
    // Initialize daughter cell
    cells[daughterId].isActive = 1;
    cells[daughterId].position = daughterPos;
    cells[daughterId].radius = 0.1;
    cells[daughterId].foodLevel = 0.0;
    cells[daughterId].linkStartIndex = daughterBase;
    cells[daughterId].linkCount = 0;
    
    // Compute daughter's voxel coordinates
    vec3 voxelCoord = floor(daughterPos / voxelSize);
    cells[daughterId].voxelCoord = voxelCoord;
    ivec3 gridRes = ivec3(gridResolution);
    ivec3 voxelOffset = gridRes / 2;
    ivec3 shiftedVoxel = ivec3(voxelCoord) + voxelOffset;
    shiftedVoxel = clamp(shiftedVoxel, ivec3(0), gridRes - ivec3(1));
    cells[daughterId].flatVoxelIndex = shiftedVoxel.x + shiftedVoxel.y * gridRes.x + shiftedVoxel.z * gridRes.x * gridRes.y;
    
    // Initialize daughter's link slots to EMPTY
    for (int i = 0; i < MAX_LINKS; ++i) {
        links[daughterBase + i] = EMPTY;
    }
    
    // Determine which neighbors go to daughter based on position
    // Neighbors on the "daughter side" of the division plane go to daughter
    // This maintains better surface topology than arbitrary splitting
    int daughterLinkIdx = 0;
    int parentLinkIdx = 0;
    
    // First, clear parent's links (we'll rebuild them)
    for (int i = 0; i < MAX_LINKS; i++) {
        links[parentBase + i] = EMPTY;
    }
    
    // Assign neighbors to parent or daughter based on which side of division plane they're on
    for (int i = 0; i < numParentNeighbors; i++) {
        uint neighbor = parentNeighbors[i];
        vec3 neighborPos = neighborPositions[i];
        
        // Check which side of the division plane this neighbor is on
        vec3 toNeighbor = neighborPos - parentPos;
        float side = dot(toNeighbor, divisionDir);
        
        if (side > 0.0 && daughterLinkIdx < MAX_LINKS - 1) {
            // Neighbor is on daughter's side - give to daughter
            links[daughterBase + daughterLinkIdx] = neighbor;
            daughterLinkIdx++;
            
            // Update neighbor's link: replace parent with daughter
            int neighborBase = cells[neighbor].linkStartIndex;
            for (int j = 0; j < MAX_LINKS; j++) {
                if (links[neighborBase + j] == parentId) {
                    links[neighborBase + j] = daughterId;
                    break;
                }
            }
        } else if (parentLinkIdx < MAX_LINKS - 1) {
            // Neighbor stays with parent
            links[parentBase + parentLinkIdx] = neighbor;
            parentLinkIdx++;
        }
    }
    
    // Create parent-daughter link
    if (parentLinkIdx < MAX_LINKS) {
        links[parentBase + parentLinkIdx] = daughterId;
    }
    if (daughterLinkIdx < MAX_LINKS) {
        links[daughterBase + daughterLinkIdx] = parentId;
    }
    
    // Note: linkCount will be recomputed in a separate pass
}
