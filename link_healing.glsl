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

uniform vec3 gridResolution;
uniform float voxelSize;
uniform float linkHealingRadius;  // Maximum distance for creating new links

const uint EMPTY = 0xFFFFFFFFu;
const int MAX_LINKS = 6;

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    vec3 P = cells[id].position;
    int selfLinkStart = cells[id].linkStartIndex;
    
    // Count current links and find empty slots
    int currentLinkCount = 0;
    int emptySlots[MAX_LINKS];
    int numEmptySlots = 0;
    
    for (int j = 0; j < MAX_LINKS; ++j) {
        uint linkVal = links[selfLinkStart + j];
        if (linkVal == EMPTY) {
            emptySlots[numEmptySlots++] = selfLinkStart + j;
        } else {
            currentLinkCount++;
        }
    }
    
    // If we have no empty slots, nothing to do
    if (numEmptySlots == 0) return;
    
    // Search neighboring voxels for potential link candidates
    ivec3 gridRes = ivec3(gridResolution);
    ivec3 voxelOffset = gridRes / 2;
    ivec3 currentVoxel = ivec3(floor(P / voxelSize)) + voxelOffset;
    currentVoxel = clamp(currentVoxel, ivec3(0), gridRes - ivec3(1));
    
    float healingRadiusSq = linkHealingRadius * linkHealingRadius;
    
    // Collect candidates with their distances
    uint candidates[32];
    float candidateDists[32];
    int numCandidates = 0;
    
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
                
                for (uint idx = startIndex; idx < startIndex + count && numCandidates < 32; idx++) {
                    uint otherID = flatVoxelCellIDs[idx];
                    if (otherID == id) continue;
                    if (otherID >= cells.length()) continue;
                    if (cells[otherID].isActive == 0) continue;
                    
                    // Check distance
                    vec3 diff = cells[otherID].position - P;
                    float dist2 = dot(diff, diff);
                    
                    if (dist2 > healingRadiusSq) continue;
                    
                    // Check if already linked
                    bool alreadyLinked = false;
                    for (int j = 0; j < MAX_LINKS; ++j) {
                        if (links[selfLinkStart + j] == otherID) {
                            alreadyLinked = true;
                            break;
                        }
                    }
                    
                    if (!alreadyLinked) {
                        candidates[numCandidates] = otherID;
                        candidateDists[numCandidates] = dist2;
                        numCandidates++;
                    }
                }
            }
        }
    }
    
    // Sort candidates by distance (simple bubble sort for small arrays)
    for (int i = 0; i < numCandidates - 1; i++) {
        for (int j = 0; j < numCandidates - i - 1; j++) {
            if (candidateDists[j] > candidateDists[j + 1]) {
                // Swap
                float tmpDist = candidateDists[j];
                candidateDists[j] = candidateDists[j + 1];
                candidateDists[j + 1] = tmpDist;
                
                uint tmpId = candidates[j];
                candidates[j] = candidates[j + 1];
                candidates[j + 1] = tmpId;
            }
        }
    }
    
    // Try to create links with closest candidates
    // Use atomics to avoid race conditions
    int emptySlotIdx = 0;
    
    for (int i = 0; i < numCandidates && emptySlotIdx < numEmptySlots; i++) {
        uint otherID = candidates[i];
        int otherLinkStart = cells[otherID].linkStartIndex;
        
        // Find empty spot in other cell
        // To avoid race conditions, only the cell with lower ID creates the link
        // This ensures only one thread handles each potential link
        if (id > otherID) continue;
        
        int otherEmpty = -1;
        for (int j = 0; j < MAX_LINKS; ++j) {
            if (links[otherLinkStart + j] == EMPTY) {
                otherEmpty = otherLinkStart + j;
                break;
            }
        }
        
        // Only link if other has space
        if (otherEmpty != -1) {
            // Use atomicCompSwap to avoid race conditions
            // Try to claim our slot
            uint oldSelf = atomicCompSwap(links[emptySlots[emptySlotIdx]], EMPTY, otherID);
            if (oldSelf == EMPTY) {
                // We got our slot, now try to claim other's slot
                uint oldOther = atomicCompSwap(links[otherEmpty], EMPTY, id);
                if (oldOther == EMPTY) {
                    // Success! Both links created
                    emptySlotIdx++;
                } else {
                    // Failed to get other's slot, undo our slot
                    links[emptySlots[emptySlotIdx]] = EMPTY;
                }
            }
        }
    }
}
