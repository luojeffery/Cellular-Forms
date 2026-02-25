#version 460
layout(local_size_x = 256) in;

// Link maintenance shader:
// 1. Breaks links that are stretched too far (prevents topology from breaking under stress)
// 2. Validates that linked cells are still active
// 
// NOTE: We do NOT create new links here. New links are only created during:
// - Initialization (CPU side)
// - Cell division (in process_division_queue.glsl)
// This preserves the surface topology as described in the Lomas paper.

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

uniform float linkBreakDistance;  // Links longer than this will be broken

const uint EMPTY = 0xFFFFFFFFu;
const int MAX_LINKS = 6;

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    vec3 P = cells[id].position;
    int selfLinkStart = cells[id].linkStartIndex;
    float breakDistSq = linkBreakDistance * linkBreakDistance;
    
    // Check each link
    for (int j = 0; j < MAX_LINKS; ++j) {
        uint linkedCell = links[selfLinkStart + j];
        
        if (linkedCell == EMPTY) continue;
        if (linkedCell >= cells.length()) {
            // Invalid link - clear it
            links[selfLinkStart + j] = EMPTY;
            continue;
        }
        
        // Check if linked cell is still active
        if (cells[linkedCell].isActive == 0) {
            // Linked cell is inactive - break the link
            links[selfLinkStart + j] = EMPTY;
            continue;
        }
        
        // Check if link is overstretched
        vec3 otherPos = cells[linkedCell].position;
        vec3 diff = otherPos - P;
        float dist2 = dot(diff, diff);
        
        if (dist2 > breakDistSq) {
            // Link is too long - break it from both sides
            // Only the lower ID cell breaks the link to avoid race conditions
            if (id < linkedCell) {
                links[selfLinkStart + j] = EMPTY;
                
                // Also remove the reverse link
                int otherLinkStart = cells[linkedCell].linkStartIndex;
                for (int k = 0; k < MAX_LINKS; ++k) {
                    if (links[otherLinkStart + k] == id) {
                        links[otherLinkStart + k] = EMPTY;
                        break;
                    }
                }
            }
        }
    }
}
