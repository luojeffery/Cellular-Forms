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

const uint EMPTY = 0xFFFFFFFFu;

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id >= cells.length() || cells[id].isActive == 0) return;

    int voxID = cells[id].flatVoxelIndex;
    uint startID = startIndexPerVoxel[voxID];
    int selfLinkStart = cells[id].linkStartIndex;

    for (uint idx = startID; idx < startID + int(cellCountPerVoxel[voxID]); idx++) {
        uint otherID = flatVoxelCellIDs[idx];
        if (otherID == id) continue;
        if (cells[otherID].isActive == 0) continue; // Skip inactive cells
        int otherLinkStart = cells[otherID].linkStartIndex;

        // Check if already linked
        bool alreadyLinked = false;
        for (int j = 0; j < 6; ++j) {
            if (links[selfLinkStart + j] == int(otherID)) {
                alreadyLinked = true;
                break;
            }
        }

        if (alreadyLinked) continue;

        // Find empty spot in self
        int selfEmpty = -1;
        for (int j = 0; j < 6; ++j) {
            if (links[selfLinkStart + j] == EMPTY) {
                selfEmpty = selfLinkStart + j;
                break;
            }
        }

        // Find empty spot in other
        int otherEmpty = -1;
        for (int j = 0; j < 6; ++j) {
            if (links[otherLinkStart + j] == EMPTY) {
                otherEmpty = otherLinkStart + j;
                break;
            }
        }

        // Only link if both have space
        if (selfEmpty != -1 && otherEmpty != -1) {
            links[selfEmpty] = otherID;
            links[otherEmpty] = id;
            // Note: linkCount will be recomputed in next recompute pass
        }
    }
}
