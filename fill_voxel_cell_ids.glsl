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


void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cells.length()) return;
    if (cells[id].isActive == 0) return; // Skip inactive cells
    int voxelIdx = cells[id].flatVoxelIndex;
    if (voxelIdx < 0 || voxelIdx >= gridResolution.x * gridResolution.y * gridResolution.z) {
        // Invalid voxel index, skip processing
        return;
    }
    // it is necessary to count up the number of cells per voxel. this way we can properly append starting from the
    // global start index.

    // localIndexInVoxel is the index of this cell within its voxel
    uint localIndexInVoxel = atomicAdd(cellCountPerVoxel[voxelIdx], 1);

    // globalStartIndex is the start of this voxel's section in the flat ID list
    uint globalStartIndex = startIndexPerVoxel[voxelIdx];

    // Write the cell ID into the voxel's slot in the flat list
    flatVoxelCellIDs[globalStartIndex + localIndexInVoxel] = id;
}
