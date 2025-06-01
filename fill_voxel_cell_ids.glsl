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
layout(std430, binding = 2) buffer CellCountPerVoxel {
    uint cellCountPerVoxel[];
};
layout(std430, binding = 3) buffer StartIndexPerVoxel {
    uint startIndexPerVoxel[];
};
layout(std430, binding = 4) buffer FlatVoxelCellIDs {
    uint flatVoxelCellIDs[];
};

uniform float voxelSize;
uniform ivec3 gridResolution;

uvec3 getVoxelCoord(vec3 pos) {
    return uvec3(floor(pos / voxelSize));
}

uint flattenVoxelIndex(uvec3 v) {
    return v.x + v.y * gridResolution.x + v.z * gridResolution.x * gridResolution.y;
}

void main() {
    uint cellID = gl_GlobalInvocationID.x;
    if (cellID >= cells.length()) return;

    // Determine the voxel this cell falls into
    uvec3 voxelCoord = getVoxelCoord(cells[cellID].position);
    if (any(greaterThanEqual(voxelCoord, uvec3(gridResolution)))) return;

    uint voxelIdx = flattenVoxelIndex(voxelCoord);

    // localIndexInVoxel is the index of this cell within its voxel
    uint localIndexInVoxel = atomicAdd(cellCountPerVoxel[voxelIdx], 1);

    // globalStartIndex is the start of this voxel's section in the flat ID list
    uint globalStartIndex = startIndexPerVoxel[voxelIdx];

    // Write the cell ID into the voxel's slot in the flat list
    flatVoxelCellIDs[globalStartIndex + localIndexInVoxel] = cellID;
}
