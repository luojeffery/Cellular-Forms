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

uniform float voxelSize;
uniform ivec3 gridResolution;

uvec3 getVoxelCoord(vec3 pos) {
    return uvec3(floor(pos / voxelSize));
}

uint flattenVoxelIndex(uvec3 v) {
    return v.x + v.y * gridResolution.x + v.z * gridResolution.x * gridResolution.y;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cells.length()) return;

    uvec3 coord = getVoxelCoord(cells[id].position);
    if (any(greaterThanEqual(coord, uvec3(gridResolution)))) return;

    uint idx = flattenVoxelIndex(coord);
    atomicAdd(cellCountPerVoxel[idx], 1);
}
