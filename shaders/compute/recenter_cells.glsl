#version 460
layout(local_size_x = 256) in;

struct Cell {
    vec3 position;
    float foodLevel;
    vec3 voxelCoord;
    float radius;
    int linkStartIndex;
    int linkCount;
    int flatVoxelIndex;
    int isActive;
};

layout(std430, binding = 0) buffer CellBuffer {
    Cell cells[];
};

uniform vec3 centerOffset;
uniform vec3 gridResolution;
uniform float voxelSize;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cells.length() || cells[id].isActive == 0) {
        return;
    }

    cells[id].position -= centerOffset;

    cells[id].voxelCoord = floor(cells[id].position / voxelSize);
    ivec3 voxelOffset = ivec3(gridResolution) / 2;
    ivec3 shiftedVoxel = ivec3(cells[id].voxelCoord) + voxelOffset;
    if (any(lessThan(shiftedVoxel, ivec3(0))) || any(greaterThanEqual(shiftedVoxel, ivec3(gridResolution)))) {
        cells[id].flatVoxelIndex = -1;
    } else {
        cells[id].flatVoxelIndex = int(shiftedVoxel.x +
                                       shiftedVoxel.y * gridResolution.x +
                                       shiftedVoxel.z * gridResolution.x * gridResolution.y);
    }
}
