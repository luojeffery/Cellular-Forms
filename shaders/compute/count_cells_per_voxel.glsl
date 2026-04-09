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

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cells.length()) return;
    if (cells[id].isActive == 0) return; // skip inactive cells
    int voxelIdx = cells[id].flatVoxelIndex;
    if (voxelIdx < 0 || voxelIdx >= cellCountPerVoxel.length()) return;
    atomicAdd(cellCountPerVoxel[voxelIdx], 1); // we found a voxel to be surrounding this cell
}
