#version 460
layout(local_size_x = 256) in;

layout(std430, binding = 2) buffer CellCountPerVoxel {
    uint cellCountPerVoxel[];
};

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id < cellCountPerVoxel.length()) {
        cellCountPerVoxel[id] = 0;
    }
}