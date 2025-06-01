#version 460
layout(local_size_x = 256) in;

struct Cell {
    vec3 position;       // 16 bytes (vec3 + padding)
    float foodLevel;     // 4  (included in above 16)
    uvec3 voxelCoord;     // 16 bytes
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

shared uint temp[256];

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    if (gid >= cellCountPerVoxel.length())
        return;

    temp[lid] = cellCountPerVoxel[gid];
    memoryBarrierShared();
    barrier();

    for (uint offset = 1; offset < 256; offset *= 2) {
        uint val = 0;
        if (lid >= offset)
            val = temp[lid - offset];

        barrier();
        temp[lid] += val;
        barrier();
    }

    if (gid > 0)
        startIndexPerVoxel[gid] = temp[lid - 1];
    else
        startIndexPerVoxel[0] = 0;
}