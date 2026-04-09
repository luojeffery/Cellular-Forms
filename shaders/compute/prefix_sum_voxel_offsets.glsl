#version 460
layout(local_size_x = 256) in;

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
    uint n = uint(cellCountPerVoxel.length());

    temp[lid] = (gid < n) ? cellCountPerVoxel[gid] : 0u;
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

    if (gid < n) {
        if (gid > 0) {
            startIndexPerVoxel[gid] = temp[lid - 1];
        } else {
            startIndexPerVoxel[0] = 0u;
        }
    }
}