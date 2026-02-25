#version 460
layout(local_size_x = 256) in;

// This is a simple sequential prefix sum that works for any size
// For small voxel counts (< 1024), this is efficient enough
// For larger counts, a proper parallel prefix sum would be needed

layout(std430, binding = 2) buffer CellCountPerVoxel {
    uint cellCountPerVoxel[];
};
layout(std430, binding = 3) buffer StartIndexPerVoxel {
    uint startIndexPerVoxel[];
};

uniform uint numVoxels;

void main() {
    // Only thread 0 computes the prefix sum
    // This is simple and correct, though not maximally parallel
    // For typical voxel counts (64-4096), this is fast enough
    if (gl_GlobalInvocationID.x != 0) return;
    
    uint runningSum = 0;
    for (uint i = 0; i < numVoxels; i++) {
        startIndexPerVoxel[i] = runningSum;
        runningSum += cellCountPerVoxel[i];
    }
}