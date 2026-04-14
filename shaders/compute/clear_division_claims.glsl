#version 460
layout(local_size_x = 256) in;

// Per-cell claim flag: 0 = unclaimed.
// Cleared to 0 before every division dispatch so that stale claims
// from a previous frame never block future divisions.
layout(std430, binding = 9) buffer DivisionClaims {
    uint divideClaim[];
};

uniform int numCells;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(numCells)) return;
    divideClaim[id] = 0u;
}
