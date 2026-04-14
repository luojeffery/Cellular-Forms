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
layout(std430, binding = 1) buffer LinkBuffer {
    uint links[];
};

const uint EMPTY = 0xFFFFFFFFu;
const int MAX_NEIGHBORS = 8;

bool has_reciprocal(uint selfId, uint otherId) {
    int otherBase = cells[otherId].linkStartIndex;
    for (int slot = 0; slot < MAX_NEIGHBORS; ++slot) {
        if (links[otherBase + slot] == selfId) {
            return true;
        }
    }
    return false;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cells.length() || cells[id].isActive == 0) {
        return;
    }

    int base = cells[id].linkStartIndex;

    for (int slot = 0; slot < MAX_NEIGHBORS; ++slot) {
        uint neighbor = links[base + slot];
        if (neighbor == EMPTY) {
            continue;
        }

        bool invalid = neighbor >= cells.length() ||
                       cells[neighbor].isActive == 0 ||
                       neighbor == id;

        bool duplicate = false;
        for (int prev = 0; prev < slot; ++prev) {
            if (links[base + prev] == neighbor) {
                duplicate = true;
                break;
            }
        }

        if (invalid || duplicate || !has_reciprocal(id, neighbor)) {
            links[base + slot] = EMPTY;
        }
    }
}