# Cellular Forms Simulation - GPU-Accelerated Morphogenesis

## Overview

This project implements **Cellular Forms**, a GPU-accelerated simulation of morphogenesis (biological growth) based on Andy Lomas's paper "Cellular Forms: an Artistic Exploration of Morphogenesis" (AISB50). The system simulates cellular growth through division, creating intricate organic structures reminiscent of plants, corals, and internal organs.

**Key Concept**: Simple rules governing cell interactions and division produce complex emergent forms without explicit top-down design.

## Theoretical Foundation

### Core Model (from Lomas Paper)

The simulation models cells as particles connected in a 2D surface topology (topologically equivalent to a sphere). Each cell:
- Has a position in 3D space
- Maintains links to neighboring cells (max 6 links per cell)
- Accumulates food/nutrients internally
- Divides when food exceeds a threshold

### Physics Model

The system uses three primary forces between directly linked cells:

1. **Spring Force** (`springFactor`): Maintains constant distance between linked cells
   - Formula: `springTarget = 1/n * Σ(Lr + linkRestLength * normalize(P - Lr))`
   - Where `Lr` are neighbor positions, `P` is current cell position

2. **Planar Force** (`planarFactor`): Reduces folds and bumps, restoring surface to local planar state
   - Formula: `planarTarget = 1/n * Σ(Lr)` (average of neighbor positions)
   - **Note**: In this implementation, planar force is made tangential to prevent radial drift on spheres

3. **Bulge Force** (`bulgeFactor`): Bulges surface outward when links are in compression
   - Calculates distance along surface normal to restore links to rest length
   - Formula uses cosine law: `bulgeDist = 1/n * Σ(sqrt(linkRestLength² - |Lr|² + dotNr²) + dotNr)`

**Repulsion Force**: Cells not directly linked but in close proximity experience repulsion to prevent self-intersection and maintain structural coherence.

**Final Position Update**:
```
P' = P + springFactor * (springTarget - P)
    + planarFactor * (planarTarget - P)  
    + bulgeFactor * (bulgeTarget - P)
    + collisionOffset
```

The system uses **no momentum** - positions are updated directly based on restoring forces (highly viscous medium assumption).

### Cell Division

When a cell's `foodLevel` exceeds `foodThreshold`:
1. Cell is enqueued for division
2. A daughter cell is created near the parent
3. Half of parent's links are severed and reassigned to daughter
4. Parent-daughter link is created
5. Neighbor links are updated to point to daughter instead of parent

## Architecture

### Technology Stack

- **Language**: Python 3.9+ (orchestration)
- **Graphics API**: OpenGL 4.6 with compute shaders
- **GPU Compute**: GLSL compute shaders for parallel simulation
- **Rendering**: Deferred rendering with SSAO (Screen-Space Ambient Occlusion)
- **Libraries**: PyOpenGL, GLFW, imgui, numpy, pyglm

### Data Structures

#### Cell Structure (64 bytes, std430 layout)
```glsl
struct Cell {
    vec3 position;        // 16 bytes (vec3 + padding)
    float foodLevel;      // 4 bytes
    vec3 voxelCoord;      // 16 bytes
    float radius;         // 4 bytes
    int linkStartIndex;   // 4 bytes
    int linkCount;        // 4 bytes
    int flatVoxelIndex;   // 4 bytes
    int isActive;         // 4 bytes
};
```

#### Shader Storage Buffer Objects (SSBOs)

| Binding | Buffer | Purpose |
|---------|--------|---------|
| 0 | `CellBuffer` | Array of all cells |
| 1 | `LinkBuffer` | Flat array of cell neighbor indices (max 6 per cell) |
| 2 | `CellCountPerVoxel` | Count of cells per voxel (spatial grid) |
| 3 | `StartIndexPerVoxel` | Starting index in flat array for each voxel |
| 4 | `FlatVoxelCellIDs` | Flattened array of cell IDs organized by voxel |
| 5 | `GlobalCounts` | `[numActiveCells, divisionQueueCount]` |
| 7 | `DivisionQueue` | Queue of cell indices ready to divide |

### Spatial Acceleration Structure

A **voxel grid** is used to accelerate neighbor queries for repulsion calculations:
- Grid resolution: `GRID_RES × GRID_RES × GRID_RES` (default: 4×4×4 = 64 voxels)
- Voxel size: `VOXEL_SIZE` (default: 4.0 units)
- Each cell stores its `voxelCoord` and `flatVoxelIndex`
- Grid is centered at origin (cells have shifted coordinates)

**Voxel Grid Pipeline** (each frame):
1. `clear_cell_counts.glsl`: Clear cell counts per voxel
2. `count_cells_per_voxel.glsl`: Count cells in each voxel
3. `prefix_sum_voxel_offsets.glsl`: Compute prefix sum for indexing
4. `fill_voxel_cell_ids.glsl`: Fill flat array with cell IDs organized by voxel

## Compute Shader Pipeline

The simulation runs entirely on GPU using compute shaders. Each frame executes:

### 1. Spatial Grid Update
- **clear_cell_counts.glsl**: Reset voxel cell counts
- **count_cells_per_voxel.glsl**: Count cells per voxel
- **prefix_sum_voxel_offsets.glsl**: Compute start indices for each voxel
- **fill_voxel_cell_ids.glsl**: Populate flat voxel→cell mapping

### 2. Food & Division Queue
- **food_enqueue.glsl**: 
  - Adds random food increment to each active cell
  - Enqueues cells with `foodLevel >= foodThreshold` for division
  - Uses atomic operations for thread-safe queue insertion

### 3. Cell Division
- **process_division_queue.glsl**:
  - Processes queued divisions
  - Creates daughter cells
  - Updates link topology (severs/reassigns links)
  - Updates neighbor references

### 4. Link Management
- **recompute_link_count.glsl**: Recomputes `linkCount` after topology changes
- **link_healing.glsl**: Creates new links between nearby unlinked cells in same voxel

### 5. Physics Simulation
- **simulate.glsl**: 
  - Computes spring, planar, and bulge forces
  - Calculates repulsion from non-linked neighbors (using voxel grid)
  - Updates cell positions
  - Recalculates voxel coordinates

## Rendering Pipeline

### Deferred Rendering with SSAO

1. **Geometry Pass** (`vs.ssao_geometry.glsl` / `fs.ssao_geometry.glsl`):
   - Renders to G-Buffer: position, normal, albedo
   - Instanced rendering of cells (spheres) from SSBO data
   - Also renders scene objects (backpack, cube room)

2. **SSAO Pass** (`vs.ssao.glsl` / `fs.ssao.glsl`):
   - Samples 64 random points in hemisphere around each pixel
   - Uses position and normal from G-Buffer
   - Calculates ambient occlusion based on depth differences

3. **Blur Pass** (`fs.ssao_blur.glsl`):
   - Blurs SSAO texture to reduce noise

4. **Lighting Pass** (`vs.ssao.glsl` / `fs.ssao_lighting.glsl`):
   - Combines G-Buffer data with SSAO
   - Applies point light with attenuation
   - Outputs final lit scene

## Key Implementation Details

### Initialization

Cells start as a **hollow sphere** with uniform spacing using golden angle spiral:
- Golden angle: `π * (3 - √5)`
- Cells distributed on sphere surface
- Initial links created to nearest neighbors (up to 6)
- Links stored in flat array: `links[cell.linkStartIndex + i]`

### Thread Safety

- **Atomic operations** used for:
  - Division queue insertion (`atomicAdd(divisionQueueCount, 1u)`)
  - Active cell count (`atomicAdd(numActiveCells, 1u)`)
- **Memory barriers** (`glMemoryBarrier`) ensure data consistency between compute passes

### Link Management

- Links are **bidirectional** (symmetric)
- Empty links marked with `EMPTY = 0xFFFFFFFF`
- Maximum 6 links per cell (hardcoded limit)
- Link healing creates new links between nearby unlinked cells

### Parameter Tuning

Key simulation parameters (in `main.py`):
- `linkRestLength`: Target distance between linked cells (default: 0.6)
- `springFactor`: Strength of spring force (default: 1.0)
- `planarFactor`: Strength of planar smoothing (default: 0.0 - disabled)
- `bulgeFactor`: Strength of bulge force (default: 1.0)
- `repulsionFactor`: Strength of repulsion (default: 0.08)
- `repulsionRadius`: Radius of repulsion influence (default: 2.0)
- `timeStep`: Integration timestep (default: 0.005)
- `foodThreshold`: Food level required for division (default: 1000)

## File Structure

```
ssao/
├── main.py                    # Main application loop, initialization, rendering
├── simulate.glsl              # Physics simulation compute shader
├── food_enqueue.glsl          # Food accumulation and division queue
├── process_division_queue.glsl # Cell division logic
├── link_healing.glsl          # Creates links between nearby cells
├── recompute_link_count.glsl  # Updates link counts after topology changes
├── count_cells_per_voxel.glsl # Spatial grid: count cells per voxel
├── prefix_sum_voxel_offsets.glsl # Spatial grid: prefix sum
├── fill_voxel_cell_ids.glsl  # Spatial grid: fill cell ID array
├── clear_cell_counts.glsl     # Spatial grid: clear counts
├── vs.ssao_geometry.glsl      # Geometry pass vertex shader
├── fs.ssao_geometry.glsl      # Geometry pass fragment shader
├── vs.ssao.glsl               # SSAO/lighting pass vertex shader
├── fs.ssao.glsl               # SSAO fragment shader
├── fs.ssao_blur.glsl          # SSAO blur fragment shader
├── fs.ssao_lighting.glsl      # Final lighting fragment shader
├── compute_shader.py          # Compute shader wrapper class
├── shader.py                  # Shader compilation utilities
├── camera.py                  # Camera controller
├── model.py                   # 3D model loader
└── mesh.py                    # Mesh utilities
```

## Current State & Limitations

### Current Implementation
- ✅ Basic cellular growth simulation
- ✅ GPU-accelerated physics
- ✅ Cell division with link topology updates
- ✅ Spatial acceleration (voxel grid)
- ✅ Deferred rendering with SSAO
- ✅ Real-time visualization

### Known Limitations
- Maximum 6 links per cell (hardcoded)
- Fixed cell capacity (`NUM_CELLS`)
- Simple division strategy (splits links in half)
- No cell differentiation (all cells identical)
- No reaction-diffusion equations (mentioned in paper but not implemented)
- No light-based nutrient creation (paper feature not implemented)
- Planar force disabled by default (causes drift issues)

### Future Enhancements (from paper)
- Cell differentiation with varying parameters
- Reaction-diffusion equations for growth patterns
- Light-based nutrient creation
- Cell death and hole formation
- Volumetric cell arrangements (currently surface-only)

## Running the Simulation

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python main.py
```

**Controls**:
- `WASD`: Move camera
- `Space/Shift`: Move up/down
- `E`: Toggle cursor (enable/disable mouse look)
- `ESC`: Exit
- UI checkboxes: Toggle SSAO, toggle AO-only view

## References

- **Primary Paper**: Lomas, A. "Cellular Forms: an Artistic Exploration of Morphogenesis" (AISB50)
- **Related Work**: 
  - Kaandorp's "Accretive Growth" model
  - George Hart's "Growth Forms"
  - Turing's reaction-diffusion equations
  - D'Arcy Thompson's "On Growth and Form"

## Notes for AI Coding Agents

### When Modifying Physics
- Always update voxel coordinates after position changes
- Ensure memory barriers between compute passes
- Use atomic operations for shared counters
- Check `isActive` flag before processing cells

### When Adding Features
- Follow the existing SSBO binding convention
- Use compute shader local size of 256
- Maintain 64-byte cell struct alignment
- Update spatial grid if adding spatial queries

### Debugging Tips
- Use `read_ssbo()` function to inspect GPU buffers
- Check `debug_ssbos()` for cell/link state
- Monitor `active_cells` count in UI
- Verify voxel grid consistency if repulsion fails

### Performance Considerations
- All simulation runs on GPU (no CPU-GPU transfers per frame)
- Voxel grid limits neighbor search to 27 voxels (3×3×3)
- Division queue limits prevent unbounded growth
- Link healing only checks same voxel (fast but limited)

---

**Project Goal**: Create emergent organic forms through simple cellular growth rules, demonstrating how complexity arises from simplicity in biological systems.
