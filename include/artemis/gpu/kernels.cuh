#pragma once

#include "../util/types.hpp"
#include "../util/math.hpp"

namespace artemis {
namespace gpu {
namespace kernels {

/**
 * @brief CUDA kernel declarations for GPU-accelerated operations
 *
 * All kernels use __global__ qualifier and are launched from host code.
 * Kernels are designed for coalesced memory access with SoA layout.
 */

// ============================================================================
// Agent Update Kernels
// ============================================================================

/**
 * @brief Update agent positions based on velocities (Euler integration)
 */
__global__ void integrate_positions(
    float2* positions,
    const float2* velocities,
    float dt,
    size_t num_agents);

/**
 * @brief Apply velocity limits and damping
 */
__global__ void apply_velocity_constraints(
    float2* velocities,
    float max_speed,
    float damping,
    size_t num_agents);

/**
 * @brief Update agent ages and energy decay
 */
__global__ void update_agent_state(
    uint32_t* ages,
    float* energies,
    float energy_decay_rate,
    size_t num_agents);

// ============================================================================
// Spatial Indexing Kernels
// ============================================================================

/**
 * @brief Compute Morton codes for all agent positions
 */
__global__ void compute_morton_codes(
    MortonCode* morton_codes,
    const float2* positions,
    BoundingBox2D bounds,
    uint32_t grid_resolution,
    size_t num_agents);

/**
 * @brief Compute grid cell IDs for all agents
 */
__global__ void compute_cell_ids(
    CellID* cell_ids,
    const float2* positions,
    BoundingBox2D bounds,
    float cell_size,
    size_t num_agents);

/**
 * @brief Build cell boundary arrays from sorted agents
 */
__global__ void build_cell_boundaries(
    uint32_t* cell_starts,
    uint32_t* cell_ends,
    const CellID* sorted_cell_ids,
    size_t num_agents,
    uint32_t num_cells);

/**
 * @brief Reorder agent data based on spatial sort
 */
__global__ void reorder_agent_data(
    float2* sorted_positions,
    float2* sorted_velocities,
    uint8_t* sorted_types,
    const float2* positions,
    const float2* velocities,
    const uint8_t* types,
    const AgentID* sorted_indices,
    size_t num_agents);

// ============================================================================
// Neighbor Query Kernels
// ============================================================================

/**
 * @brief Find all neighbors within radius for each agent
 */
__global__ void query_neighbors_radius(
    const float2* query_positions,
    const float2* agent_positions,
    const uint32_t* cell_starts,
    const uint32_t* cell_ends,
    float radius,
    BoundingBox2D bounds,
    float cell_size,
    uint32_t grid_width,
    AgentID* neighbor_ids,
    uint32_t* neighbor_counts,
    uint32_t max_neighbors,
    size_t num_queries);

/**
 * @brief Find K-nearest neighbors for each agent
 */
__global__ void query_knn(
    const float2* query_positions,
    const float2* agent_positions,
    const uint32_t* cell_starts,
    const uint32_t* cell_ends,
    BoundingBox2D bounds,
    float cell_size,
    uint32_t grid_width,
    uint32_t k,
    AgentID* neighbor_ids,
    float* neighbor_distances,
    size_t num_queries);

// ============================================================================
// Collision Detection Kernels
// ============================================================================

/**
 * @brief Broad-phase collision detection using spatial grid
 */
__global__ void detect_collisions_broad(
    const float2* positions,
    const float* radii,
    const uint32_t* cell_starts,
    const uint32_t* cell_ends,
    float cell_size,
    uint32_t grid_width,
    AgentID* collision_pairs,
    uint32_t* num_collisions,
    uint32_t max_collisions,
    size_t num_agents);

/**
 * @brief Resolve collisions with elastic response
 */
__global__ void resolve_collisions(
    float2* positions,
    float2* velocities,
    const AgentID* collision_pairs,
    const float* masses,
    float restitution,
    uint32_t num_collisions);

// ============================================================================
// Boundary Condition Kernels
// ============================================================================

/**
 * @brief Apply bounded boundary conditions (reflection)
 */
__global__ void apply_bounded_boundaries(
    float2* positions,
    float2* velocities,
    BoundingBox2D bounds,
    size_t num_agents);

/**
 * @brief Apply toroidal (wrap-around) boundaries
 */
__global__ void apply_torus_boundaries(
    float2* positions,
    BoundingBox2D bounds,
    size_t num_agents);

// ============================================================================
// Metrics Computation Kernels
// ============================================================================

/**
 * @brief Compute center of mass for all agents
 */
__global__ void compute_center_of_mass(
    const float2* positions,
    float2* result,
    size_t num_agents);

/**
 * @brief Compute average velocity (polarization metric)
 */
__global__ void compute_average_velocity(
    const float2* velocities,
    float2* result,
    size_t num_agents);

/**
 * @brief Compute order parameter (alignment measure)
 */
__global__ void compute_order_parameter(
    const float2* velocities,
    float* result,
    size_t num_agents);

/**
 * @brief Compute spatial histogram (density distribution)
 */
__global__ void compute_spatial_histogram(
    const float2* positions,
    BoundingBox2D bounds,
    uint32_t* histogram,
    uint32_t bins_x,
    uint32_t bins_y,
    size_t num_agents);

/**
 * @brief Parallel reduction kernel (sum)
 */
template<typename T>
__global__ void reduce_sum(
    const T* input,
    T* output,
    size_t num_elements);

/**
 * @brief Parallel reduction kernel (min/max)
 */
template<typename T>
__global__ void reduce_minmax(
    const T* input,
    T* min_output,
    T* max_output,
    size_t num_elements);

// ============================================================================
// Utility Kernels
// ============================================================================

/**
 * @brief Initialize array with constant value
 */
template<typename T>
__global__ void fill_array(
    T* array,
    T value,
    size_t num_elements);

/**
 * @brief Copy data between arrays
 */
template<typename T>
__global__ void copy_array(
    T* dst,
    const T* src,
    size_t num_elements);

/**
 * @brief Generate random positions in bounding box
 */
__global__ void generate_random_positions(
    float2* positions,
    BoundingBox2D bounds,
    uint64_t seed,
    size_t num_agents);

} // namespace kernels
} // namespace gpu
} // namespace artemis
