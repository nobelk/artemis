#pragma once

#include "../util/types.hpp"
#include "../util/gpu_util.hpp"
#include "../core/agent_arrays.hpp"
#include <vector>

namespace artemis {
namespace gpu {

/**
 * @brief GPU-accelerated spatial index for efficient neighbor queries
 *
 * Implements a uniform grid with Morton code (Z-order curve) sorting
 * for cache-coherent spatial queries. Achieves O(n log n) construction
 * and O(1) neighborhood queries.
 *
 * Algorithm:
 * 1. Compute grid cell for each agent (parallel)
 * 2. Compute Morton code from cell coordinates
 * 3. Sort agents by Morton code (GPU radix sort)
 * 4. Build cell start/end boundaries (parallel scan)
 * 5. Query 3×3 neighborhood in constant time
 *
 * Design Pattern: Strategy for spatial queries
 */
class SpatialIndex {
public:
    SpatialIndex() = default;
    explicit SpatialIndex(const BoundingBox2D& bounds, float cell_size);

    /**
     * @brief Initialize spatial index
     * @param bounds Spatial domain bounds
     * @param cell_size Size of each grid cell (typically 2x interaction radius)
     */
    void initialize(const BoundingBox2D& bounds, float cell_size);

    /**
     * @brief Rebuild index from current agent positions
     * @param agents Agent data containing positions
     *
     * This must be called each step before spatial queries
     */
    void rebuild(core::AgentArrays& agents);

    /**
     * @brief Query neighbors within radius (GPU kernel)
     * @param query_position Position to query from
     * @param radius Search radius
     * @param agents Agent data
     * @param max_neighbors Maximum neighbors to return
     * @param neighbor_ids Output array for neighbor IDs (device pointer)
     * @param neighbor_count Output count of neighbors found
     */
    void query_neighbors_gpu(
        const float2* query_positions,
        float radius,
        const core::AgentArrays& agents,
        size_t num_queries,
        AgentID* neighbor_ids,
        uint32_t* neighbor_counts,
        uint32_t max_neighbors_per_query);

    /**
     * @brief Get all agents within a cell
     * @return Range [start, end) of agent indices in cell
     */
    std::pair<uint32_t, uint32_t> get_cell_range(CellID cell_id) const;

    /**
     * @brief Get grid cell ID for a position
     */
    CellID position_to_cell(const float2& position) const;

    /**
     * @brief Get cell bounds in world space
     */
    BoundingBox2D get_cell_bounds(CellID cell_id) const;

    // Statistics and diagnostics
    struct Statistics {
        float rebuild_time_ms;
        float sort_time_ms;
        uint32_t num_cells;
        uint32_t num_occupied_cells;
        float avg_agents_per_cell;
        float max_agents_per_cell;
    };

    const Statistics& statistics() const { return stats_; }

    // Configuration
    const BoundingBox2D& bounds() const { return bounds_; }
    float cell_size() const { return cell_size_; }
    uint32_t grid_width() const { return grid_width_; }
    uint32_t grid_height() const { return grid_height_; }
    uint32_t num_cells() const { return grid_width_ * grid_height_; }

private:
    // Grid configuration
    BoundingBox2D bounds_;
    float cell_size_ = 10.0f;
    uint32_t grid_width_ = 0;
    uint32_t grid_height_ = 0;

    // Cell boundary arrays
    gpu::DeviceBuffer<uint32_t> cell_starts_;
    gpu::DeviceBuffer<uint32_t> cell_ends_;

    // Temporary buffers for sorting
    gpu::DeviceBuffer<MortonCode> morton_codes_temp_;
    gpu::DeviceBuffer<AgentID> agent_ids_temp_;

    // Statistics
    Statistics stats_;

    // CUDA streams for parallel work
    std::unique_ptr<Stream> compute_stream_;
};

/**
 * @brief Hierarchical spatial index using octree/quadtree
 *
 * Alternative to uniform grid for non-uniform distributions.
 * Adaptive subdivision based on agent density.
 *
 * Note: Planned for v1.5
 */
class HierarchicalSpatialIndex {
public:
    // Future implementation
};

} // namespace gpu
} // namespace artemis
