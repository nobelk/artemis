#pragma once

#include "../util/types.hpp"
#include "../util/gpu_util.hpp"
#include <vector>

namespace artemis {
namespace core {

/**
 * @brief Structure of Arrays (SoA) for efficient GPU agent storage
 *
 * This class manages agent data in a GPU-friendly SoA layout for optimal
 * memory coalescing and cache performance. Each property is stored in a
 * separate contiguous array.
 *
 * Design Pattern: Data-Oriented Design (DoD)
 */
class AgentArrays {
public:
    AgentArrays() = default;
    explicit AgentArrays(size_t capacity);

    // Prevent copying (expensive operation)
    AgentArrays(const AgentArrays&) = delete;
    AgentArrays& operator=(const AgentArrays&) = delete;

    // Allow moving
    AgentArrays(AgentArrays&&) noexcept = default;
    AgentArrays& operator=(AgentArrays&&) noexcept = default;

    /**
     * @brief Allocate GPU memory for specified number of agents
     */
    void allocate(size_t count);

    /**
     * @brief Resize arrays to new capacity (may reallocate)
     */
    void resize(size_t new_count);

    /**
     * @brief Upload agent data from host to device
     */
    void upload_from_host(const std::vector<AgentState>& host_agents);

    /**
     * @brief Download agent data from device to host
     */
    void download_to_host(std::vector<AgentState>& host_agents) const;

    /**
     * @brief Initialize agents with random positions and velocities
     */
    void initialize_random(
        const BoundingBox2D& bounds,
        uint64_t seed = 42);

    /**
     * @brief Zero all arrays (useful for metrics/temporary arrays)
     */
    void zero_all();

    // Accessors (device pointers)
    float2* positions() { return positions_.data(); }
    const float2* positions() const { return positions_.data(); }

    float2* velocities() { return velocities_.data(); }
    const float2* velocities() const { return velocities_.data(); }

    uint8_t* types() { return types_.data(); }
    const uint8_t* types() const { return types_.data(); }

    float* energies() { return energies_.data(); }
    const float* energies() const { return energies_.data(); }

    uint32_t* ages() { return ages_.data(); }
    const uint32_t* ages() const { return ages_.data(); }

    MortonCode* morton_codes() { return morton_codes_.data(); }
    const MortonCode* morton_codes() const { return morton_codes_.data(); }

    AgentID* sorted_indices() { return sorted_indices_.data(); }
    const AgentID* sorted_indices() const { return sorted_indices_.data(); }

    // Metadata
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }

private:
    // Core agent properties (always allocated)
    gpu::DeviceBuffer<float2> positions_;
    gpu::DeviceBuffer<float2> velocities_;
    gpu::DeviceBuffer<uint8_t> types_;
    gpu::DeviceBuffer<float> energies_;
    gpu::DeviceBuffer<uint32_t> ages_;

    // Spatial indexing support
    gpu::DeviceBuffer<MortonCode> morton_codes_;
    gpu::DeviceBuffer<AgentID> sorted_indices_;

    size_t size_ = 0;
    size_t capacity_ = 0;
};

} // namespace core
} // namespace artemis
