#pragma once

#include "../util/types.hpp"
#include "agent_arrays.hpp"
#include <memory>

namespace artemis {
namespace core {

/**
 * @brief Manages the spatial environment and agent interactions
 *
 * The Environment class is responsible for:
 * - Maintaining spatial bounds
 * - Managing topology (bounded, torus, infinite)
 * - Handling boundary conditions
 * - Providing spatial queries
 *
 * Design Pattern: Facade for spatial operations
 */
class Environment {
public:
    Environment() = default;
    explicit Environment(const BoundingBox2D& bounds, TopologyType topology);

    /**
     * @brief Initialize environment with configuration
     */
    void initialize(const BoundingBox2D& bounds, TopologyType topology);

    /**
     * @brief Apply boundary conditions to agent positions
     * @param agents Agent data to update
     */
    void apply_boundary_conditions(AgentArrays& agents);

    /**
     * @brief Wrap position according to topology
     */
    float2 wrap_position(const float2& position) const;

    /**
     * @brief Calculate wrapped displacement vector between two positions
     * @return Shortest displacement considering topology
     */
    float2 displacement(const float2& from, const float2& to) const;

    /**
     * @brief Get distance between two positions (considering topology)
     */
    float distance(const float2& a, const float2& b) const;

    /**
     * @brief Check if position is within valid bounds
     */
    bool is_valid_position(const float2& position) const;

    // Getters
    const BoundingBox2D& bounds() const { return bounds_; }
    TopologyType topology() const { return topology_; }
    float2 size() const { return bounds_.size(); }
    float2 center() const { return bounds_.center(); }

    // Setters
    void set_bounds(const BoundingBox2D& bounds) { bounds_ = bounds; }
    void set_topology(TopologyType topology) { topology_ = topology; }

private:
    BoundingBox2D bounds_;
    TopologyType topology_ = TopologyType::BOUNDED;
};

} // namespace core
} // namespace artemis
