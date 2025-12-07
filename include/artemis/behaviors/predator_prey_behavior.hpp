#pragma once

#include "behavior.hpp"

namespace artemis {
namespace behaviors {

/**
 * @brief Predator-prey interaction behavior
 *
 * Implements classic predator-prey dynamics:
 * - Prey flee from nearby predators
 * - Predators chase nearby prey
 * - Energy dynamics (hunting, eating, starvation)
 * - Reproduction when energy threshold reached
 */
class PredatorPreyBehavior : public Behavior {
public:
    struct Parameters {
        // Perception radii
        float predator_perception_radius = 30.0f;
        float prey_perception_radius = 40.0f;

        // Movement parameters
        float predator_max_speed = 6.0f;
        float prey_max_speed = 5.0f;
        float flee_force = 2.0f;
        float chase_force = 1.5f;

        // Energy dynamics
        float predator_energy_decay = 0.1f;
        float prey_energy_decay = 0.05f;
        float eat_energy_gain = 50.0f;
        float reproduction_threshold = 100.0f;
        float reproduction_cost = 40.0f;

        // Interaction
        float kill_distance = 2.0f;
    };

    PredatorPreyBehavior() = default;
    explicit PredatorPreyBehavior(const Parameters& params);

    void execute(
        core::AgentArrays& agents,
        gpu::SpatialIndex& spatial_index,
        float dt) override;

    std::string name() const override { return "predator_prey"; }

    void initialize(const void* config) override;

    void get_parameters(std::map<std::string, float>& params) const override;
    void set_parameters(const std::map<std::string, float>& params) override;

    const Parameters& parameters() const { return params_; }
    void set_parameters(const Parameters& params) { params_ = params; }

    // Statistics
    struct Statistics {
        uint32_t num_predators;
        uint32_t num_prey;
        uint32_t num_kills;
        uint32_t num_births;
        float avg_predator_energy;
        float avg_prey_energy;
    };

    Statistics get_statistics(const core::AgentArrays& agents) const;

private:
    Parameters params_;

    // GPU buffers for tracking events
    gpu::DeviceBuffer<uint32_t> kill_events_;
    gpu::DeviceBuffer<uint32_t> birth_events_;
};

} // namespace behaviors
} // namespace artemis
