#pragma once

#include "behavior.hpp"

namespace artemis {
namespace behaviors {

/**
 * @brief Social interaction behavior
 *
 * Models agent social dynamics:
 * - Opinion formation and influence
 * - Social network formation
 * - Group membership and clustering
 * - Information diffusion
 */
class SocialBehavior : public Behavior {
public:
    struct Parameters {
        // Interaction parameters
        float interaction_radius = 20.0f;
        float influence_strength = 0.1f;
        float homophily_bias = 0.5f;  // Preference for similar agents

        // Opinion dynamics
        float opinion_noise = 0.01f;
        float convergence_threshold = 0.05f;

        // Movement (social attraction/repulsion)
        float attraction_strength = 0.5f;
        float repulsion_strength = 0.3f;
        float personal_space = 5.0f;
    };

    SocialBehavior() = default;
    explicit SocialBehavior(const Parameters& params);

    void execute(
        core::AgentArrays& agents,
        gpu::SpatialIndex& spatial_index,
        float dt) override;

    std::string name() const override { return "social"; }

    void initialize(const void* config) override;

    void get_parameters(std::map<std::string, float>& params) const override;
    void set_parameters(const std::map<std::string, float>& params) override;

    const Parameters& parameters() const { return params_; }
    void set_parameters(const Parameters& params) { params_ = params; }

private:
    Parameters params_;

    // GPU buffers for social state
    gpu::DeviceBuffer<float> opinions_;       // Agent opinions/states
    gpu::DeviceBuffer<uint32_t> group_ids_;   // Social group membership
    gpu::DeviceBuffer<float> influence_sum_;  // Accumulated influence
};

} // namespace behaviors
} // namespace artemis
