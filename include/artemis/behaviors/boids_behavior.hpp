#pragma once

#include "behavior.hpp"

namespace artemis {
namespace behaviors {

/**
 * @brief Boids flocking behavior (Reynolds 1987)
 *
 * Implements three classic flocking rules:
 * 1. Separation: Avoid crowding neighbors
 * 2. Alignment: Steer towards average heading of neighbors
 * 3. Cohesion: Steer towards average position of neighbors
 *
 * All computations are GPU-accelerated.
 */
class BoidsBehavior : public Behavior {
public:
    struct Parameters {
        float separation_radius = 5.0f;
        float alignment_radius = 15.0f;
        float cohesion_radius = 15.0f;

        float separation_weight = 1.5f;
        float alignment_weight = 1.0f;
        float cohesion_weight = 1.0f;

        float max_force = 0.5f;
        float max_speed = 5.0f;
    };

    BoidsBehavior() = default;
    explicit BoidsBehavior(const Parameters& params);

    void execute(
        core::AgentArrays& agents,
        gpu::SpatialIndex& spatial_index,
        float dt) override;

    std::string name() const override { return "boids"; }

    void initialize(const void* config) override;

    void get_parameters(std::map<std::string, float>& params) const override;
    void set_parameters(const std::map<std::string, float>& params) override;

    const Parameters& parameters() const { return params_; }
    void set_parameters(const Parameters& params) { params_ = params; }

private:
    Parameters params_;

    // GPU buffers for intermediate forces
    gpu::DeviceBuffer<float2> separation_forces_;
    gpu::DeviceBuffer<float2> alignment_forces_;
    gpu::DeviceBuffer<float2> cohesion_forces_;
};

} // namespace behaviors
} // namespace artemis
