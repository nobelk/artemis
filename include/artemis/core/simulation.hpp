#pragma once

#include "agent_arrays.hpp"
#include "environment.hpp"
#include "scheduler.hpp"
#include "../config/simulation_config.hpp"
#include "../behaviors/behavior_manager.hpp"
#include "../gpu/spatial_index.hpp"
#include "../analysis/metrics_engine.hpp"
#include <memory>

namespace artemis {
namespace core {

/**
 * @brief Main simulation orchestrator
 *
 * The Simulation class is the primary interface for running multiagent
 * simulations. It coordinates all subsystems:
 * - Agent management
 * - Environment
 * - Behaviors
 * - Spatial indexing
 * - Metrics collection
 * - Scheduling
 *
 * Design Pattern: Facade + Mediator
 */
class Simulation {
public:
    Simulation() = default;
    ~Simulation() = default;

    // Prevent copying (contains unique resources)
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    /**
     * @brief Initialize simulation from configuration
     * @param config Configuration object
     */
    void initialize(const config::SimulationConfig& config);

    /**
     * @brief Initialize simulation programmatically
     */
    void initialize(
        size_t num_agents,
        const BoundingBox2D& bounds,
        TopologyType topology = TopologyType::TORUS);

    /**
     * @brief Execute a single simulation step
     *
     * Pipeline:
     * 1. Update spatial index
     * 2. Execute agent behaviors
     * 3. Apply physics/integration
     * 4. Handle collisions
     * 5. Apply boundary conditions
     * 6. Compute metrics
     * 7. Process scheduled events
     */
    void step();

    /**
     * @brief Run simulation for specified number of steps
     */
    void run(size_t num_steps);

    /**
     * @brief Run simulation until condition is met
     */
    void run_until(std::function<bool()> stop_condition);

    /**
     * @brief Pause the simulation
     */
    void pause();

    /**
     * @brief Resume paused simulation
     */
    void resume();

    /**
     * @brief Reset simulation to initial state
     */
    void reset();

    /**
     * @brief Save simulation state to checkpoint
     */
    void save_checkpoint(const std::string& filepath);

    /**
     * @brief Load simulation state from checkpoint
     */
    void load_checkpoint(const std::string& filepath);

    // Component accessors
    AgentArrays& agents() { return agents_; }
    const AgentArrays& agents() const { return agents_; }

    Environment& environment() { return environment_; }
    const Environment& environment() const { return environment_; }

    Scheduler& scheduler() { return scheduler_; }
    const Scheduler& scheduler() const { return scheduler_; }

    behaviors::BehaviorManager& behavior_manager() { return *behavior_manager_; }
    const behaviors::BehaviorManager& behavior_manager() const { return *behavior_manager_; }

    gpu::SpatialIndex& spatial_index() { return *spatial_index_; }
    const gpu::SpatialIndex& spatial_index() const { return *spatial_index_; }

    analysis::MetricsEngine& metrics() { return *metrics_engine_; }
    const analysis::MetricsEngine& metrics() const { return *metrics_engine_; }

    // State queries
    SimulationState state() const { return state_; }
    TimeStep current_step() const { return scheduler_.current_step(); }
    float current_time() const { return scheduler_.current_time(); }
    size_t num_agents() const { return agents_.size(); }

private:
    // Core components
    AgentArrays agents_;
    Environment environment_;
    Scheduler scheduler_;

    // Subsystems (heap-allocated for flexibility)
    std::unique_ptr<behaviors::BehaviorManager> behavior_manager_;
    std::unique_ptr<gpu::SpatialIndex> spatial_index_;
    std::unique_ptr<analysis::MetricsEngine> metrics_engine_;

    SimulationState state_ = SimulationState::UNINITIALIZED;

    // Performance tracking
    float last_step_time_ms_ = 0.0f;
};

} // namespace core
} // namespace artemis
