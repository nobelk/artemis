#pragma once

#include "behavior.hpp"
#include "../core/agent_arrays.hpp"
#include "../gpu/spatial_index.hpp"
#include <vector>
#include <memory>

namespace artemis {
namespace behaviors {

/**
 * @brief Manages and executes multiple behaviors
 *
 * Coordinates execution of multiple behaviors for different agent types.
 * Behaviors can be composed and executed in sequence.
 *
 * Design Pattern: Composite + Chain of Responsibility
 */
class BehaviorManager {
public:
    BehaviorManager() = default;

    /**
     * @brief Add a behavior to the execution chain
     * @param behavior Behavior to add (takes ownership)
     * @param priority Execution priority (lower = earlier)
     */
    void add_behavior(std::unique_ptr<Behavior> behavior, int priority = 0);

    /**
     * @brief Remove behavior by name
     */
    void remove_behavior(const std::string& name);

    /**
     * @brief Execute all behaviors in priority order
     */
    void execute_all(
        core::AgentArrays& agents,
        gpu::SpatialIndex& spatial_index,
        float dt);

    /**
     * @brief Get behavior by name
     */
    Behavior* get_behavior(const std::string& name);
    const Behavior* get_behavior(const std::string& name) const;

    /**
     * @brief Check if behavior exists
     */
    bool has_behavior(const std::string& name) const;

    /**
     * @brief Get list of all behavior names
     */
    std::vector<std::string> behavior_names() const;

    /**
     * @brief Clear all behaviors
     */
    void clear();

    /**
     * @brief Get number of active behaviors
     */
    size_t num_behaviors() const { return behaviors_.size(); }

    /**
     * @brief Enable/disable a specific behavior
     */
    void enable_behavior(const std::string& name, bool enabled);

    /**
     * @brief Check if behavior is enabled
     */
    bool is_behavior_enabled(const std::string& name) const;

private:
    struct BehaviorEntry {
        std::unique_ptr<Behavior> behavior;
        int priority;
        bool enabled;

        bool operator<(const BehaviorEntry& other) const {
            return priority < other.priority;
        }
    };

    std::vector<BehaviorEntry> behaviors_;

    void sort_behaviors();
};

/**
 * @brief Composite behavior that combines multiple sub-behaviors
 *
 * Allows creating complex behaviors by composing simpler ones
 */
class CompositeBehavior : public Behavior {
public:
    CompositeBehavior() = default;

    void add_sub_behavior(std::unique_ptr<Behavior> behavior, float weight = 1.0f);

    void execute(
        core::AgentArrays& agents,
        gpu::SpatialIndex& spatial_index,
        float dt) override;

    std::string name() const override { return "composite"; }

    void initialize(const void* config) override;

    void get_parameters(std::map<std::string, float>& params) const override;
    void set_parameters(const std::map<std::string, float>& params) override;

private:
    struct SubBehavior {
        std::unique_ptr<Behavior> behavior;
        float weight;
    };

    std::vector<SubBehavior> sub_behaviors_;
};

} // namespace behaviors
} // namespace artemis
