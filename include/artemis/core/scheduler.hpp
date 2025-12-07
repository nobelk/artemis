#pragma once

#include "../util/types.hpp"
#include <functional>
#include <vector>
#include <string>

namespace artemis {
namespace core {

/**
 * @brief Scheduler types for agent updates
 */
enum class SchedulerType {
    SYNCHRONOUS,    // All agents update simultaneously (default)
    RANDOM,         // Random activation order
    STAGED          // Update in predefined stages
};

/**
 * @brief Manages temporal aspects of simulation
 *
 * Responsibilities:
 * - Controls simulation stepping
 * - Manages time progression
 * - Schedules events and callbacks
 * - Handles update ordering
 *
 * Design Pattern: Command pattern for scheduled actions
 */
class Scheduler {
public:
    using EventCallback = std::function<void(TimeStep)>;

    Scheduler() = default;
    explicit Scheduler(SchedulerType type, float dt);

    /**
     * @brief Initialize scheduler
     * @param type Type of scheduling strategy
     * @param dt Time step size
     */
    void initialize(SchedulerType type, float dt);

    /**
     * @brief Advance simulation by one time step
     * @return Current time step after advancement
     */
    TimeStep step();

    /**
     * @brief Reset scheduler to initial state
     */
    void reset();

    /**
     * @brief Register a recurring event callback
     * @param interval Steps between callback invocations
     * @param callback Function to call
     * @return Event ID for later removal
     */
    uint32_t register_event(uint32_t interval, EventCallback callback);

    /**
     * @brief Unregister an event
     */
    void unregister_event(uint32_t event_id);

    /**
     * @brief Process all scheduled events for current step
     */
    void process_events();

    // Getters
    TimeStep current_step() const { return current_step_; }
    float current_time() const { return current_step_ * dt_; }
    float delta_time() const { return dt_; }
    SchedulerType type() const { return type_; }

    // Setters
    void set_delta_time(float dt) { dt_ = dt; }
    void set_type(SchedulerType type) { type_ = type; }

private:
    struct ScheduledEvent {
        uint32_t id;
        uint32_t interval;
        TimeStep next_trigger;
        EventCallback callback;
    };

    SchedulerType type_ = SchedulerType::SYNCHRONOUS;
    TimeStep current_step_ = 0;
    float dt_ = 0.1f;

    std::vector<ScheduledEvent> events_;
    uint32_t next_event_id_ = 0;
};

} // namespace core
} // namespace artemis
