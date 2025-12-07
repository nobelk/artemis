#pragma once

#include "../util/types.hpp"
#include "../core/agent_arrays.hpp"
#include "../gpu/spatial_index.hpp"
#include <memory>
#include <string>

namespace artemis {
namespace behaviors {

/**
 * @brief Abstract base class for agent behaviors
 *
 * Implements Strategy pattern for pluggable agent behaviors.
 * Each behavior defines how agents update their velocities/states
 * based on local information and neighbors.
 *
 * Design Pattern: Strategy + Template Method
 */
class Behavior {
public:
    virtual ~Behavior() = default;

    /**
     * @brief Execute behavior for all agents
     *
     * @param agents Agent data (positions, velocities, etc.)
     * @param spatial_index Spatial index for neighbor queries
     * @param dt Time step size
     *
     * This method is called each simulation step and should update
     * agent velocities (or other state) on the GPU.
     */
    virtual void execute(
        core::AgentArrays& agents,
        gpu::SpatialIndex& spatial_index,
        float dt) = 0;

    /**
     * @brief Get behavior name/identifier
     */
    virtual std::string name() const = 0;

    /**
     * @brief Initialize behavior with configuration
     */
    virtual void initialize(const void* config) = 0;

    /**
     * @brief Check if behavior is GPU-accelerated
     */
    virtual bool is_gpu_accelerated() const { return true; }

    /**
     * @brief Get behavior-specific parameters
     */
    virtual void get_parameters(std::map<std::string, float>& params) const = 0;

    /**
     * @brief Set behavior-specific parameters
     */
    virtual void set_parameters(const std::map<std::string, float>& params) = 0;

protected:
    // Helper to allocate temporary GPU buffers
    template<typename T>
    gpu::DeviceBuffer<T>& get_temp_buffer(const std::string& name);

private:
    // Cache for temporary GPU buffers (avoid reallocation)
    std::map<std::string, std::shared_ptr<void>> temp_buffers_;
};

/**
 * @brief Factory for creating behaviors
 *
 * Design Pattern: Abstract Factory
 */
class BehaviorFactory {
public:
    using CreateFunc = std::function<std::unique_ptr<Behavior>()>;

    static BehaviorFactory& instance();

    /**
     * @brief Register a behavior type
     */
    void register_behavior(const std::string& name, CreateFunc creator);

    /**
     * @brief Create behavior by name
     */
    std::unique_ptr<Behavior> create(const std::string& name) const;

    /**
     * @brief Get list of registered behaviors
     */
    std::vector<std::string> available_behaviors() const;

private:
    BehaviorFactory() = default;
    std::map<std::string, CreateFunc> creators_;
};

/**
 * @brief Helper macro for registering behaviors
 */
#define REGISTER_BEHAVIOR(BehaviorClass, name) \
    namespace { \
        struct BehaviorClass##Registrar { \
            BehaviorClass##Registrar() { \
                ::artemis::behaviors::BehaviorFactory::instance().register_behavior( \
                    name, \
                    []() -> std::unique_ptr<::artemis::behaviors::Behavior> { \
                        return std::make_unique<BehaviorClass>(); \
                    } \
                ); \
            } \
        }; \
        static BehaviorClass##Registrar global_##BehaviorClass##_registrar; \
    }

} // namespace behaviors
} // namespace artemis
