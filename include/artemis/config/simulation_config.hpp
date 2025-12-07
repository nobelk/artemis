#pragma once

#include "../util/types.hpp"
#include <string>
#include <vector>
#include <map>
#include <variant>

namespace artemis {
namespace config {

/**
 * @brief Type-safe configuration value
 */
using ConfigValue = std::variant<
    bool,
    int64_t,
    double,
    std::string,
    std::vector<double>,
    std::vector<std::string>
>;

/**
 * @brief Agent type configuration
 */
struct AgentTypeConfig {
    std::string name;
    uint32_t count;

    // Physical properties
    float max_speed = 5.0f;
    float perception_radius = 15.0f;
    float collision_radius = 1.0f;
    float mass = 1.0f;

    // Behavior weights (interpretation depends on behavior type)
    std::map<std::string, float> behavior_weights;

    // Custom properties
    std::map<std::string, ConfigValue> custom_properties;
};

/**
 * @brief Environment configuration
 */
struct EnvironmentConfig {
    std::string type = "grid_2d";  // grid_2d, grid_3d, continuous
    std::vector<float> dimensions = {1000.0f, 1000.0f};
    std::string topology = "torus";  // bounded, torus, infinite
    float cell_size = 50.0f;

    // Environment-specific parameters
    std::map<std::string, ConfigValue> parameters;
};

/**
 * @brief Temporal configuration
 */
struct TemporalConfig {
    uint64_t max_steps = 10000;
    float dt = 0.1f;
    std::string scheduler = "synchronous";  // synchronous, random, staged

    // Event scheduling
    struct EventConfig {
        std::string name;
        uint32_t interval;
        std::string callback;
    };
    std::vector<EventConfig> events;
};

/**
 * @brief GPU configuration
 */
struct GPUConfig {
    int device_id = 0;
    uint32_t threads_per_block = 256;
    bool use_streams = true;
    size_t max_memory_mb = 0;  // 0 = auto

    // Performance tuning
    bool enable_profiling = false;
    bool enable_peer_access = false;  // Multi-GPU
};

/**
 * @brief Analysis configuration
 */
struct AnalysisConfig {
    std::vector<std::string> metrics = {
        "polarization",
        "clustering_coefficient",
        "neighbor_distribution"
    };

    uint32_t frequency = 10;  // Compute every N steps
    bool export_timeseries = true;
    std::string output_format = "hdf5";  // hdf5, csv, json

    // Pattern detection
    struct PatternConfig {
        std::string type;
        std::map<std::string, ConfigValue> parameters;
    };
    std::vector<PatternConfig> patterns;
};

/**
 * @brief Behavior configuration
 */
struct BehaviorConfig {
    std::string type;  // boids, predator_prey, social, custom
    std::map<std::string, ConfigValue> parameters;
};

/**
 * @brief I/O configuration
 */
struct IOConfig {
    std::string output_directory = "./output";
    bool enable_checkpointing = false;
    uint32_t checkpoint_interval = 1000;
    bool compress_output = true;
    uint32_t compression_level = 6;
};

/**
 * @brief Visualization configuration
 */
struct VisualizationConfig {
    bool enabled = false;
    uint32_t window_width = 1920;
    uint32_t window_height = 1080;
    uint32_t fps_limit = 60;
    std::string renderer = "opengl";  // opengl, vulkan, headless

    // Visual settings
    float point_size = 2.0f;
    bool show_velocity_vectors = false;
    bool show_spatial_grid = false;
    std::string colormap = "agent_type";  // agent_type, velocity, energy
};

/**
 * @brief Complete simulation configuration
 *
 * This is the top-level configuration object that contains all
 * simulation parameters. Can be loaded from YAML/JSON or constructed
 * programmatically.
 *
 * Design Pattern: Builder pattern for configuration construction
 */
struct SimulationConfig {
    // Metadata
    std::string name = "unnamed_simulation";
    std::string description = "";
    uint64_t random_seed = 42;

    // Component configurations
    std::vector<AgentTypeConfig> agent_types;
    EnvironmentConfig environment;
    TemporalConfig temporal;
    GPUConfig gpu;
    AnalysisConfig analysis;
    std::vector<BehaviorConfig> behaviors;
    IOConfig io;
    VisualizationConfig visualization;

    /**
     * @brief Validate configuration for consistency
     * @throws std::runtime_error if configuration is invalid
     */
    void validate() const;

    /**
     * @brief Get total number of agents across all types
     */
    size_t total_agent_count() const;

    /**
     * @brief Print configuration summary
     */
    void print_summary() const;
};

/**
 * @brief Builder for SimulationConfig
 *
 * Provides fluent interface for constructing configurations
 */
class ConfigBuilder {
public:
    ConfigBuilder() = default;

    ConfigBuilder& set_name(const std::string& name);
    ConfigBuilder& set_description(const std::string& desc);
    ConfigBuilder& set_seed(uint64_t seed);

    ConfigBuilder& add_agent_type(const AgentTypeConfig& agent_type);
    ConfigBuilder& set_environment(const EnvironmentConfig& env);
    ConfigBuilder& set_temporal(const TemporalConfig& temporal);
    ConfigBuilder& set_gpu(const GPUConfig& gpu);
    ConfigBuilder& set_analysis(const AnalysisConfig& analysis);
    ConfigBuilder& add_behavior(const BehaviorConfig& behavior);
    ConfigBuilder& set_io(const IOConfig& io);
    ConfigBuilder& set_visualization(const VisualizationConfig& viz);

    SimulationConfig build() const;

private:
    SimulationConfig config_;
};

} // namespace config
} // namespace artemis
