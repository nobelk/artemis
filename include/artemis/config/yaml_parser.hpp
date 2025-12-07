#pragma once

#include "simulation_config.hpp"
#include <string>
#include <istream>

namespace artemis {
namespace config {

/**
 * @brief YAML configuration parser
 *
 * Parses YAML configuration files into SimulationConfig objects.
 * Provides validation and helpful error messages.
 *
 * Design Pattern: Parser + Factory
 */
class YAMLParser {
public:
    YAMLParser() = default;

    /**
     * @brief Load configuration from YAML file
     * @param filepath Path to YAML configuration file
     * @return Parsed configuration
     * @throws std::runtime_error if file not found or parsing fails
     */
    static SimulationConfig load_from_file(const std::string& filepath);

    /**
     * @brief Load configuration from YAML string
     * @param yaml_string YAML content as string
     * @return Parsed configuration
     * @throws std::runtime_error if parsing fails
     */
    static SimulationConfig load_from_string(const std::string& yaml_string);

    /**
     * @brief Load configuration from input stream
     * @param stream Input stream containing YAML
     * @return Parsed configuration
     */
    static SimulationConfig load_from_stream(std::istream& stream);

    /**
     * @brief Save configuration to YAML file
     * @param config Configuration to save
     * @param filepath Output file path
     */
    static void save_to_file(
        const SimulationConfig& config,
        const std::string& filepath);

    /**
     * @brief Convert configuration to YAML string
     * @param config Configuration to convert
     * @return YAML string representation
     */
    static std::string to_yaml_string(const SimulationConfig& config);

    /**
     * @brief Validate YAML file without parsing
     * @param filepath Path to YAML file
     * @return true if valid, false otherwise
     */
    static bool validate_file(const std::string& filepath);

    /**
     * @brief Get schema for configuration YAML
     * @return JSON schema string for validation
     */
    static std::string get_schema();

private:
    // Helper methods for parsing subsections
    static AgentTypeConfig parse_agent_type(const void* node);
    static EnvironmentConfig parse_environment(const void* node);
    static TemporalConfig parse_temporal(const void* node);
    static GPUConfig parse_gpu(const void* node);
    static AnalysisConfig parse_analysis(const void* node);
    static BehaviorConfig parse_behavior(const void* node);
    static IOConfig parse_io(const void* node);
    static VisualizationConfig parse_visualization(const void* node);

    // Helper for custom properties
    static ConfigValue parse_config_value(const void* node);
};

/**
 * @brief JSON configuration parser (alternative to YAML)
 *
 * Provides same functionality as YAMLParser but for JSON format
 */
class JSONParser {
public:
    static SimulationConfig load_from_file(const std::string& filepath);
    static SimulationConfig load_from_string(const std::string& json_string);
    static void save_to_file(
        const SimulationConfig& config,
        const std::string& filepath);
    static std::string to_json_string(
        const SimulationConfig& config,
        bool pretty_print = true);
};

} // namespace config
} // namespace artemis
