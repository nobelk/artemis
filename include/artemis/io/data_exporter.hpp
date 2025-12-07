#pragma once

#include "../core/agent_arrays.hpp"
#include "../analysis/metrics_engine.hpp"
#include <string>
#include <fstream>

namespace artemis {
namespace io {

/**
 * @brief Export format options
 */
enum class ExportFormat {
    CSV,
    JSON,
    BINARY
};

/**
 * @brief CSV file writer for metrics and agent data
 *
 * Exports data in comma-separated format for easy analysis
 * in Excel, Python, R, etc.
 */
class CSVExporter {
public:
    CSVExporter() = default;

    /**
     * @brief Export metrics to CSV file
     * @param filepath Output file path
     * @param metrics Metrics time series data
     */
    void export_metrics(
        const std::string& filepath,
        const std::map<std::string, analysis::MetricTimeSeries>& metrics);

    /**
     * @brief Export agent positions to CSV
     * @param filepath Output file path
     * @param agents Agent data
     * @param timestep Timestep label (optional)
     */
    void export_agent_positions(
        const std::string& filepath,
        const core::AgentArrays& agents,
        TimeStep timestep = 0);

    /**
     * @brief Export agent velocities to CSV
     */
    void export_agent_velocities(
        const std::string& filepath,
        const core::AgentArrays& agents,
        TimeStep timestep = 0);

    /**
     * @brief Export complete agent state to CSV
     */
    void export_agent_state(
        const std::string& filepath,
        const core::AgentArrays& agents,
        TimeStep timestep = 0);

    /**
     * @brief Set delimiter character (default: comma)
     */
    void set_delimiter(char delimiter) { delimiter_ = delimiter; }

    /**
     * @brief Enable/disable header row
     */
    void set_write_header(bool write) { write_header_ = write; }

private:
    char delimiter_ = ',';
    bool write_header_ = true;
};

/**
 * @brief JSON file writer for structured data export
 */
class JSONExporter {
public:
    JSONExporter() = default;

    /**
     * @brief Export metrics to JSON file
     */
    void export_metrics(
        const std::string& filepath,
        const std::map<std::string, analysis::MetricTimeSeries>& metrics,
        bool pretty_print = true);

    /**
     * @brief Export agent state to JSON
     */
    void export_agent_state(
        const std::string& filepath,
        const core::AgentArrays& agents,
        TimeStep timestep = 0,
        bool pretty_print = true);

    /**
     * @brief Export simulation summary statistics
     */
    void export_summary(
        const std::string& filepath,
        const std::map<std::string, std::string>& summary,
        bool pretty_print = true);

    /**
     * @brief Set indentation for pretty printing
     */
    void set_indent(int spaces) { indent_ = spaces; }

private:
    int indent_ = 2;
};

/**
 * @brief Binary exporter for compact storage
 *
 * Uses custom binary format for efficient storage of large datasets
 */
class BinaryExporter {
public:
    BinaryExporter() = default;

    /**
     * @brief Export agent trajectory to binary file
     */
    void export_trajectory(
        const std::string& filepath,
        const std::vector<std::vector<float2>>& positions,
        const std::vector<std::vector<float2>>& velocities);

    /**
     * @brief Load trajectory from binary file
     */
    void load_trajectory(
        const std::string& filepath,
        std::vector<std::vector<float2>>& positions,
        std::vector<std::vector<float2>>& velocities);

private:
    static constexpr uint32_t MAGIC_NUMBER = 0x41525445;  // "ARTE"
    static constexpr uint32_t VERSION = 1;
};

/**
 * @brief Unified data exporter with format auto-detection
 *
 * Design Pattern: Factory + Strategy
 */
class DataExporter {
public:
    DataExporter() = default;

    /**
     * @brief Export metrics with format auto-detected from extension
     * @param filepath Output file path (.csv, .json, .h5)
     * @param metrics Metrics to export
     */
    void export_metrics(
        const std::string& filepath,
        const std::map<std::string, analysis::MetricTimeSeries>& metrics);

    /**
     * @brief Export agent state with format auto-detection
     */
    void export_agents(
        const std::string& filepath,
        const core::AgentArrays& agents,
        TimeStep timestep = 0);

    /**
     * @brief Set preferred export format (overrides auto-detection)
     */
    void set_format(ExportFormat format) { format_ = format; }

private:
    ExportFormat format_ = ExportFormat::CSV;

    ExportFormat detect_format(const std::string& filepath) const;
};

} // namespace io
} // namespace artemis
