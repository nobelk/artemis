#pragma once

#include "../util/types.hpp"
#include "../core/agent_arrays.hpp"
#include "../analysis/metrics_engine.hpp"
#include <string>
#include <vector>
#include <map>

#ifdef ARTEMIS_HDF5_ENABLED
// Forward declare HDF5 types to avoid including hdf5.h in header
using hid_t = long long;
#endif

namespace artemis {
namespace io {

/**
 * @brief HDF5 file writer for simulation data export
 *
 * Exports simulation data to HDF5 format for analysis:
 * - Agent trajectories (time series of positions/velocities)
 * - Metrics time series
 * - Simulation metadata
 * - Configuration
 *
 * HDF5 File Structure:
 * /
 * ├── metadata/
 * │   ├── simulation_name
 * │   ├── timesteps
 * │   └── num_agents
 * ├── config/
 * │   └── [configuration YAML as string]
 * ├── agents/
 * │   ├── positions (N × T × 2 dataset)
 * │   ├── velocities (N × T × 2 dataset)
 * │   ├── types (N × T dataset)
 * │   └── energies (N × T dataset)
 * └── metrics/
 *     ├── polarization (T dataset)
 *     ├── clustering_coefficient (T dataset)
 *     └── [other metrics...]
 *
 * Design Pattern: Facade for HDF5 API
 */
class HDF5Writer {
public:
    HDF5Writer() = default;
    ~HDF5Writer();

    // Prevent copying
    HDF5Writer(const HDF5Writer&) = delete;
    HDF5Writer& operator=(const HDF5Writer&) = delete;

    /**
     * @brief Open HDF5 file for writing
     * @param filepath Output file path
     * @param overwrite If true, overwrite existing file
     */
    void open(const std::string& filepath, bool overwrite = false);

    /**
     * @brief Close HDF5 file
     */
    void close();

    /**
     * @brief Check if file is currently open
     */
    bool is_open() const;

    /**
     * @brief Write simulation metadata
     */
    void write_metadata(
        const std::string& simulation_name,
        size_t num_agents,
        size_t num_timesteps,
        float dt);

    /**
     * @brief Write configuration as YAML string
     */
    void write_config(const std::string& yaml_config);

    /**
     * @brief Initialize agent datasets (must be called before writing frames)
     * @param num_agents Number of agents
     * @param num_timesteps Expected total timesteps (for pre-allocation)
     */
    void initialize_agent_datasets(size_t num_agents, size_t num_timesteps);

    /**
     * @brief Write agent state for a single timestep
     * @param agents Agent data
     * @param timestep Timestep index
     */
    void write_agent_frame(
        const core::AgentArrays& agents,
        TimeStep timestep);

    /**
     * @brief Write complete agent trajectory (all timesteps)
     */
    void write_agent_trajectory(
        const std::vector<std::vector<float2>>& positions,
        const std::vector<std::vector<float2>>& velocities);

    /**
     * @brief Write metrics time series
     */
    void write_metrics(
        const std::map<std::string, analysis::MetricTimeSeries>& metrics);

    /**
     * @brief Write a single metric dataset
     */
    void write_metric_dataset(
        const std::string& name,
        const std::vector<float>& values);

    /**
     * @brief Set compression level (0-9)
     */
    void set_compression(int level);

    /**
     * @brief Set chunk size for datasets (affects I/O performance)
     */
    void set_chunk_size(size_t chunk_size);

    /**
     * @brief Flush data to disk
     */
    void flush();

private:
#ifdef ARTEMIS_HDF5_ENABLED
    hid_t file_id_ = -1;
    hid_t agents_group_ = -1;
    hid_t metrics_group_ = -1;
    hid_t metadata_group_ = -1;

    // Dataset handles
    hid_t positions_dataset_ = -1;
    hid_t velocities_dataset_ = -1;
    hid_t types_dataset_ = -1;
    hid_t energies_dataset_ = -1;

    int compression_level_ = 6;
    size_t chunk_size_ = 1024;

    void create_groups();
    hid_t create_dataset_2d(const std::string& name, size_t dim1, size_t dim2);
    hid_t create_dataset_3d(const std::string& name, size_t dim1, size_t dim2, size_t dim3);
#endif
};

/**
 * @brief HDF5 file reader for loading simulation data
 */
class HDF5Reader {
public:
    HDF5Reader() = default;
    ~HDF5Reader();

    // Prevent copying
    HDF5Reader(const HDF5Reader&) = delete;
    HDF5Reader& operator=(const HDF5Reader&) = delete;

    /**
     * @brief Open HDF5 file for reading
     */
    void open(const std::string& filepath);

    /**
     * @brief Close HDF5 file
     */
    void close();

    /**
     * @brief Check if file is currently open
     */
    bool is_open() const;

    /**
     * @brief Read simulation metadata
     */
    struct Metadata {
        std::string simulation_name;
        size_t num_agents;
        size_t num_timesteps;
        float dt;
    };
    Metadata read_metadata();

    /**
     * @brief Read configuration YAML
     */
    std::string read_config();

    /**
     * @brief Read agent positions for all timesteps
     */
    std::vector<std::vector<float2>> read_positions();

    /**
     * @brief Read agent positions for specific timestep
     */
    std::vector<float2> read_positions_at(TimeStep timestep);

    /**
     * @brief Read agent velocities for all timesteps
     */
    std::vector<std::vector<float2>> read_velocities();

    /**
     * @brief Read metric time series
     */
    std::vector<float> read_metric(const std::string& name);

    /**
     * @brief List available metrics in file
     */
    std::vector<std::string> list_metrics();

private:
#ifdef ARTEMIS_HDF5_ENABLED
    hid_t file_id_ = -1;
#endif
};

} // namespace io
} // namespace artemis
