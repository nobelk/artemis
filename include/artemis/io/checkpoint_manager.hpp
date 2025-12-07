#pragma once

#include "../core/simulation.hpp"
#include <string>
#include <vector>

namespace artemis {
namespace io {

/**
 * @brief Checkpoint metadata
 */
struct CheckpointMetadata {
    std::string filepath;
    TimeStep timestep;
    size_t num_agents;
    std::string simulation_name;
    uint64_t random_seed;
    std::string timestamp;  // ISO 8601 format
    size_t file_size_bytes;

    // Checksums for validation
    std::string checksum_md5;
    std::string checksum_sha256;
};

/**
 * @brief Manages simulation checkpointing and restoration
 *
 * Checkpoints include:
 * - Complete agent state (positions, velocities, types, energies, ages)
 * - Simulation metadata (timestep, seed, configuration)
 * - Analysis state (metrics history)
 * - RNG state for exact reproducibility
 *
 * Design Pattern: Memento for state persistence
 */
class CheckpointManager {
public:
    CheckpointManager() = default;

    /**
     * @brief Save simulation state to checkpoint file
     * @param simulation Simulation to checkpoint
     * @param filepath Output file path
     * @param compress Enable compression (default: true)
     */
    void save_checkpoint(
        const core::Simulation& simulation,
        const std::string& filepath,
        bool compress = true);

    /**
     * @brief Load simulation state from checkpoint
     * @param simulation Simulation to restore into
     * @param filepath Checkpoint file path
     */
    void load_checkpoint(
        core::Simulation& simulation,
        const std::string& filepath);

    /**
     * @brief Validate checkpoint file integrity
     * @return true if valid, false otherwise
     */
    bool validate_checkpoint(const std::string& filepath);

    /**
     * @brief Read checkpoint metadata without loading full state
     */
    CheckpointMetadata read_metadata(const std::string& filepath);

    /**
     * @brief List all checkpoints in directory
     * @param directory Directory to scan
     * @return Vector of checkpoint metadata
     */
    std::vector<CheckpointMetadata> list_checkpoints(
        const std::string& directory);

    /**
     * @brief Delete checkpoint file
     */
    void delete_checkpoint(const std::string& filepath);

    /**
     * @brief Get checkpoint file format version
     */
    static uint32_t format_version() { return 1; }

    /**
     * @brief Set compression level (0-9, higher = better compression)
     */
    void set_compression_level(int level) { compression_level_ = level; }

private:
    int compression_level_ = 6;

    void write_header(void* file, const CheckpointMetadata& metadata);
    CheckpointMetadata read_header(void* file);
    std::string compute_checksum(const std::string& filepath);
};

/**
 * @brief Automatic checkpointing manager
 *
 * Handles periodic checkpointing with rotation policy
 */
class AutoCheckpointer {
public:
    struct Config {
        std::string directory = "./checkpoints";
        uint32_t interval = 1000;        // Checkpoint every N steps
        uint32_t max_checkpoints = 10;   // Keep at most N checkpoints
        bool compress = true;
        bool timestamp_filenames = true;
    };

    AutoCheckpointer() = default;
    explicit AutoCheckpointer(const Config& config);

    /**
     * @brief Initialize auto-checkpointer
     */
    void initialize(const Config& config);

    /**
     * @brief Check if checkpoint should be saved this step
     * @param current_step Current timestep
     * @return true if checkpoint should be saved
     */
    bool should_checkpoint(TimeStep current_step) const;

    /**
     * @brief Save checkpoint with automatic naming
     * @param simulation Simulation to checkpoint
     * @param current_step Current timestep
     * @return Checkpoint filepath
     */
    std::string auto_checkpoint(
        const core::Simulation& simulation,
        TimeStep current_step);

    /**
     * @brief Clean up old checkpoints based on rotation policy
     */
    void cleanup_old_checkpoints();

    /**
     * @brief Get list of managed checkpoints
     */
    std::vector<CheckpointMetadata> managed_checkpoints() const;

    const Config& config() const { return config_; }

private:
    Config config_;
    CheckpointManager checkpoint_manager_;

    std::string generate_checkpoint_path(TimeStep timestep) const;
};

} // namespace io
} // namespace artemis
