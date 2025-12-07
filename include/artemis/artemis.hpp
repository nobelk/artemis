#pragma once

/**
 * @file artemis.hpp
 * @brief Main Artemis library header - include this to use the full framework
 *
 * Artemis: GPU-Accelerated Multiagent Simulation Platform
 *
 * This is the main convenience header that includes all Artemis components.
 * For faster compilation, you can include only the specific headers you need.
 */

// Version information
#define ARTEMIS_VERSION_MAJOR 1
#define ARTEMIS_VERSION_MINOR 0
#define ARTEMIS_VERSION_PATCH 0

// Utility layer
#include "util/types.hpp"
#include "util/math.hpp"
#include "util/gpu_util.hpp"

// Core simulation
#include "core/agent_arrays.hpp"
#include "core/environment.hpp"
#include "core/scheduler.hpp"
#include "core/simulation.hpp"

// GPU computation layer
#include "gpu/device_manager.hpp"
#include "gpu/spatial_index.hpp"
#include "gpu/kernels.cuh"

// Configuration system
#include "config/simulation_config.hpp"
#include "config/yaml_parser.hpp"

// Behavior system
#include "behaviors/behavior.hpp"
#include "behaviors/behavior_manager.hpp"
#include "behaviors/boids_behavior.hpp"
#include "behaviors/predator_prey_behavior.hpp"
#include "behaviors/social_behavior.hpp"

// Analysis pipeline
#include "analysis/metrics_engine.hpp"
#include "analysis/pattern_detector.hpp"

// I/O system
#include "io/checkpoint_manager.hpp"
#include "io/data_exporter.hpp"

#ifdef ARTEMIS_HDF5_ENABLED
#include "io/hdf5_writer.hpp"
#endif

namespace artemis {

/**
 * @brief Get Artemis version string
 */
inline std::string version_string() {
    return std::to_string(ARTEMIS_VERSION_MAJOR) + "." +
           std::to_string(ARTEMIS_VERSION_MINOR) + "." +
           std::to_string(ARTEMIS_VERSION_PATCH);
}

/**
 * @brief Initialize Artemis library
 *
 * This should be called once at program start before using any Artemis features.
 * It initializes the GPU subsystem and sets up global state.
 *
 * @param device_id GPU device to use (-1 for auto-select)
 */
inline void initialize(int device_id = -1) {
    gpu::DeviceManager::instance().initialize(device_id);
}

/**
 * @brief Shutdown Artemis library
 *
 * Call this at program end to clean up GPU resources.
 */
inline void shutdown() {
    gpu::DeviceManager::instance().shutdown();
}

/**
 * @brief Print Artemis banner and system information
 */
inline void print_banner() {
    std::cout << "╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║            ARTEMIS Simulation Platform                ║\n";
    std::cout << "║  GPU-Accelerated Multiagent Simulation for Research   ║\n";
    std::cout << "║                                                       ║\n";
    std::cout << "║  Version: " << version_string() << "                                      ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════╝\n";
    std::cout << std::endl;

    gpu::DeviceManager::instance().print_device_info();
}

} // namespace artemis
