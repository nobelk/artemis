#pragma once

#include "../util/gpu_util.hpp"
#include <vector>
#include <string>

namespace artemis {
namespace gpu {

/**
 * @brief Manages GPU device selection and resource allocation
 *
 * Responsibilities:
 * - Device enumeration and selection
 * - Memory pool management
 * - Multi-GPU coordination (future)
 * - Performance monitoring
 *
 * Design Pattern: Singleton for global GPU state management
 */
class DeviceManager {
public:
    static DeviceManager& instance();

    // Prevent copying and moving
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    /**
     * @brief Initialize GPU subsystem
     * @param device_id Preferred device ID (-1 for auto-select)
     */
    void initialize(int device_id = -1);

    /**
     * @brief Shutdown GPU subsystem and free resources
     */
    void shutdown();

    /**
     * @brief Select best available GPU based on criteria
     * @return Selected device ID
     */
    int select_best_device();

    /**
     * @brief Set active GPU device
     */
    void set_device(int device_id);

    /**
     * @brief Get current active device ID
     */
    int current_device() const { return current_device_; }

    /**
     * @brief Get list of all available devices
     */
    const std::vector<DeviceInfo>& available_devices() const {
        return devices_;
    }

    /**
     * @brief Get info for current device
     */
    const DeviceInfo& current_device_info() const;

    /**
     * @brief Query available GPU memory
     */
    size_t available_memory() const;

    /**
     * @brief Query total GPU memory
     */
    size_t total_memory() const;

    /**
     * @brief Check if device supports required features
     */
    bool supports_compute_capability(int major, int minor) const;

    /**
     * @brief Synchronize all GPU operations
     */
    void synchronize();

    /**
     * @brief Print device information to stdout
     */
    void print_device_info() const;

    // Performance monitoring
    struct PerformanceStats {
        float total_compute_time_ms = 0.0f;
        float total_memory_transfer_ms = 0.0f;
        size_t total_memory_allocated = 0;
        size_t peak_memory_used = 0;
        uint64_t num_kernel_launches = 0;
    };

    const PerformanceStats& performance_stats() const { return stats_; }
    void reset_performance_stats();

private:
    DeviceManager() = default;
    ~DeviceManager();

    void enumerate_devices();

    int current_device_ = -1;
    std::vector<DeviceInfo> devices_;
    PerformanceStats stats_;
    bool initialized_ = false;
};

} // namespace gpu
} // namespace artemis
