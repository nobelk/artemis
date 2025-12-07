#pragma once

#include "../util/types.hpp"
#include "../util/gpu_util.hpp"
#include "../core/agent_arrays.hpp"
#include <string>
#include <map>
#include <vector>
#include <memory>

namespace artemis {
namespace analysis {

/**
 * @brief Metric value variant
 */
using MetricValue = std::variant<float, float2, std::vector<float>>;

/**
 * @brief Time series of metric values
 */
struct MetricTimeSeries {
    std::string name;
    std::vector<TimeStep> timesteps;
    std::vector<MetricValue> values;

    void append(TimeStep step, const MetricValue& value);
    void clear();
    size_t size() const { return timesteps.size(); }
};

/**
 * @brief Abstract base class for metrics
 *
 * Design Pattern: Strategy for different metric computations
 */
class Metric {
public:
    virtual ~Metric() = default;

    /**
     * @brief Compute metric from agent data
     * @param agents Agent data
     * @return Computed metric value
     */
    virtual MetricValue compute(const core::AgentArrays& agents) = 0;

    /**
     * @brief Get metric name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Check if metric is GPU-accelerated
     */
    virtual bool is_gpu_accelerated() const { return true; }
};

/**
 * @brief Polarization metric (order parameter for flocking)
 *
 * Measures alignment of velocities:
 * P = |⟨v⟩| / v_avg
 * Range: [0, 1], where 1 = perfect alignment
 */
class PolarizationMetric : public Metric {
public:
    MetricValue compute(const core::AgentArrays& agents) override;
    std::string name() const override { return "polarization"; }
};

/**
 * @brief Clustering coefficient metric
 *
 * Measures spatial clustering using neighbor graph
 */
class ClusteringCoefficientMetric : public Metric {
public:
    explicit ClusteringCoefficientMetric(float interaction_radius);

    MetricValue compute(const core::AgentArrays& agents) override;
    std::string name() const override { return "clustering_coefficient"; }

private:
    float interaction_radius_;
};

/**
 * @brief Neighbor distribution metric
 *
 * Computes histogram of neighbor counts
 */
class NeighborDistributionMetric : public Metric {
public:
    explicit NeighborDistributionMetric(float interaction_radius, uint32_t max_bins = 50);

    MetricValue compute(const core::AgentArrays& agents) override;
    std::string name() const override { return "neighbor_distribution"; }

private:
    float interaction_radius_;
    uint32_t max_bins_;
    gpu::DeviceBuffer<uint32_t> histogram_;
};

/**
 * @brief Center of mass metric
 */
class CenterOfMassMetric : public Metric {
public:
    MetricValue compute(const core::AgentArrays& agents) override;
    std::string name() const override { return "center_of_mass"; }
};

/**
 * @brief Average velocity metric
 */
class AverageVelocityMetric : public Metric {
public:
    MetricValue compute(const core::AgentArrays& agents) override;
    std::string name() const override { return "average_velocity"; }
};

/**
 * @brief Spatial density metric
 *
 * Computes 2D histogram of agent positions
 */
class SpatialDensityMetric : public Metric {
public:
    explicit SpatialDensityMetric(
        const BoundingBox2D& bounds,
        uint32_t bins_x = 50,
        uint32_t bins_y = 50);

    MetricValue compute(const core::AgentArrays& agents) override;
    std::string name() const override { return "spatial_density"; }

private:
    BoundingBox2D bounds_;
    uint32_t bins_x_;
    uint32_t bins_y_;
    gpu::DeviceBuffer<uint32_t> histogram_;
};

/**
 * @brief Energy statistics metric
 */
class EnergyStatisticsMetric : public Metric {
public:
    struct Stats {
        float mean;
        float min;
        float max;
        float stddev;
    };

    MetricValue compute(const core::AgentArrays& agents) override;
    std::string name() const override { return "energy_statistics"; }
};

/**
 * @brief Manages and computes multiple metrics
 *
 * Design Pattern: Facade + Observer (for metric updates)
 */
class MetricsEngine {
public:
    MetricsEngine() = default;

    /**
     * @brief Register a metric for computation
     */
    void register_metric(std::unique_ptr<Metric> metric);

    /**
     * @brief Unregister metric by name
     */
    void unregister_metric(const std::string& name);

    /**
     * @brief Compute all registered metrics
     * @param agents Agent data
     * @param current_step Current simulation timestep
     */
    void compute_all(const core::AgentArrays& agents, TimeStep current_step);

    /**
     * @brief Compute specific metric by name
     */
    MetricValue compute_metric(
        const std::string& name,
        const core::AgentArrays& agents);

    /**
     * @brief Get time series for a metric
     */
    const MetricTimeSeries& get_timeseries(const std::string& name) const;

    /**
     * @brief Get latest value for a metric
     */
    MetricValue get_latest_value(const std::string& name) const;

    /**
     * @brief Check if metric exists
     */
    bool has_metric(const std::string& name) const;

    /**
     * @brief Get list of registered metric names
     */
    std::vector<std::string> metric_names() const;

    /**
     * @brief Clear all time series data
     */
    void clear_timeseries();

    /**
     * @brief Export all time series to map
     */
    std::map<std::string, MetricTimeSeries> export_timeseries() const;

    /**
     * @brief Set computation frequency (compute every N steps)
     */
    void set_frequency(uint32_t frequency) { frequency_ = frequency; }

    uint32_t frequency() const { return frequency_; }

private:
    std::map<std::string, std::unique_ptr<Metric>> metrics_;
    std::map<std::string, MetricTimeSeries> timeseries_;
    uint32_t frequency_ = 1;
};

} // namespace analysis
} // namespace artemis
