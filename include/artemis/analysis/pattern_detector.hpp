#pragma once

#include "../util/types.hpp"
#include "../core/agent_arrays.hpp"
#include <vector>
#include <string>

namespace artemis {
namespace analysis {

/**
 * @brief Types of emergent patterns
 */
enum class PatternType {
    CLUSTERING,          // Spatial clustering/aggregation
    WAVES,              // Traveling waves
    PHASE_TRANSITION,   // Critical transitions
    SYNCHRONIZATION,    // Temporal synchronization
    SEGREGATION,        // Spatial segregation by type
    FLOCKING,           // Coherent movement patterns
    CUSTOM              // User-defined pattern
};

/**
 * @brief Detected pattern information
 */
struct DetectedPattern {
    PatternType type;
    TimeStep detected_at;
    float confidence;           // [0, 1]
    std::string description;

    // Pattern-specific data
    std::map<std::string, float> parameters;
    std::vector<AgentID> involved_agents;
};

/**
 * @brief Abstract base class for pattern detectors
 *
 * Design Pattern: Strategy + Observer
 */
class PatternDetector {
public:
    virtual ~PatternDetector() = default;

    /**
     * @brief Detect patterns in current simulation state
     * @param agents Agent data
     * @param current_step Current timestep
     * @return List of detected patterns
     */
    virtual std::vector<DetectedPattern> detect(
        const core::AgentArrays& agents,
        TimeStep current_step) = 0;

    /**
     * @brief Get detector name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get pattern type this detector looks for
     */
    virtual PatternType pattern_type() const = 0;
};

/**
 * @brief Detects spatial clustering patterns using DBSCAN
 */
class ClusteringDetector : public PatternDetector {
public:
    struct Parameters {
        float epsilon = 10.0f;      // Neighborhood radius
        uint32_t min_points = 5;    // Minimum cluster size
        float min_confidence = 0.7f;
    };

    ClusteringDetector() = default;
    explicit ClusteringDetector(const Parameters& params);

    std::vector<DetectedPattern> detect(
        const core::AgentArrays& agents,
        TimeStep current_step) override;

    std::string name() const override { return "clustering_detector"; }
    PatternType pattern_type() const override { return PatternType::CLUSTERING; }

private:
    Parameters params_;
};

/**
 * @brief Detects phase transitions using critical slowing down
 */
class PhaseTransitionDetector : public PatternDetector {
public:
    struct Parameters {
        uint32_t window_size = 100;      // Timesteps to analyze
        float variance_threshold = 2.0f;  // Variance increase threshold
        float min_confidence = 0.8f;
    };

    PhaseTransitionDetector() = default;
    explicit PhaseTransitionDetector(const Parameters& params);

    std::vector<DetectedPattern> detect(
        const core::AgentArrays& agents,
        TimeStep current_step) override;

    std::string name() const override { return "phase_transition_detector"; }
    PatternType pattern_type() const override { return PatternType::PHASE_TRANSITION; }

private:
    Parameters params_;
    std::vector<float> variance_history_;
};

/**
 * @brief Detects traveling wave patterns
 */
class WaveDetector : public PatternDetector {
public:
    struct Parameters {
        float wavelength_min = 10.0f;
        float wavelength_max = 100.0f;
        float min_confidence = 0.75f;
    };

    WaveDetector() = default;
    explicit WaveDetector(const Parameters& params);

    std::vector<DetectedPattern> detect(
        const core::AgentArrays& agents,
        TimeStep current_step) override;

    std::string name() const override { return "wave_detector"; }
    PatternType pattern_type() const override { return PatternType::WAVES; }

private:
    Parameters params_;
};

/**
 * @brief Detects spatial segregation patterns
 */
class SegregationDetector : public PatternDetector {
public:
    struct Parameters {
        float segregation_threshold = 0.6f;  // Schelling-like threshold
        float min_confidence = 0.7f;
    };

    SegregationDetector() = default;
    explicit SegregationDetector(const Parameters& params);

    std::vector<DetectedPattern> detect(
        const core::AgentArrays& agents,
        TimeStep current_step) override;

    std::string name() const override { return "segregation_detector"; }
    PatternType pattern_type() const override { return PatternType::SEGREGATION; }

private:
    Parameters params_;
};

/**
 * @brief Manages multiple pattern detectors
 */
class PatternDetectionEngine {
public:
    PatternDetectionEngine() = default;

    /**
     * @brief Register a pattern detector
     */
    void register_detector(std::unique_ptr<PatternDetector> detector);

    /**
     * @brief Run all detectors on current state
     */
    std::vector<DetectedPattern> detect_all(
        const core::AgentArrays& agents,
        TimeStep current_step);

    /**
     * @brief Get history of detected patterns
     */
    const std::vector<DetectedPattern>& pattern_history() const {
        return pattern_history_;
    }

    /**
     * @brief Clear pattern history
     */
    void clear_history();

    /**
     * @brief Get patterns detected in last detection
     */
    const std::vector<DetectedPattern>& last_detected() const {
        return last_detected_;
    }

    /**
     * @brief Export pattern history as structured data
     */
    std::map<std::string, std::vector<DetectedPattern>> export_patterns_by_type() const;

private:
    std::vector<std::unique_ptr<PatternDetector>> detectors_;
    std::vector<DetectedPattern> pattern_history_;
    std::vector<DetectedPattern> last_detected_;
};

} // namespace analysis
} // namespace artemis
