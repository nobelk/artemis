/**
 * @file test_metrics.cpp
 * @brief Tests for metrics system
 */

#include <artemis/analysis/metrics_engine.hpp>
#include <artemis/core/agent_arrays.hpp>
#include <iostream>

namespace test {
    void assert_true(bool condition, const std::string& message);
    void assert_equal(float a, float b, const std::string& message, float epsilon);
}

void test_metrics() {
    using namespace artemis;

    // Test 1: MetricsEngine construction
    {
        analysis::MetricsEngine engine;
        test::assert_equal(engine.metric_names().size(), 0, "Should start empty", 0.0f);
    }

    // Test 2: Register metrics
    {
        analysis::MetricsEngine engine;

        engine.register_metric(
            std::make_unique<analysis::PolarizationMetric>()
        );
        engine.register_metric(
            std::make_unique<analysis::CenterOfMassMetric>()
        );

        test::assert_equal(engine.metric_names().size(), 2, "Should have 2 metrics", 0.0f);
        test::assert_true(engine.has_metric("polarization"), "Should have polarization");
        test::assert_true(engine.has_metric("center_of_mass"), "Should have center_of_mass");
    }

    // Test 3: Compute metrics
    {
        analysis::MetricsEngine engine;
        engine.register_metric(
            std::make_unique<analysis::PolarizationMetric>()
        );

        // Create agents with aligned velocities
        core::AgentArrays agents;
        const size_t num_agents = 100;

        std::vector<AgentState> host_agents(num_agents);
        for (size_t i = 0; i < num_agents; ++i) {
            host_agents[i].position = float2(i * 1.0f, 0.0f);
            host_agents[i].velocity = float2(1.0f, 0.0f);  // All same direction
        }

        agents.upload_from_host(host_agents);

        // Compute metrics
        engine.compute_all(agents, 0);

        // Polarization should be ~1.0 (perfectly aligned)
        auto polarization = std::get<float>(
            engine.get_latest_value("polarization")
        );

        test::assert_true(polarization > 0.95f, "Polarization should be high for aligned velocities");
    }

    // Test 4: Time series
    {
        analysis::MetricsEngine engine;
        engine.register_metric(
            std::make_unique<analysis::PolarizationMetric>()
        );

        core::AgentArrays agents;
        agents.allocate(100);

        // Compute metrics for multiple timesteps
        for (TimeStep t = 0; t < 10; ++t) {
            engine.compute_all(agents, t);
        }

        const auto& timeseries = engine.get_timeseries("polarization");
        test::assert_equal(timeseries.size(), 10, "Should have 10 values", 0.0f);
    }

    std::cout << "  All Metrics tests passed\n";
}
