/**
 * @file basic_boids.cpp
 * @brief Simple boids flocking simulation example
 *
 * This example demonstrates:
 * - Programmatic simulation setup
 * - Boids behavior configuration
 * - Metrics collection
 * - Basic visualization loop
 */

#include <artemis/artemis.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Initialize Artemis
    artemis::initialize();
    artemis::print_banner();

    // Create simulation
    artemis::core::Simulation sim;

    // Configure simulation programmatically
    const size_t num_agents = 10000;
    artemis::BoundingBox2D bounds(
        artemis::float2(0.0f, 0.0f),
        artemis::float2(1000.0f, 1000.0f)
    );

    sim.initialize(num_agents, bounds, artemis::TopologyType::TORUS);

    // Configure scheduler
    sim.scheduler().set_delta_time(0.1f);
    sim.scheduler().set_type(artemis::core::SchedulerType::SYNCHRONOUS);

    // Add boids behavior
    artemis::behaviors::BoidsBehavior::Parameters boids_params;
    boids_params.separation_radius = 5.0f;
    boids_params.alignment_radius = 15.0f;
    boids_params.cohesion_radius = 15.0f;
    boids_params.separation_weight = 1.5f;
    boids_params.alignment_weight = 1.0f;
    boids_params.cohesion_weight = 1.0f;
    boids_params.max_speed = 5.0f;

    auto boids_behavior = std::make_unique<artemis::behaviors::BoidsBehavior>(boids_params);
    sim.behavior_manager().add_behavior(std::move(boids_behavior));

    // Register metrics
    sim.metrics().register_metric(
        std::make_unique<artemis::analysis::PolarizationMetric>()
    );
    sim.metrics().register_metric(
        std::make_unique<artemis::analysis::CenterOfMassMetric>()
    );
    sim.metrics().set_frequency(10);  // Compute every 10 steps

    // Run simulation
    const size_t num_steps = 1000;
    std::cout << "Running simulation for " << num_steps << " steps...\n";

    for (size_t step = 0; step < num_steps; ++step) {
        sim.step();

        // Print progress every 100 steps
        if (step % 100 == 0) {
            std::cout << "Step " << step << "/" << num_steps
                      << " - Agents: " << sim.num_agents() << "\n";

            // Get latest polarization metric
            if (step > 0) {
                auto polarization = std::get<float>(
                    sim.metrics().get_latest_value("polarization")
                );
                std::cout << "  Polarization: " << polarization << "\n";
            }
        }
    }

    std::cout << "Simulation complete!\n";

    // Export results
    std::cout << "Exporting metrics to CSV...\n";
    artemis::io::CSVExporter csv_exporter;
    csv_exporter.export_metrics(
        "boids_metrics.csv",
        sim.metrics().export_timeseries()
    );

    std::cout << "Exporting final state...\n";
    artemis::io::JSONExporter json_exporter;
    json_exporter.export_agent_state(
        "boids_final_state.json",
        sim.agents(),
        sim.current_step()
    );

    // Shutdown
    artemis::shutdown();

    std::cout << "Done!\n";
    return 0;
}
