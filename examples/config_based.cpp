/**
 * @file config_based.cpp
 * @brief Example of simulation setup from YAML configuration
 *
 * This example demonstrates:
 * - Loading configuration from YAML file
 * - Running simulation with multiple behaviors
 * - Automatic checkpointing
 * - HDF5 data export
 */

#include <artemis/artemis.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>\n";
        return 1;
    }

    // Initialize Artemis
    artemis::initialize();
    artemis::print_banner();

    // Load configuration from YAML
    std::string config_path = argv[1];
    std::cout << "Loading configuration from: " << config_path << "\n";

    artemis::config::SimulationConfig config;
    try {
        config = artemis::config::YAMLParser::load_from_file(config_path);
        config.validate();
        config.print_summary();
    } catch (const std::exception& e) {
        std::cerr << "Configuration error: " << e.what() << "\n";
        return 1;
    }

    // Create simulation from config
    artemis::core::Simulation sim;
    sim.initialize(config);

    std::cout << "\nSimulation initialized with:\n";
    std::cout << "  Agents: " << sim.num_agents() << "\n";
    std::cout << "  Environment: " << config.environment.dimensions[0]
              << " x " << config.environment.dimensions[1] << "\n";
    std::cout << "  Max steps: " << config.temporal.max_steps << "\n";
    std::cout << "  Behaviors: " << sim.behavior_manager().num_behaviors() << "\n";

    // Setup auto-checkpointing
    artemis::io::AutoCheckpointer auto_checkpoint;
    if (config.io.enable_checkpointing) {
        artemis::io::AutoCheckpointer::Config checkpoint_config;
        checkpoint_config.directory = config.io.output_directory + "/checkpoints";
        checkpoint_config.interval = config.io.checkpoint_interval;
        checkpoint_config.max_checkpoints = 10;
        checkpoint_config.compress = config.io.compress_output;

        auto_checkpoint.initialize(checkpoint_config);
        std::cout << "Auto-checkpointing enabled (every "
                  << checkpoint_config.interval << " steps)\n";
    }

#ifdef ARTEMIS_HDF5_ENABLED
    // Setup HDF5 writer
    artemis::io::HDF5Writer hdf5_writer;
    if (config.analysis.export_timeseries &&
        config.analysis.output_format == "hdf5") {

        std::string output_path = config.io.output_directory + "/" +
                                 config.name + ".h5";
        hdf5_writer.open(output_path, true);
        hdf5_writer.write_metadata(
            config.name,
            sim.num_agents(),
            config.temporal.max_steps,
            config.temporal.dt
        );
        hdf5_writer.write_config(
            artemis::config::YAMLParser::to_yaml_string(config)
        );
        hdf5_writer.initialize_agent_datasets(
            sim.num_agents(),
            config.temporal.max_steps
        );

        std::cout << "HDF5 output enabled: " << output_path << "\n";
    }
#endif

    // Run simulation
    std::cout << "\nStarting simulation...\n";

    const size_t max_steps = config.temporal.max_steps;
    const size_t print_interval = max_steps / 10;  // Print 10 times

    for (size_t step = 0; step < max_steps; ++step) {
        // Step simulation
        sim.step();

        // Auto-checkpoint
        if (config.io.enable_checkpointing &&
            auto_checkpoint.should_checkpoint(step)) {
            std::string checkpoint_path = auto_checkpoint.auto_checkpoint(sim, step);
            std::cout << "  Checkpoint saved: " << checkpoint_path << "\n";
        }

#ifdef ARTEMIS_HDF5_ENABLED
        // Write agent frame to HDF5
        if (hdf5_writer.is_open()) {
            hdf5_writer.write_agent_frame(sim.agents(), step);
        }
#endif

        // Print progress
        if (step % print_interval == 0 || step == max_steps - 1) {
            float progress = 100.0f * step / max_steps;
            std::cout << "Progress: " << progress << "% (step " << step
                      << "/" << max_steps << ")\n";

            // Print metrics
            for (const auto& metric_name : sim.metrics().metric_names()) {
                auto value = sim.metrics().get_latest_value(metric_name);
                if (std::holds_alternative<float>(value)) {
                    std::cout << "  " << metric_name << ": "
                              << std::get<float>(value) << "\n";
                }
            }
        }
    }

    std::cout << "\nSimulation complete!\n";

#ifdef ARTEMIS_HDF5_ENABLED
    // Finalize HDF5
    if (hdf5_writer.is_open()) {
        hdf5_writer.write_metrics(sim.metrics().export_timeseries());
        hdf5_writer.close();
        std::cout << "HDF5 data written successfully.\n";
    }
#endif

    // Export final metrics
    std::string metrics_path = config.io.output_directory + "/" +
                              config.name + "_metrics.csv";
    artemis::io::CSVExporter csv_exporter;
    csv_exporter.export_metrics(metrics_path, sim.metrics().export_timeseries());
    std::cout << "Metrics exported to: " << metrics_path << "\n";

    // Cleanup
    if (config.io.enable_checkpointing) {
        auto_checkpoint.cleanup_old_checkpoints();
    }

    artemis::shutdown();

    std::cout << "Done!\n";
    return 0;
}
