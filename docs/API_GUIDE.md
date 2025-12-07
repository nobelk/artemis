# Artemis API Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Programmatic API](#programmatic-api)
4. [Configuration-Based Setup](#configuration-based-setup)
5. [Custom Behaviors](#custom-behaviors)
6. [Custom Metrics](#custom-metrics)
7. [Data Export](#data-export)
8. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Minimal Example

```cpp
#include <artemis/artemis.hpp>

int main() {
    // Initialize
    artemis::initialize();

    // Create simulation
    artemis::core::Simulation sim;
    sim.initialize(
        10000,  // num_agents
        artemis::BoundingBox2D(
            artemis::float2(0, 0),
            artemis::float2(1000, 1000)
        )
    );

    // Add behavior
    auto behavior = std::make_unique<artemis::behaviors::BoidsBehavior>();
    sim.behavior_manager().add_behavior(std::move(behavior));

    // Run simulation
    for (int i = 0; i < 1000; ++i) {
        sim.step();
    }

    // Cleanup
    artemis::shutdown();
    return 0;
}
```

---

## Core Concepts

### Agent State

Agents are stored in Structure of Arrays (SoA) format for GPU efficiency:

```cpp
// Agent properties are stored in separate arrays
float2* positions;     // Position vectors
float2* velocities;    // Velocity vectors
uint8_t* types;        // Agent types
float* energies;       // Energy levels
uint32_t* ages;        // Age counters
```

### Simulation Pipeline

Each simulation step executes:

1. **Spatial indexing** - Build neighbor lookup structure
2. **Behavior execution** - Update agent velocities
3. **Physics integration** - Update positions from velocities
4. **Boundary conditions** - Wrap/reflect at boundaries
5. **Metrics computation** - Calculate statistics
6. **Event processing** - Handle scheduled callbacks

---

## Programmatic API

### Creating a Simulation

```cpp
using namespace artemis;

// Method 1: Simple initialization
core::Simulation sim;
sim.initialize(
    num_agents,
    bounds,
    TopologyType::TORUS
);

// Method 2: From configuration
config::SimulationConfig config = /* ... */;
sim.initialize(config);
```

### Configuring the Environment

```cpp
// Set environment bounds
BoundingBox2D bounds(float2(0, 0), float2(1000, 1000));

// Set topology
sim.environment().set_topology(TopologyType::TORUS);
// Options: BOUNDED, TORUS, INFINITE

// Check if position is valid
bool valid = sim.environment().is_valid_position(position);
```

### Adding Behaviors

```cpp
// Create behavior with parameters
behaviors::BoidsBehavior::Parameters params;
params.separation_weight = 1.5f;
params.alignment_weight = 1.0f;
params.cohesion_weight = 1.0f;

auto behavior = std::make_unique<behaviors::BoidsBehavior>(params);

// Add to simulation with priority
sim.behavior_manager().add_behavior(std::move(behavior), priority);

// Enable/disable behaviors
sim.behavior_manager().enable_behavior("boids", true);
```

### Scheduling

```cpp
// Set time step
sim.scheduler().set_delta_time(0.1f);

// Register periodic callback
sim.scheduler().register_event(
    100,  // interval (every 100 steps)
    [](TimeStep step) {
        std::cout << "Callback at step " << step << "\n";
    }
);

// Manual stepping
sim.step();

// Run for N steps
sim.run(1000);

// Run until condition
sim.run_until([]() {
    return /* stop condition */;
});
```

### Metrics Collection

```cpp
// Register metrics
sim.metrics().register_metric(
    std::make_unique<analysis::PolarizationMetric>()
);

sim.metrics().register_metric(
    std::make_unique<analysis::ClusteringCoefficientMetric>(
        interaction_radius
    )
);

// Set computation frequency
sim.metrics().set_frequency(10);  // Every 10 steps

// Get latest value
auto value = sim.metrics().get_latest_value("polarization");
float polarization = std::get<float>(value);

// Get time series
const auto& ts = sim.metrics().get_timeseries("polarization");
for (size_t i = 0; i < ts.timesteps.size(); ++i) {
    TimeStep step = ts.timesteps[i];
    float value = std::get<float>(ts.values[i]);
}
```

### Pattern Detection

```cpp
using namespace artemis::analysis;

// Create pattern detector
ClusteringDetector::Parameters params;
params.epsilon = 10.0f;
params.min_points = 5;

auto detector = std::make_unique<ClusteringDetector>(params);

// Detect patterns
auto patterns = detector->detect(sim.agents(), sim.current_step());

for (const auto& pattern : patterns) {
    std::cout << "Found " << pattern.description
              << " with confidence " << pattern.confidence << "\n";
}
```

---

## Configuration-Based Setup

### YAML Configuration

```yaml
name: "my_simulation"
random_seed: 42

agent_types:
  - name: "boid"
    count: 10000
    max_speed: 5.0
    perception_radius: 15.0

environment:
  type: "grid_2d"
  dimensions: [1000, 1000]
  topology: "torus"
  cell_size: 50.0

temporal:
  max_steps: 10000
  dt: 0.1
  scheduler: "synchronous"

behaviors:
  - type: "boids"
    parameters:
      separation_weight: 1.5
      alignment_weight: 1.0
      cohesion_weight: 1.0

analysis:
  metrics:
    - "polarization"
    - "clustering_coefficient"
  frequency: 10
```

### Loading Configuration

```cpp
// From file
auto config = config::YAMLParser::load_from_file("config.yaml");

// Validate
config.validate();

// Print summary
config.print_summary();

// Initialize simulation
sim.initialize(config);
```

### Building Configuration Programmatically

```cpp
using namespace artemis::config;

auto config = ConfigBuilder()
    .set_name("my_simulation")
    .set_seed(42)
    .add_agent_type(AgentTypeConfig{
        .name = "boid",
        .count = 10000,
        .max_speed = 5.0f
    })
    .set_environment(EnvironmentConfig{
        .dimensions = {1000.0f, 1000.0f},
        .topology = "torus"
    })
    .build();
```

---

## Custom Behaviors

### Implementing a Custom Behavior

```cpp
#include <artemis/behaviors/behavior.hpp>

class MyCustomBehavior : public artemis::behaviors::Behavior {
public:
    struct Parameters {
        float strength = 1.0f;
        float radius = 10.0f;
    };

    MyCustomBehavior(const Parameters& params)
        : params_(params) {}

    void execute(
        core::AgentArrays& agents,
        gpu::SpatialIndex& spatial_index,
        float dt) override {

        // Launch CUDA kernel to update agent velocities
        launch_my_kernel<<<grid, block>>>(
            agents.positions(),
            agents.velocities(),
            agents.size(),
            params_.strength,
            params_.radius
        );

        CUDA_CHECK_LAST_ERROR();
    }

    std::string name() const override {
        return "my_custom_behavior";
    }

    void initialize(const void* config) override {
        // Initialize from config if needed
    }

    void get_parameters(std::map<std::string, float>& params) const override {
        params["strength"] = params_.strength;
        params["radius"] = params_.radius;
    }

    void set_parameters(const std::map<std::string, float>& params) override {
        if (params.count("strength"))
            params_.strength = params.at("strength");
        if (params.count("radius"))
            params_.radius = params.at("radius");
    }

private:
    Parameters params_;
};

// Register the behavior
REGISTER_BEHAVIOR(MyCustomBehavior, "my_custom_behavior");
```

### CUDA Kernel Example

```cuda
__global__ void launch_my_kernel(
    float2* positions,
    float2* velocities,
    size_t num_agents,
    float strength,
    float radius) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;

    float2 pos = positions[idx];
    float2 vel = velocities[idx];

    // Your custom behavior logic here
    float2 force = compute_custom_force(pos, vel);

    // Update velocity
    velocities[idx] = vel + force * strength;
}
```

---

## Custom Metrics

### Implementing a Custom Metric

```cpp
#include <artemis/analysis/metrics_engine.hpp>

class MyCustomMetric : public artemis::analysis::Metric {
public:
    MetricValue compute(const core::AgentArrays& agents) override {
        // Compute metric on GPU
        float result = 0.0f;

        // Launch reduction kernel
        gpu_compute_my_metric<<<grid, block>>>(
            agents.positions(),
            agents.size(),
            result_buffer.data()
        );

        // Download result
        result_buffer.download(&result, 1);

        return result;
    }

    std::string name() const override {
        return "my_custom_metric";
    }

private:
    gpu::DeviceBuffer<float> result_buffer{1};
};

// Register and use
sim.metrics().register_metric(
    std::make_unique<MyCustomMetric>()
);
```

---

## Data Export

### HDF5 Export

```cpp
#ifdef ARTEMIS_HDF5_ENABLED
io::HDF5Writer writer;
writer.open("output.h5", true);  // overwrite

// Write metadata
writer.write_metadata("simulation_name", num_agents, num_steps, dt);

// Write agent frames
for (size_t step = 0; step < num_steps; ++step) {
    sim.step();
    writer.write_agent_frame(sim.agents(), step);
}

// Write metrics
writer.write_metrics(sim.metrics().export_timeseries());

writer.close();
#endif
```

### CSV Export

```cpp
io::CSVExporter csv;

// Export metrics
csv.export_metrics(
    "metrics.csv",
    sim.metrics().export_timeseries()
);

// Export agent state
csv.export_agent_state(
    "agents.csv",
    sim.agents(),
    sim.current_step()
);
```

### JSON Export

```cpp
io::JSONExporter json;

json.export_agent_state(
    "state.json",
    sim.agents(),
    sim.current_step(),
    true  // pretty print
);
```

---

## Advanced Topics

### Checkpointing

```cpp
// Save checkpoint
sim.save_checkpoint("checkpoint_step_1000.art");

// Load checkpoint
sim.load_checkpoint("checkpoint_step_1000.art");

// Auto-checkpointing
io::AutoCheckpointer::Config checkpoint_config;
checkpoint_config.interval = 1000;
checkpoint_config.max_checkpoints = 10;

io::AutoCheckpointer auto_checkpoint(checkpoint_config);

for (size_t step = 0; step < max_steps; ++step) {
    sim.step();

    if (auto_checkpoint.should_checkpoint(step)) {
        auto_checkpoint.auto_checkpoint(sim, step);
    }
}
```

### GPU Device Management

```cpp
// Select specific GPU
artemis::initialize(0);  // Use GPU 0

// Auto-select best GPU
artemis::initialize(-1);

// Get device info
auto& device_mgr = gpu::DeviceManager::instance();
const auto& info = device_mgr.current_device_info();

std::cout << "GPU: " << info.name << "\n";
std::cout << "Memory: " << info.total_memory / (1024*1024) << " MB\n";
std::cout << "Compute: " << info.compute_capability_major << "."
          << info.compute_capability_minor << "\n";

// Performance stats
const auto& stats = device_mgr.performance_stats();
std::cout << "Compute time: " << stats.total_compute_time_ms << " ms\n";
```

### Custom Spatial Queries

```cpp
// Get spatial index
auto& spatial_index = sim.spatial_index();

// Query neighbors
const float radius = 15.0f;
const size_t max_neighbors = 100;

gpu::DeviceBuffer<AgentID> neighbors(max_neighbors);
gpu::DeviceBuffer<uint32_t> neighbor_counts(1);

spatial_index.query_neighbors_gpu(
    agents.positions(),
    radius,
    agents,
    1,  // num_queries
    neighbors.data(),
    neighbor_counts.data(),
    max_neighbors
);
```

### Performance Profiling

```cpp
// Enable profiling
gpu::Event start, end;

start.record();

// Your simulation code
sim.step();

end.record();
end.synchronize();

float elapsed_ms = end.elapsed_time(start);
std::cout << "Step time: " << elapsed_ms << " ms\n";
```

---

## Best Practices

1. **Always initialize and shutdown**
   ```cpp
   artemis::initialize();
   // ... use Artemis ...
   artemis::shutdown();
   ```

2. **Validate configurations**
   ```cpp
   config.validate();  // Throws on error
   ```

3. **Handle GPU errors**
   ```cpp
   try {
       sim.step();
   } catch (const gpu::CudaException& e) {
       std::cerr << "GPU error: " << e.what() << "\n";
   }
   ```

4. **Minimize CPU↔GPU transfers**
   - Keep data on GPU between steps
   - Batch transfers when possible
   - Use GPU reduction for metrics

5. **Optimize spatial index**
   - Set cell_size = 2 × interaction_radius
   - Rebuild only when agents move significantly

6. **Save checkpoints periodically**
   - Use auto-checkpointing for long simulations
   - Enables fault tolerance and result validation

---

## Examples

See `examples/` directory for complete working examples:
- `basic_boids.cpp` - Programmatic setup
- `config_based.cpp` - YAML configuration
- `configs/boids.yaml` - Example configuration file

## API Reference

For detailed API documentation, see the header files in `include/artemis/`.
Each class and function is documented with Doxygen-style comments.
