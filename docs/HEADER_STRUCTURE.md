# Artemis Header File Structure

## Complete Header File Organization

```
include/artemis/
├── artemis.hpp                          # Main convenience header (include this)
│
├── util/                                # Utility Layer
│   ├── types.hpp                       # Core types, enums, GPU-compatible structs
│   ├── math.hpp                        # Vector operations, Morton codes, RNG
│   └── gpu_util.hpp                    # GPU memory (RAII), error handling, streams
│
├── core/                                # Simulation Core
│   ├── agent_arrays.hpp                # Structure of Arrays for agent data
│   ├── environment.hpp                 # Spatial bounds, topology, boundaries
│   ├── scheduler.hpp                   # Temporal control, event scheduling
│   └── simulation.hpp                  # Main simulation orchestrator (Facade)
│
├── gpu/                                 # GPU Computation Layer
│   ├── device_manager.hpp              # GPU device selection and management
│   ├── spatial_index.hpp               # Uniform grid with Morton codes
│   └── kernels.cuh                     # CUDA kernel declarations
│
├── config/                              # Configuration System
│   ├── simulation_config.hpp           # Configuration structures (Builder)
│   └── yaml_parser.hpp                 # YAML/JSON parsing
│
├── behaviors/                           # Behavior System (Strategy Pattern)
│   ├── behavior.hpp                    # Abstract base + Factory
│   ├── behavior_manager.hpp            # Composite + Chain of Responsibility
│   ├── boids_behavior.hpp              # Flocking (Reynolds 1987)
│   ├── predator_prey_behavior.hpp      # Predator-prey dynamics
│   └── social_behavior.hpp             # Social interactions, opinion dynamics
│
├── analysis/                            # Analysis Pipeline
│   ├── metrics_engine.hpp              # Metrics computation (Observer pattern)
│   └── pattern_detector.hpp            # Emergent pattern detection
│
└── io/                                  # I/O System
    ├── checkpoint_manager.hpp          # State persistence (Memento pattern)
    ├── hdf5_writer.hpp                 # HDF5 time series export
    └── data_exporter.hpp               # CSV/JSON export
```

## Header Dependencies

### Level 0: Foundation (No dependencies)
- `util/types.hpp`

### Level 1: Utilities
- `util/math.hpp` → `types.hpp`
- `util/gpu_util.hpp` → `types.hpp`

### Level 2: Core Components
- `core/agent_arrays.hpp` → `types.hpp`, `gpu_util.hpp`
- `core/environment.hpp` → `types.hpp`, `agent_arrays.hpp`
- `core/scheduler.hpp` → `types.hpp`
- `gpu/device_manager.hpp` → `gpu_util.hpp`
- `gpu/spatial_index.hpp` → `types.hpp`, `gpu_util.hpp`, `agent_arrays.hpp`
- `gpu/kernels.cuh` → `types.hpp`, `math.hpp`

### Level 3: Configuration & Behaviors
- `config/simulation_config.hpp` → `types.hpp`
- `config/yaml_parser.hpp` → `simulation_config.hpp`
- `behaviors/behavior.hpp` → `types.hpp`, `agent_arrays.hpp`, `spatial_index.hpp`
- `behaviors/boids_behavior.hpp` → `behavior.hpp`
- `behaviors/predator_prey_behavior.hpp` → `behavior.hpp`
- `behaviors/social_behavior.hpp` → `behavior.hpp`
- `behaviors/behavior_manager.hpp` → `behavior.hpp`

### Level 4: Analysis & I/O
- `analysis/metrics_engine.hpp` → `types.hpp`, `agent_arrays.hpp`, `gpu_util.hpp`
- `analysis/pattern_detector.hpp` → `types.hpp`, `agent_arrays.hpp`
- `io/checkpoint_manager.hpp` → `simulation.hpp`
- `io/hdf5_writer.hpp` → `types.hpp`, `agent_arrays.hpp`, `metrics_engine.hpp`
- `io/data_exporter.hpp` → `agent_arrays.hpp`, `metrics_engine.hpp`

### Level 5: Simulation Orchestrator
- `core/simulation.hpp` → ALL previous levels

### Level 6: Convenience Header
- `artemis.hpp` → ALL headers

## Design Patterns by Component

### Creational Patterns
- **Factory**: `BehaviorFactory` (behaviors/behavior.hpp)
- **Builder**: `ConfigBuilder` (config/simulation_config.hpp)
- **Singleton**: `DeviceManager` (gpu/device_manager.hpp)

### Structural Patterns
- **Facade**: `Simulation` (core/simulation.hpp), `HDF5Writer` (io/hdf5_writer.hpp)
- **Adapter**: Data exporters (io/data_exporter.hpp)
- **Composite**: `CompositeBehavior` (behaviors/behavior_manager.hpp)

### Behavioral Patterns
- **Strategy**: `Behavior` hierarchy (behaviors/behavior.hpp)
- **Observer**: Metrics system (analysis/metrics_engine.hpp)
- **Template Method**: `Behavior::execute()` (behaviors/behavior.hpp)
- **Chain of Responsibility**: `BehaviorManager` (behaviors/behavior_manager.hpp)
- **Memento**: `CheckpointManager` (io/checkpoint_manager.hpp)
- **Command**: Event callbacks (core/scheduler.hpp)

### Other Patterns
- **RAII**: GPU resources (util/gpu_util.hpp)
- **Data-Oriented Design**: SoA layout (core/agent_arrays.hpp)

## Key Architectural Decisions

### 1. Header-Only vs Implementation Split
- **Headers**: Interfaces, inline functions, templates
- **Implementation**: Complex logic, CUDA kernels, I/O

### 2. GPU Memory Management
- RAII wrappers (`DeviceBuffer`, `Stream`, `Event`)
- Move-only semantics (prevent accidental copies)
- Automatic cleanup (no manual cudaFree)

### 3. Type Safety
- Strong typing for IDs (`AgentID`, `CellID`, `MortonCode`)
- Enums for configuration (`TopologyType`, `SchedulerType`)
- `std::variant` for heterogeneous data (`MetricValue`, `ConfigValue`)

### 4. Extensibility Points
- Virtual interfaces for behaviors and metrics
- Factory registration macros
- Configuration-driven initialization

### 5. GPU-CPU Boundary
- Minimal transfers (keep data on GPU)
- Explicit device/host functions (`__host__ __device__`)
- Async operations via streams

## Usage Patterns

### Quick Start (Programmatic)
```cpp
#include <artemis/artemis.hpp>

artemis::initialize();
artemis::core::Simulation sim;
sim.initialize(num_agents, bounds);
sim.run(num_steps);
artemis::shutdown();
```

### Configuration-Based
```cpp
#include <artemis/artemis.hpp>

auto config = artemis::config::YAMLParser::load_from_file("config.yaml");
artemis::core::Simulation sim;
sim.initialize(config);
sim.run(config.temporal.max_steps);
```

### Custom Behavior
```cpp
#include <artemis/behaviors/behavior.hpp>

class MyBehavior : public artemis::behaviors::Behavior {
    void execute(/* ... */) override { /* GPU kernel launch */ }
    std::string name() const override { return "my_behavior"; }
};
```

## Header File Guidelines

### 1. All headers are self-contained
- Include all dependencies
- Can be included independently

### 2. Include guards via `#pragma once`
- Modern, widely supported
- Simpler than traditional guards

### 3. Forward declarations where possible
- Reduce compilation dependencies
- Faster builds

### 4. Namespace organization
```cpp
namespace artemis {
    namespace core { /* ... */ }
    namespace gpu { /* ... */ }
    namespace behaviors { /* ... */ }
    // etc.
}
```

### 5. Documentation style
- Doxygen-compatible comments
- `@brief`, `@param`, `@return` tags
- Design pattern annotations

## Build System Integration

### CMake Usage
```cmake
find_package(Artemis REQUIRED)
target_link_libraries(my_app PRIVATE Artemis::artemis)
```

### Include Paths
```cpp
#include <artemis/artemis.hpp>           // Main header
#include <artemis/core/simulation.hpp>   // Specific component
```

### Conditional Compilation
```cpp
#ifdef ARTEMIS_HDF5_ENABLED
    // HDF5-specific code
#endif
```

## Performance Considerations

### 1. Compilation Time
- Avoid including `artemis.hpp` in frequently changed files
- Include only needed headers
- Use forward declarations in headers

### 2. GPU Memory
- SoA layout for coalesced access
- Device buffers auto-managed (RAII)
- Minimal host-device transfers

### 3. Kernel Launches
- Declared in `kernels.cuh`
- Implemented in `.cu` files
- Launch from behavior/metric classes

## Future Extensibility

### Planned Additions (v1.5)
- `gpu/multi_gpu_manager.hpp` - Multi-GPU coordination
- `gpu/octree_index.hpp` - Hierarchical spatial indexing
- `visualization/renderer.hpp` - Real-time rendering
- `ml/pytorch_integration.hpp` - ML integration

### Plugin Architecture (v2.0)
- Dynamic behavior loading
- Custom metric plugins
- External visualization backends

## References

- **Style Guide**: Follow C++ Core Guidelines
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Documentation**: Doxygen format
- **CUDA**: Follow CUDA C++ Programming Guide
