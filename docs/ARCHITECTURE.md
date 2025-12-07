# Artemis Architecture Documentation

## Overview

Artemis is structured as a layered architecture with clear separation of concerns. Each layer has well-defined responsibilities and interfaces.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│              (CLI Tools, Python Bindings)                   │
├─────────────────────────────────────────────────────────────┤
│                  Configuration Layer                        │
│           (YAML/JSON Parsing, Config Builders)              │
├─────────────────────────────────────────────────────────────┤
│                   Simulation Core                           │
│        (Simulation, Environment, Scheduler)                 │
├─────────────────────────────────────────────────────────────┤
│          Behavior System    │    Analysis Pipeline          │
│       (Strategy Pattern)    │  (Metrics, Patterns)          │
├─────────────────────────────┼───────────────────────────────┤
│                  GPU Computation Layer                      │
│         (Spatial Index, CUDA Kernels)                       │
├─────────────────────────────────────────────────────────────┤
│                    I/O System                               │
│         (HDF5, Checkpoints, Export)                         │
├─────────────────────────────────────────────────────────────┤
│                  Utility Layer                              │
│         (Types, Math, GPU Utilities)                        │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
include/artemis/
├── artemis.hpp              # Main convenience header
├── util/                    # Utility layer
│   ├── types.hpp           # Core types and data structures
│   ├── math.hpp            # Math utilities and vector operations
│   └── gpu_util.hpp        # GPU memory management and utilities
├── core/                    # Core simulation
│   ├── agent_arrays.hpp    # Structure of Arrays for agents
│   ├── environment.hpp     # Spatial environment management
│   ├── scheduler.hpp       # Temporal scheduling
│   └── simulation.hpp      # Main simulation orchestrator
├── gpu/                     # GPU computation layer
│   ├── device_manager.hpp  # GPU device management
│   ├── spatial_index.hpp   # Spatial indexing structures
│   └── kernels.cuh         # CUDA kernel declarations
├── config/                  # Configuration system
│   ├── simulation_config.hpp  # Configuration structures
│   └── yaml_parser.hpp     # YAML/JSON parsing
├── behaviors/               # Behavior system
│   ├── behavior.hpp        # Base behavior interface
│   ├── behavior_manager.hpp   # Behavior orchestration
│   ├── boids_behavior.hpp  # Flocking behavior
│   ├── predator_prey_behavior.hpp  # Predator-prey
│   └── social_behavior.hpp # Social dynamics
├── analysis/                # Analysis pipeline
│   ├── metrics_engine.hpp  # Metrics computation
│   └── pattern_detector.hpp   # Pattern detection
└── io/                      # I/O system
    ├── checkpoint_manager.hpp  # State persistence
    ├── hdf5_writer.hpp     # HDF5 export
    └── data_exporter.hpp   # CSV/JSON export
```

## Design Patterns Used

### 1. **Structure of Arrays (SoA)** - Data-Oriented Design
- **Location**: `core/agent_arrays.hpp`
- **Purpose**: Optimize GPU memory access patterns
- **Benefit**: Coalesced memory reads, 10x bandwidth improvement

### 2. **Strategy Pattern** - Pluggable Behaviors
- **Location**: `behaviors/behavior.hpp`
- **Purpose**: Allow runtime selection of agent behaviors
- **Benefit**: Extensible, testable, composable behaviors

### 3. **Factory Pattern** - Object Creation
- **Location**: `behaviors/behavior.hpp`, `config/yaml_parser.hpp`
- **Purpose**: Centralized creation of behaviors and configurations
- **Benefit**: Decoupled construction from usage

### 4. **Facade Pattern** - Simplified Interfaces
- **Location**: `core/simulation.hpp`, `io/hdf5_writer.hpp`
- **Purpose**: Hide complex subsystem interactions
- **Benefit**: Simple API for users, internal flexibility

### 5. **Observer Pattern** - Metrics Updates
- **Location**: `analysis/metrics_engine.hpp`
- **Purpose**: Automatic metric computation on state changes
- **Benefit**: Decoupled analysis from simulation logic

### 6. **RAII** - Resource Management
- **Location**: `util/gpu_util.hpp` (DeviceBuffer, Stream, Event)
- **Purpose**: Automatic GPU memory management
- **Benefit**: Exception-safe, no leaks

### 7. **Memento Pattern** - State Persistence
- **Location**: `io/checkpoint_manager.hpp`
- **Purpose**: Save and restore simulation state
- **Benefit**: Reproducibility, fault tolerance

### 8. **Builder Pattern** - Configuration Construction
- **Location**: `config/simulation_config.hpp`
- **Purpose**: Fluent interface for building configs
- **Benefit**: Readable, flexible configuration

### 9. **Template Method** - Algorithm Framework
- **Location**: `behaviors/behavior.hpp`
- **Purpose**: Define behavior execution skeleton
- **Benefit**: Consistent interface, customizable steps

### 10. **Singleton Pattern** - Global State
- **Location**: `gpu/device_manager.hpp`
- **Purpose**: Single GPU manager instance
- **Benefit**: Centralized GPU resource coordination

## Core Architectural Principles

### 1. **GPU-First Design**
Everything is optimized for GPU execution:
- SoA memory layout for coalesced access
- Minimal CPU↔GPU transfers
- Kernel fusion where beneficial
- Zero-copy operations where possible

### 2. **Separation of Concerns**
Each component has a single responsibility:
- Core: Simulation orchestration
- Behaviors: Agent logic
- Analysis: Metrics and patterns
- I/O: Persistence and export
- Config: Initialization

### 3. **Composition Over Inheritance**
Prefer composition for flexibility:
- Behaviors are composed, not inherited
- Metrics are registered, not subclassed (mostly)
- Simulation owns components, not extends them

### 4. **Dependency Inversion**
High-level modules don't depend on low-level:
- Simulation depends on interfaces, not implementations
- Behaviors are abstract strategies
- I/O formats are pluggable

### 5. **Interface Segregation**
Keep interfaces focused:
- Each behavior defines minimal interface
- Metrics expose only compute() method
- Exporters have format-specific interfaces

## Data Flow

### Typical Simulation Step

```
1. Scheduler::step()
   └─> Increment timestep

2. SpatialIndex::rebuild(agents)
   ├─> Compute Morton codes (GPU)
   ├─> Sort agents by code (GPU)
   └─> Build cell boundaries (GPU)

3. BehaviorManager::execute_all(agents, spatial_index, dt)
   ├─> For each behavior:
   │   └─> Behavior::execute(agents, spatial_index, dt)
   │       ├─> Query neighbors (GPU)
   │       ├─> Compute forces (GPU)
   │       └─> Update velocities (GPU)

4. Physics integration (GPU)
   ├─> Integrate velocities → positions
   └─> Apply velocity constraints

5. Environment::apply_boundary_conditions(agents)
   └─> Wrap/reflect positions (GPU)

6. MetricsEngine::compute_all(agents, timestep)
   ├─> For each metric:
   │   └─> Metric::compute(agents)
   │       └─> GPU reduction/analysis
   └─> Store in time series

7. Scheduler::process_events()
   └─> Execute scheduled callbacks
```

## Memory Management Strategy

### Host (CPU) Memory
- Configuration data
- Metric time series (small)
- File I/O buffers

### Device (GPU) Memory
- Agent arrays (SoA layout)
- Spatial index structures
- Temporary computation buffers
- Reduction results

### Transfer Minimization
- Keep data on GPU between steps
- Only transfer for I/O or analysis
- Use GPU reduction for statistics
- Direct GPU→File for checkpoints (when possible)

## Extension Points

### Adding a New Behavior
1. Inherit from `Behavior`
2. Implement `execute()` method
3. Define parameters structure
4. Register with `REGISTER_BEHAVIOR` macro
5. Add CUDA kernel if needed

### Adding a New Metric
1. Inherit from `Metric`
2. Implement `compute()` method
3. Write GPU kernel for computation
4. Register with `MetricsEngine`

### Adding a New Export Format
1. Inherit from appropriate exporter base
2. Implement format-specific methods
3. Add to `DataExporter` factory

## Performance Considerations

### Memory Layout
- SoA for GPU (coalesced access)
- AoS for CPU (cache locality)
- Convert at CPU↔GPU boundary

### Kernel Launch
- Use occupancy calculator for block size
- Minimize kernel launches (fusion)
- Stream parallel work when possible

### Spatial Indexing
- Cell size = 2× interaction radius
- Morton codes for cache coherence
- Rebuild only when necessary

### Analysis
- Compute metrics on GPU
- Reduce before CPU transfer
- Batch metric computation

## Testing Strategy

### Unit Tests
- Each component tested independently
- Mock GPU operations for CPU tests
- Validate GPU kernels with small inputs

### Integration Tests
- Full simulation pipeline
- Configuration loading
- I/O round-trips

### Performance Tests
- Scaling with agent count
- Kernel benchmarks
- Memory bandwidth tests

## Future Enhancements

### Version 1.5 (Planned)
- Multi-GPU support (domain decomposition)
- Advanced spatial structures (octree)
- Machine learning integration
- Web visualization

### Version 2.0 (Vision)
- Distributed simulation (multi-node)
- Differentiable simulation
- Reinforcement learning environments
- Cloud deployment

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Design Patterns (Gang of Four)](https://en.wikipedia.org/wiki/Design_Patterns)
- [Data-Oriented Design](https://www.dataorienteddesign.com/)
- [GPU Gems 3 - Parallel Algorithms](https://developer.nvidia.com/gpugems/gpugems3/)
