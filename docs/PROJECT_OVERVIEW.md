# Artemis Project Overview

## Project Structure

```
Artemis/
├── CMakeLists.txt                   # Root build configuration
├── README.md                        # Project description
│
├── include/artemis/                 # Public API headers (31 files)
│   ├── artemis.hpp                 # Main convenience header
│   ├── util/                       # 3 headers: types, math, gpu_util
│   ├── core/                       # 4 headers: agents, environment, scheduler, simulation
│   ├── gpu/                        # 3 headers: device_manager, spatial_index, kernels
│   ├── config/                     # 2 headers: simulation_config, yaml_parser
│   ├── behaviors/                  # 5 headers: behavior, manager, boids, predator_prey, social
│   ├── analysis/                   # 2 headers: metrics_engine, pattern_detector
│   └── io/                         # 3 headers: checkpoint_manager, hdf5_writer, data_exporter
│
├── src/                            # Implementation files (to be created)
│   ├── core/                       # simulation.cpp, environment.cpp, scheduler.cpp
│   ├── config/                     # yaml_parser.cpp, simulation_config.cpp
│   ├── analysis/                   # metrics_engine.cpp, pattern_detector.cpp
│   ├── io/                         # checkpoint_manager.cpp, hdf5_writer.cpp
│   └── cli/                        # main.cpp (CLI tool)
│
├── cuda/                           # CUDA implementation files (to be created)
│   └── kernels/                    # agent_update.cu, spatial_index.cu, collision.cu, metrics.cu
│
├── tests/                          # Unit tests (4 files)
│   ├── CMakeLists.txt
│   ├── test_main.cpp
│   ├── test_agent_arrays.cpp
│   ├── test_spatial_index.cpp
│   └── test_metrics.cpp
│
├── examples/                       # Example programs (4 files)
│   ├── CMakeLists.txt
│   ├── basic_boids.cpp
│   ├── config_based.cpp
│   └── configs/
│       └── boids.yaml
│
└── docs/                           # Documentation (4 files)
    ├── ARCHITECTURE.md             # System architecture and design patterns
    ├── API_GUIDE.md                # Complete API usage guide
    ├── HEADER_STRUCTURE.md         # Header organization and dependencies
    └── PROJECT_OVERVIEW.md         # This file

```

## Statistics

### Header Files: 31 total
- Utility layer: 3 files
- Core simulation: 4 files
- GPU layer: 3 files
- Configuration: 2 files
- Behaviors: 5 files
- Analysis: 2 files
- I/O: 3 files
- Main header: 1 file

### Design Patterns: 10+ implemented
1. **Strategy** - Pluggable behaviors
2. **Factory** - Behavior creation
3. **Builder** - Configuration construction
4. **Facade** - Simulation interface
5. **Observer** - Metrics updates
6. **RAII** - GPU resource management
7. **Memento** - State checkpointing
8. **Composite** - Behavior composition
9. **Chain of Responsibility** - Behavior manager
10. **Template Method** - Behavior execution

### Code Organization
- **Clean Architecture**: Layered design with clear dependencies
- **SOLID Principles**:
  - Single Responsibility: Each class has one purpose
  - Open/Closed: Extensible via interfaces
  - Liskov Substitution: Behaviors are interchangeable
  - Interface Segregation: Focused interfaces
  - Dependency Inversion: High-level code doesn't depend on low-level

## Key Features

### GPU-First Design
- Structure of Arrays (SoA) for coalesced memory access
- RAII wrappers for automatic GPU memory management
- Minimal CPU↔GPU transfers
- CUDA kernels for all compute-intensive operations

### Behavior System
- Pluggable behavior strategies
- Built-in behaviors: Boids, Predator-Prey, Social
- Easy to extend with custom behaviors
- Composable behaviors

### Analysis Pipeline
- Real-time GPU metrics computation
- Pattern detection (clustering, phase transitions, waves)
- Time series collection
- Export to HDF5, CSV, JSON

### Configuration System
- YAML/JSON support
- Programmatic API
- Builder pattern for construction
- Validation and error reporting

### I/O System
- Checkpointing with auto-rotation
- HDF5 time series export
- CSV/JSON data export
- Compression support

## Build System

### Dependencies
- **Required**:
  - CUDA Toolkit 12.x
  - C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
  - CMake 3.18+
  - NVIDIA GPU (compute capability 7.5+)

- **Optional**:
  - HDF5 (for time series export)
  - GLFW + OpenGL (for visualization)
  - Python 3.8+ (for Python bindings)

### Build Commands
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest                # Run tests
make install         # Install
```

## API Levels

### Level 1: Quick Start (Beginner)
```cpp
#include <artemis/artemis.hpp>

artemis::initialize();
artemis::core::Simulation sim;
sim.initialize(10000, bounds);
sim.run(1000);
artemis::shutdown();
```

### Level 2: Configuration (Intermediate)
```yaml
# config.yaml
agent_types:
  - name: "boid"
    count: 10000
behaviors:
  - type: "boids"
```

```cpp
auto config = artemis::config::YAMLParser::load_from_file("config.yaml");
sim.initialize(config);
sim.run(config.temporal.max_steps);
```

### Level 3: Custom Behaviors (Advanced)
```cpp
class MyBehavior : public artemis::behaviors::Behavior {
    void execute(core::AgentArrays& agents, /*...*/) override {
        // Launch custom CUDA kernel
    }
};
REGISTER_BEHAVIOR(MyBehavior, "my_behavior");
```

### Level 4: Custom Kernels (Expert)
```cuda
__global__ void my_kernel(float2* positions, /*...*/) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Custom GPU computation
}
```

## Performance Targets

Based on README specifications:

| Agents | Target FPS | GPU Requirement |
|--------|-----------|-----------------|
| 10K    | 1000+     | RTX 2060+       |
| 100K   | 100+      | RTX 3070+       |
| 1M     | 30+       | RTX 4090        |
| 10M    | 5+        | A100            |

## Development Workflow

### Phase 1: Headers (COMPLETED ✅)
- ✅ Project structure
- ✅ All header files (31 files)
- ✅ Build system (CMake)
- ✅ Examples (2 programs)
- ✅ Tests (4 test files)
- ✅ Documentation (4 docs)

### Phase 2: Implementation (NEXT)
- [ ] Core simulation logic (C++)
- [ ] CUDA kernels (GPU)
- [ ] Configuration parsing (YAML/JSON)
- [ ] I/O implementations (HDF5, CSV, JSON)
- [ ] Behavior implementations

### Phase 3: Testing & Validation
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Validation against known results

### Phase 4: Documentation & Examples
- [ ] API documentation (Doxygen)
- [ ] Tutorial notebooks
- [ ] More examples
- [ ] Performance guide

### Phase 5: Advanced Features (v1.5+)
- [ ] Multi-GPU support
- [ ] Web visualization
- [ ] Python bindings
- [ ] ML integration

## Design Decisions

### Why SoA instead of AoS?
- **GPU Performance**: Coalesced memory access (10x faster)
- **Memory Efficiency**: Better cache utilization
- **Scalability**: Linear scaling to millions of agents

### Why CUDA instead of OpenCL/SYCL?
- **Performance**: Best GPU performance
- **Ecosystem**: Mature tools (Thrust, cuBLAS, etc.)
- **Hardware**: Targeting NVIDIA GPUs (A100, RTX series)

### Why C++ instead of Python?
- **Performance**: GPU kernels need C++/CUDA
- **Control**: Low-level memory management
- **Production**: Deploy as library or CLI tool
- **Note**: Python bindings planned for v1.5

### Why YAML for configuration?
- **Human-readable**: Easy to write and edit
- **Hierarchy**: Natural for nested configs
- **Ecosystem**: Standard in scientific computing

## Extension Points

### Adding New Components

1. **Custom Behavior**
   - Inherit from `Behavior`
   - Implement `execute()` with CUDA kernel
   - Register with macro

2. **Custom Metric**
   - Inherit from `Metric`
   - Implement `compute()` with GPU reduction
   - Register with `MetricsEngine`

3. **Custom Export Format**
   - Inherit from exporter base
   - Implement format-specific methods
   - Add to `DataExporter` factory

4. **Custom Pattern Detector**
   - Inherit from `PatternDetector`
   - Implement `detect()` algorithm
   - Register with `PatternDetectionEngine`

## Quality Standards

### Code Quality
- **RAII**: No manual memory management
- **const-correctness**: Proper const usage
- **Error handling**: Exceptions for errors, CUDA_CHECK for GPU
- **Documentation**: All public APIs documented

### Testing
- **Unit tests**: Each component tested independently
- **Integration tests**: Full pipeline tested
- **Performance tests**: Scaling verified
- **Regression tests**: Prevent performance degradation

### Performance
- **Memory**: <100 bytes per agent
- **Bandwidth**: >80% peak memory bandwidth
- **Occupancy**: >50% GPU occupancy
- **Scaling**: Linear with agent count

## Next Steps

### Immediate (Phase 2)
1. Implement core simulation loop
2. Write basic CUDA kernels
3. Implement boids behavior
4. Test with 10K agents

### Short-term (v1.0)
1. Complete all behaviors
2. Full metrics suite
3. HDF5 export
4. Comprehensive tests
5. Performance validation

### Medium-term (v1.5)
1. Multi-GPU support
2. Web visualization
3. Python bindings
4. Advanced spatial structures

### Long-term (v2.0)
1. Distributed simulation
2. ML integration
3. Cloud deployment
4. Community ecosystem

## Resources

### Documentation
- `docs/ARCHITECTURE.md` - System design
- `docs/API_GUIDE.md` - Usage examples
- `docs/HEADER_STRUCTURE.md` - Code organization

### Examples
- `examples/basic_boids.cpp` - Simple programmatic usage
- `examples/config_based.cpp` - YAML configuration
- `examples/configs/boids.yaml` - Configuration template

### Learning Path
1. Read README.md for overview
2. Read ARCHITECTURE.md for design
3. Try basic_boids.cpp example
4. Read API_GUIDE.md for details
5. Implement custom behavior

## Contributing

### Areas for Contribution
- **Core Development**: CUDA kernels, optimizations
- **Behaviors**: New agent behaviors
- **Analysis**: Pattern detection algorithms
- **Visualization**: Rendering, web interface
- **Documentation**: Tutorials, examples
- **Testing**: Unit tests, benchmarks

### Development Environment
- Linux (Ubuntu 20.04+) or Windows 10+
- NVIDIA GPU (RTX 2060+)
- CUDA Toolkit 12.x
- C++17 compiler
- CMake 3.18+

## License

Apache 2.0 (per README.md)

## Contact

- GitHub Issues: Bug reports and feature requests
- Discussions: Community forum
- Email: artemis-dev@example.com

---

**Status**: Headers complete ✅ | Implementation in progress 🚧 | v1.0 target Q2 2024
