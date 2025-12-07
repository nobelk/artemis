# Build Instructions

## Quick Start

### Using Makefile (Recommended)

```bash
# 1. Check dependencies
make deps-check

# 2. Display build info
make info

# 3. Build and test
make ci

# Or step by step:
make build
make test
make examples
```

### Using CMake Directly

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest
```

## Build Requirements

### Minimum Requirements

| Component | Version | Required |
|-----------|---------|----------|
| **CMake** | 3.18+ | ✅ Yes |
| **C++ Compiler** | GCC 9+, Clang 10+, MSVC 2019+ | ✅ Yes |
| **CUDA Toolkit** | 12.x (11.0+ supported) | ✅ Yes |
| **NVIDIA GPU** | Compute 7.5+ (RTX 2000+) | ✅ Yes |
| **Make** | GNU Make or compatible | ✅ Yes |
| **HDF5** | Any recent version | ⚠️ Optional |
| **GLFW** | 3.x | ⚠️ Optional |
| **Python** | 3.8+ | ⚠️ Optional |

### Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully Supported | Ubuntu 20.04+, RHEL 8+, Fedora 35+ |
| **Windows** | ✅ Supported | Windows 10/11 with Visual Studio 2019+ |
| **macOS** | ⚠️ Limited | No NVIDIA GPU support in recent macOS |

## Makefile Targets

### Build Targets

```bash
make build          # Build the project (Release by default)
make debug          # Build with debug symbols
make release        # Build optimized release
make rebuild        # Clean and rebuild
```

### Test Targets

```bash
make test           # Run all tests
make test-verbose   # Run tests with verbose output
```

### Example Targets

```bash
make examples           # Build examples
make run-basic-boids    # Run basic boids example
make run-config-boids   # Run config-based example
```

### Maintenance Targets

```bash
make clean          # Remove build directory
make clean-all      # Remove all generated files
make format         # Format code with clang-format
make lint           # Run static analysis
```

### Information Targets

```bash
make help           # Show all available commands
make info           # Display build configuration
make version        # Display Artemis version
make deps-check     # Check for required dependencies
```

## Build Options

### CMake Options

```bash
# Enable/disable tests
cmake .. -DBUILD_TESTS=ON

# Enable/disable examples
cmake .. -DBUILD_EXAMPLES=ON

# Enable/disable HDF5 support
cmake .. -DENABLE_HDF5=ON

# Enable/disable visualization
cmake .. -DBUILD_VISUALIZATION=ON

# Specify CUDA architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"

# Specify build type
cmake .. -DCMAKE_BUILD_TYPE=Release  # or Debug, RelWithDebInfo
```

### Makefile Options

```bash
# Specify build type
make build BUILD_TYPE=Debug

# Specify number of parallel jobs
make build JOBS=8

# Specify build directory
make build BUILD_DIR=my_build

# Enable/disable options
make build ENABLE_HDF5=OFF ENABLE_TESTS=ON
```

## Build Troubleshooting

### CUDA Not Found

**Problem:** CMake cannot find CUDA Toolkit

**Solution:**
```bash
# Set CUDA path explicitly
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Then try again
make configure
```

### HDF5 Not Found

**Problem:** CMake cannot find HDF5 library

**Solution:**
```bash
# Install HDF5
# Ubuntu/Debian
sudo apt-get install libhdf5-dev

# Fedora/RHEL
sudo dnf install hdf5-devel

# macOS
brew install hdf5

# Or disable HDF5
make build ENABLE_HDF5=OFF
```

### Compiler Version Too Old

**Problem:** C++ compiler doesn't support C++17

**Solution:**
```bash
# Ubuntu/Debian - install newer GCC
sudo apt-get install gcc-11 g++-11
export CXX=g++-11
export CC=gcc-11

# Or use Clang
sudo apt-get install clang-14
export CXX=clang++-14
export CC=clang-14
```

### CUDA Architecture Mismatch

**Problem:** Compiled code doesn't work on your GPU

**Solution:**
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Set appropriate architecture (e.g., 8.6 for RTX 3090)
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"
```

## Build Verification

### Quick Verification

```bash
# Check dependencies
make deps-check

# Display configuration
make info

# Try to configure
make configure

# If successful, build
make build
```

### Full Verification

```bash
# Run full CI pipeline
make ci

# This will:
# 1. Clean build directory
# 2. Configure CMake
# 3. Build project
# 4. Run tests
```

## Platform-Specific Instructions

### Ubuntu 22.04 / Debian 12

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libhdf5-dev \
    pkg-config

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-2

# Build
make ci
```

### Fedora 38+ / RHEL 9

```bash
# Install dependencies
sudo dnf install -y \
    gcc-c++ \
    cmake \
    ninja-build \
    hdf5-devel

# Install CUDA
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install -y cuda-toolkit-12-2

# Build
make ci
```

### Windows 10/11

```cmd
# Install Visual Studio 2019 or 2022 with C++ support
# Install CUDA Toolkit from NVIDIA website
# Install CMake from cmake.org

# Open "x64 Native Tools Command Prompt for VS 2022"
cd artemis
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Or use nmake
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
nmake
```

## Development Build

For development with debugging:

```bash
# Setup development build
make dev

# This configures with:
# - BUILD_TYPE=Debug
# - Debug symbols enabled
# - Tests enabled
# - Assertions enabled
```

## Performance Build

For maximum performance:

```bash
# Build optimized release
make release

# This configures with:
# - BUILD_TYPE=Release
# - Full optimizations (-O3, -march=native)
# - Assertions disabled
# - Fast math enabled
```

## Cross-Compilation

For building on different architectures:

```bash
# Specify target architecture
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -DCMAKE_CXX_FLAGS="-march=haswell"

make build
```

## CI/CD Integration

### GitHub Actions

The project includes GitHub Actions workflows:

- `.github/workflows/ci.yml` - Main CI pipeline
- `.github/workflows/release.yml` - Release automation

To run CI locally:

```bash
make ci
```

### Docker Build

```bash
# Build Docker image
make docker-build

# Run tests in Docker
make docker-test
```

## Next Steps

After successful build:

1. Run tests: `make test`
2. Try examples: `make run-basic-boids`
3. Read documentation: `docs/API_GUIDE.md`
4. Write your first simulation!

## Getting Help

If you encounter build issues:

1. Check this document
2. Run `make deps-check` to verify dependencies
3. Run `make info` to see configuration
4. Open an issue on GitHub with build logs
5. Join our Discord community

---

**Build Status:** [![CI](https://github.com/yourusername/artemis/workflows/CI/badge.svg)](https://github.com/yourusername/artemis/actions)
