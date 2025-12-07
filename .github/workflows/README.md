# GitHub Actions Workflows

## Overview

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### 1. CI Pipeline (`ci.yml`)

**Purpose:** Build, test, and validate code on every push and pull request

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs:**

#### `build-and-test`
Builds and tests the project on Ubuntu 22.04 with CUDA support.

**Dependencies Installed:**
```
CUDA 12.2 (nvcc, cudart, thrust)
  ↓
System Tools (build-essential, cmake, ninja, pkg-config)
  ↓
GCC 11 (C++17 compiler)
  ↓
Optional Libraries (HDF5)
  ↓
Verification (all tools checked)
  ↓
Build & Test
```

**Steps:**
1. ✅ Checkout code
2. ✅ Install CUDA Toolkit 12.2
3. ✅ Install system dependencies
4. ✅ Setup GCC 11 compiler
5. ✅ Verify CUDA installation
6. ✅ **NEW:** Verify all dependencies
7. ✅ Display system information
8. ✅ Check dependencies with Makefile
9. ✅ Configure CMake
10. ✅ Build project
11. ✅ Run tests
12. ✅ Build examples
13. ✅ Archive artifacts

#### `code-quality`
Checks code formatting and runs static analysis.

**Tools:**
- `clang-format` - Code formatting
- `cppcheck` - Static analysis

#### `documentation`
Validates documentation files and YAML configs.

**Checks:**
- All docs files present
- YAML files valid

#### `build-status`
Aggregates status from all jobs.

### 2. Release Pipeline (`release.yml`)

**Purpose:** Create releases when version tags are pushed

**Triggers:**
- Tags matching `v*.*.*` pattern
- Manual workflow dispatch

**Jobs:**

#### `create-release`
Creates GitHub release with auto-generated notes.

#### `build-release-artifacts`
Builds and packages release artifacts.

**Artifacts:**
- `artemis-headers.tar.gz` - All header files
- `artemis-docs.tar.gz` - Documentation
- `artemis-examples.tar.gz` - Example code

## Dependency Verification

All dependencies are verified before build:

```bash
CMake: cmake version 3.25+
GCC: gcc (Ubuntu 11.4.0) 11.4.0
G++: g++ (Ubuntu 11.4.0) 11.4.0
NVCC: Cuda compilation tools, release 12.2
Make: GNU Make 4.3
Ninja: 1.10.1
pkg-config: 0.29.2
HDF5: 1.10.7 (optional)
```

## Build Matrix

| OS | CUDA | GCC | Status |
|----|-|-----|--------|
| ubuntu-22.04 | 12.2 | 11 | ✅ Active |

## Caching

No caching currently implemented. Future optimization:
- Cache apt packages
- Cache CUDA installation
- Cache build artifacts

## Secrets Required

None currently required. For future:
- `GITHUB_TOKEN` (automatic, for releases)

## Local Testing

Test the CI pipeline locally:
```bash
make ci
```

## Modifying Workflows

### Adding Dependencies

Add to `Install system dependencies` step:
```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y \
      build-essential \
      your-new-package \
      ...
```

### Adding Verification

Add to `Verify all dependencies` step:
```yaml
echo "YourTool: $(your-tool --version)"
```

### Changing Build Options

Modify `Configure CMake` step:
```yaml
- name: Configure CMake
  run: make configure BUILD_TYPE=Debug ENABLE_HDF5=OFF
```

## Troubleshooting

### Build Fails

1. Check dependency verification step
2. Ensure all required tools installed
3. Check CUDA compatibility

### Tests Fail

- Expected in CI (no GPU hardware)
- Tests marked as `continue-on-error: true`

### Artifacts Not Uploaded

- Check build completed successfully
- Verify path patterns correct

## Documentation

- [DEPENDENCY_ANALYSIS.md](DEPENDENCY_ANALYSIS.md) - Dependency requirements
- [CI_DEPENDENCY_VERIFICATION.md](CI_DEPENDENCY_VERIFICATION.md) - Verification report

## Status Badges

Add to README:
```markdown
[![CI](https://github.com/username/artemis/workflows/CI/badge.svg)](https://github.com/username/artemis/actions)
```

## Future Improvements

- [ ] Add caching for faster builds
- [ ] Multi-GPU test support
- [ ] Windows build support
- [ ] macOS build (CPU-only)
- [ ] Code coverage reports
- [ ] Performance benchmarks
- [ ] Docker image publishing
