cmake_minimum_required(VERSION 3.18)

# Check for CUDA availability
include(CheckLanguage)
check_language(CUDA)

# Project with conditional CUDA support
if(CMAKE_CUDA_COMPILER)
    project(Artemis
        VERSION 1.0.0
        DESCRIPTION "GPU-Accelerated Multiagent Simulation Platform"
        LANGUAGES CXX CUDA
    )
    set(ARTEMIS_CUDA_AVAILABLE ON)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    project(Artemis
        VERSION 1.0.0
        DESCRIPTION "GPU-Accelerated Multiagent Simulation Platform"
        LANGUAGES CXX
    )
    set(ARTEMIS_CUDA_AVAILABLE OFF)
    message(WARNING "CUDA compiler not found - building without GPU support (headers only)")
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard (if available)
if(ARTEMIS_CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_VISUALIZATION "Build visualization tools" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(ENABLE_HDF5 "Enable HDF5 support" ON)

# Find packages
if(ARTEMIS_CUDA_AVAILABLE)
    find_package(CUDAToolkit REQUIRED)
endif()

if(ENABLE_HDF5)
    find_package(HDF5 COMPONENTS CXX)
endif()

if(BUILD_VISUALIZATION)
    find_package(OpenGL)
    find_package(glfw3)
endif()

# CUDA architecture detection (only if CUDA is available)
if(ARTEMIS_CUDA_AVAILABLE)
    # Set CUDA architectures (default to common architectures)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(ARTEMIS_CUDA_AVAILABLE)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math -DNDEBUG")
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

if(ARTEMIS_CUDA_AVAILABLE)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Source files
set(ARTEMIS_CORE_SOURCES
    src/core/simulation.cpp
    src/core/environment.cpp
    src/core/scheduler.cpp
)

if(ARTEMIS_CUDA_AVAILABLE)
    set(ARTEMIS_GPU_SOURCES
        cuda/kernels/agent_update.cu
        cuda/kernels/spatial_index.cu
        cuda/kernels/collision.cu
        cuda/kernels/metrics.cu
    )
else()
    set(ARTEMIS_GPU_SOURCES "")
endif()

set(ARTEMIS_CONFIG_SOURCES
    src/config/yaml_parser.cpp
    src/config/simulation_config.cpp
)

set(ARTEMIS_ANALYSIS_SOURCES
    src/analysis/metrics_engine.cpp
    src/analysis/pattern_detector.cpp
)

set(ARTEMIS_IO_SOURCES
    src/io/checkpoint_manager.cpp
)

if(ENABLE_HDF5 AND HDF5_FOUND)
    list(APPEND ARTEMIS_IO_SOURCES src/io/hdf5_writer.cpp)
endif()

# Main library
add_library(artemis STATIC
    ${ARTEMIS_CORE_SOURCES}
    ${ARTEMIS_GPU_SOURCES}
    ${ARTEMIS_CONFIG_SOURCES}
    ${ARTEMIS_ANALYSIS_SOURCES}
    ${ARTEMIS_IO_SOURCES}
)

if(ARTEMIS_CUDA_AVAILABLE)
    target_link_libraries(artemis
        PUBLIC
            CUDA::cudart
            CUDA::cuda_driver
    )
    target_compile_definitions(artemis PUBLIC ARTEMIS_CUDA_ENABLED)
endif()

if(ENABLE_HDF5 AND HDF5_FOUND)
    target_link_libraries(artemis PUBLIC ${HDF5_CXX_LIBRARIES})
    target_include_directories(artemis PUBLIC ${HDF5_INCLUDE_DIRS})
    target_compile_definitions(artemis PUBLIC ARTEMIS_HDF5_ENABLED)
endif()

# CLI tool
add_executable(artemis_cli
    src/cli/main.cpp
)

target_link_libraries(artemis_cli
    PRIVATE artemis
)

# Testing
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS artemis artemis_cli
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/artemis
    DESTINATION include
)

# Print configuration summary
message(STATUS "")
message(STATUS "Artemis Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Available: ${ARTEMIS_CUDA_AVAILABLE}")
if(ARTEMIS_CUDA_AVAILABLE)
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  HDF5 Support: ${ENABLE_HDF5}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "")
