# Examples CMakeLists.txt

# Basic boids example
add_executable(example_basic_boids
    basic_boids.cpp
)

target_link_libraries(example_basic_boids
    PRIVATE artemis
)

# Config-based example
add_executable(example_config_based
    config_based.cpp
)

target_link_libraries(example_config_based
    PRIVATE artemis
)

# Install examples
install(TARGETS example_basic_boids example_config_based
    RUNTIME DESTINATION bin/examples
)

# Copy example config files
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/boids.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/boids.yaml
    COPYONLY
)
