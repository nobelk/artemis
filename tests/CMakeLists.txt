# Tests CMakeLists.txt

# Enable testing
enable_testing()

# Add test executable
add_executable(artemis_tests
    test_main.cpp
    test_agent_arrays.cpp
    test_spatial_index.cpp
    test_metrics.cpp
)

target_link_libraries(artemis_tests
    PRIVATE artemis
)

# Add tests
add_test(NAME ArtemisTests COMMAND artemis_tests)

# Set test properties
set_tests_properties(ArtemisTests PROPERTIES
    TIMEOUT 300
)
