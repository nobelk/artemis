/**
 * @file test_spatial_index.cpp
 * @brief Tests for SpatialIndex class
 */

#include <artemis/gpu/spatial_index.hpp>
#include <artemis/core/agent_arrays.hpp>
#include <iostream>

namespace test {
    void assert_true(bool condition, const std::string& message);
    void assert_equal(float a, float b, const std::string& message, float epsilon);
}

void test_spatial_index() {
    using namespace artemis;

    // Test 1: Construction
    {
        BoundingBox2D bounds(float2(0, 0), float2(100, 100));
        gpu::SpatialIndex index(bounds, 10.0f);

        test::assert_equal(index.cell_size(), 10.0f, "Cell size", 1e-5f);
        test::assert_true(index.num_cells() > 0, "Should have cells");
    }

    // Test 2: Rebuild
    {
        BoundingBox2D bounds(float2(0, 0), float2(100, 100));
        gpu::SpatialIndex index(bounds, 10.0f);

        core::AgentArrays agents;
        agents.allocate(100);

        // Initialize with random positions
        agents.initialize_random(bounds);

        // Rebuild index
        index.rebuild(agents);

        // Check statistics
        const auto& stats = index.statistics();
        test::assert_true(stats.num_occupied_cells > 0, "Should have occupied cells");
    }

    // Test 3: Cell ID computation
    {
        BoundingBox2D bounds(float2(0, 0), float2(100, 100));
        gpu::SpatialIndex index(bounds, 10.0f);

        float2 pos(5.0f, 5.0f);  // Should be in cell (0, 0)
        CellID cell_id = index.position_to_cell(pos);

        test::assert_equal(cell_id, 0, "Cell ID for (5,5)", 0.0f);
    }

    std::cout << "  All SpatialIndex tests passed\n";
}
