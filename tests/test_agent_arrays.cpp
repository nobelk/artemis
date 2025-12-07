/**
 * @file test_agent_arrays.cpp
 * @brief Tests for AgentArrays class
 */

#include <artemis/core/agent_arrays.hpp>
#include <iostream>

// External test framework (from test_main.cpp)
namespace test {
    void assert_true(bool condition, const std::string& message);
    void assert_equal(float a, float b, const std::string& message, float epsilon);
}

void test_agent_arrays() {
    using namespace artemis;

    // Test 1: Construction and allocation
    {
        core::AgentArrays agents;
        test::assert_true(agents.empty(), "New AgentArrays should be empty");
        test::assert_equal(agents.size(), 0, "Size should be 0", 0.0f);
    }

    // Test 2: Allocation
    {
        core::AgentArrays agents;
        const size_t num_agents = 1000;
        agents.allocate(num_agents);

        test::assert_equal(agents.size(), num_agents, "Size after allocation", 0.0f);
        test::assert_true(agents.positions() != nullptr, "Positions should be allocated");
        test::assert_true(agents.velocities() != nullptr, "Velocities should be allocated");
    }

    // Test 3: Upload/Download
    {
        core::AgentArrays agents;
        const size_t num_agents = 100;

        std::vector<AgentState> host_agents(num_agents);
        for (size_t i = 0; i < num_agents; ++i) {
            host_agents[i].position = float2(i * 1.0f, i * 2.0f);
            host_agents[i].velocity = float2(1.0f, 0.0f);
            host_agents[i].energy = 100.0f;
        }

        agents.upload_from_host(host_agents);
        test::assert_equal(agents.size(), num_agents, "Size after upload", 0.0f);

        std::vector<AgentState> downloaded(num_agents);
        agents.download_to_host(downloaded);

        // Verify first agent
        test::assert_equal(downloaded[0].position.x, 0.0f, "Position[0].x", 1e-5f);
        test::assert_equal(downloaded[0].position.y, 0.0f, "Position[0].y", 1e-5f);
        test::assert_equal(downloaded[0].energy, 100.0f, "Energy[0]", 1e-5f);
    }

    std::cout << "  All AgentArrays tests passed\n";
}
