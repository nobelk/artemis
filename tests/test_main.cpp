/**
 * @file test_main.cpp
 * @brief Main test runner
 *
 * This file contains the main function for running all Artemis tests.
 * Uses a simple test framework (can be replaced with Google Test, Catch2, etc.)
 */

#include <iostream>
#include <vector>
#include <functional>
#include <string>

// Simple test framework
namespace test {

struct Test {
    std::string name;
    std::function<void()> func;
};

std::vector<Test>& get_tests() {
    static std::vector<Test> tests;
    return tests;
}

void register_test(const std::string& name, std::function<void()> func) {
    get_tests().push_back({name, func});
}

void assert_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Assertion failed: " + message);
    }
}

void assert_equal(float a, float b, const std::string& message, float epsilon = 1e-5f) {
    if (std::abs(a - b) > epsilon) {
        throw std::runtime_error("Assertion failed: " + message +
                               " (got " + std::to_string(a) +
                               ", expected " + std::to_string(b) + ")");
    }
}

} // namespace test

// Test declarations (implemented in separate files)
void test_agent_arrays();
void test_spatial_index();
void test_metrics();

int main() {
    std::cout << "Running Artemis Tests\n";
    std::cout << "=====================\n\n";

    // Register tests
    test::register_test("AgentArrays", test_agent_arrays);
    test::register_test("SpatialIndex", test_spatial_index);
    test::register_test("Metrics", test_metrics);

    // Run all tests
    int passed = 0;
    int failed = 0;

    for (const auto& test : test::get_tests()) {
        std::cout << "Running test: " << test.name << "... ";
        try {
            test.func();
            std::cout << "PASSED\n";
            passed++;
        } catch (const std::exception& e) {
            std::cout << "FAILED\n";
            std::cout << "  Error: " << e.what() << "\n";
            failed++;
        }
    }

    std::cout << "\n=====================\n";
    std::cout << "Tests passed: " << passed << "\n";
    std::cout << "Tests failed: " << failed << "\n";

    return (failed == 0) ? 0 : 1;
}
