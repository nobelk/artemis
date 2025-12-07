#pragma once

#include <cstdint>
#include <cmath>

#ifdef ARTEMIS_CUDA_ENABLED
#include <cuda_runtime.h>
#define ARTEMIS_HOST_DEVICE __host__ __device__
#else
#define ARTEMIS_HOST_DEVICE
#endif

namespace artemis {

// ============================================================================
// Basic Types
// ============================================================================

using AgentID = uint32_t;
using CellID = uint32_t;
using MortonCode = uint64_t;
using TimeStep = uint64_t;

// ============================================================================
// Vector Types (GPU-compatible)
// ============================================================================

struct float2 {
    float x, y;

    ARTEMIS_HOST_DEVICE float2() : x(0.0f), y(0.0f) {}
    ARTEMIS_HOST_DEVICE float2(float x_, float y_) : x(x_), y(y_) {}

    ARTEMIS_HOST_DEVICE float2 operator+(const float2& other) const {
        return float2(x + other.x, y + other.y);
    }

    ARTEMIS_HOST_DEVICE float2 operator-(const float2& other) const {
        return float2(x - other.x, y - other.y);
    }

    ARTEMIS_HOST_DEVICE float2 operator*(float scalar) const {
        return float2(x * scalar, y * scalar);
    }

    ARTEMIS_HOST_DEVICE float length() const {
        return sqrtf(x * x + y * y);
    }

    ARTEMIS_HOST_DEVICE float length_squared() const {
        return x * x + y * y;
    }

    ARTEMIS_HOST_DEVICE float2 normalized() const {
        float len = length();
        if (len > 1e-6f) {
            return float2(x / len, y / len);
        }
        return float2(0.0f, 0.0f);
    }
};

struct float3 {
    float x, y, z;

    ARTEMIS_HOST_DEVICE float3() : x(0.0f), y(0.0f), z(0.0f) {}
    ARTEMIS_HOST_DEVICE float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    ARTEMIS_HOST_DEVICE float3 operator+(const float3& other) const {
        return float3(x + other.x, y + other.y, z + other.z);
    }

    ARTEMIS_HOST_DEVICE float3 operator-(const float3& other) const {
        return float3(x - other.x, y - other.y, z - other.z);
    }

    ARTEMIS_HOST_DEVICE float3 operator*(float scalar) const {
        return float3(x * scalar, y * scalar, z * scalar);
    }

    ARTEMIS_HOST_DEVICE float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    ARTEMIS_HOST_DEVICE float length_squared() const {
        return x * x + y * y + z * z;
    }

    ARTEMIS_HOST_DEVICE float3 normalized() const {
        float len = length();
        if (len > 1e-6f) {
            return float3(x / len, y / len, z / len);
        }
        return float3(0.0f, 0.0f, 0.0f);
    }
};

// ============================================================================
// Agent Type Enumeration
// ============================================================================

enum class AgentType : uint8_t {
    BOID = 0,
    PREDATOR = 1,
    PREY = 2,
    SOCIAL_AGENT = 3,
    CUSTOM = 255
};

// ============================================================================
// Agent State (Structure of Arrays friendly)
// ============================================================================

struct AgentState {
    float2 position;
    float2 velocity;
    AgentType type;
    float energy;
    uint32_t age;

    ARTEMIS_HOST_DEVICE AgentState()
        : position(0.0f, 0.0f)
        , velocity(0.0f, 0.0f)
        , type(AgentType::BOID)
        , energy(100.0f)
        , age(0) {}
};

// ============================================================================
// Spatial Bounds
// ============================================================================

struct BoundingBox2D {
    float2 min;
    float2 max;

    ARTEMIS_HOST_DEVICE BoundingBox2D()
        : min(0.0f, 0.0f), max(1.0f, 1.0f) {}

    ARTEMIS_HOST_DEVICE BoundingBox2D(float2 min_, float2 max_)
        : min(min_), max(max_) {}

    ARTEMIS_HOST_DEVICE float2 size() const {
        return float2(max.x - min.x, max.y - min.y);
    }

    ARTEMIS_HOST_DEVICE float2 center() const {
        return float2((min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f);
    }

    ARTEMIS_HOST_DEVICE bool contains(const float2& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y;
    }
};

// ============================================================================
// Environment Topology
// ============================================================================

enum class TopologyType {
    BOUNDED,    // Hard boundaries
    TORUS,      // Wrap-around (periodic)
    INFINITE    // Unbounded space
};

// ============================================================================
// Simulation State
// ============================================================================

enum class SimulationState {
    UNINITIALIZED,
    READY,
    RUNNING,
    PAUSED,
    COMPLETED,
    ERROR
};

} // namespace artemis
