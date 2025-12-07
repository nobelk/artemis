#pragma once

#include "types.hpp"
#include <cmath>
#include <random>

namespace artemis {
namespace math {

// ============================================================================
// Constants
// ============================================================================

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float EPSILON = 1e-6f;

// ============================================================================
// Vector Operations
// ============================================================================

ARTEMIS_HOST_DEVICE inline float dot(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}

ARTEMIS_HOST_DEVICE inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

ARTEMIS_HOST_DEVICE inline float3 cross(const float3& a, const float3& b) {
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

ARTEMIS_HOST_DEVICE inline float distance(const float2& a, const float2& b) {
    return (a - b).length();
}

ARTEMIS_HOST_DEVICE inline float distance_squared(const float2& a, const float2& b) {
    return (a - b).length_squared();
}

ARTEMIS_HOST_DEVICE inline float2 clamp(const float2& v, const float2& min_val, const float2& max_val) {
    return float2(
        fmaxf(min_val.x, fminf(max_val.x, v.x)),
        fmaxf(min_val.y, fminf(max_val.y, v.y))
    );
}

ARTEMIS_HOST_DEVICE inline float clamp(float v, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, v));
}

ARTEMIS_HOST_DEVICE inline float2 limit_magnitude(const float2& v, float max_length) {
    float len = v.length();
    if (len > max_length) {
        return v * (max_length / len);
    }
    return v;
}

// ============================================================================
// Angle Utilities
// ============================================================================

ARTEMIS_HOST_DEVICE inline float angle_between(const float2& a, const float2& b) {
    float dot_product = dot(a.normalized(), b.normalized());
    return acosf(clamp(dot_product, -1.0f, 1.0f));
}

ARTEMIS_HOST_DEVICE inline float2 rotate(const float2& v, float angle) {
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    return float2(
        v.x * cos_a - v.y * sin_a,
        v.x * sin_a + v.y * cos_a
    );
}

ARTEMIS_HOST_DEVICE inline float wrap_angle(float angle) {
    while (angle > PI) angle -= TWO_PI;
    while (angle < -PI) angle += TWO_PI;
    return angle;
}

// ============================================================================
// Morton Code (Z-order curve for spatial indexing)
// ============================================================================

ARTEMIS_HOST_DEVICE inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

ARTEMIS_HOST_DEVICE inline MortonCode compute_morton_code_2d(uint32_t x, uint32_t y) {
    return (static_cast<MortonCode>(expand_bits(y)) << 1) |
           static_cast<MortonCode>(expand_bits(x));
}

ARTEMIS_HOST_DEVICE inline MortonCode position_to_morton(
    const float2& pos,
    const BoundingBox2D& bounds,
    uint32_t grid_resolution) {

    float2 size = bounds.size();
    float2 normalized = float2(
        (pos.x - bounds.min.x) / size.x,
        (pos.y - bounds.min.y) / size.y
    );

    uint32_t x = static_cast<uint32_t>(clamp(normalized.x, 0.0f, 0.9999f) * grid_resolution);
    uint32_t y = static_cast<uint32_t>(clamp(normalized.y, 0.0f, 0.9999f) * grid_resolution);

    return compute_morton_code_2d(x, y);
}

// ============================================================================
// Grid Cell Utilities
// ============================================================================

ARTEMIS_HOST_DEVICE inline CellID position_to_cell(
    const float2& pos,
    const BoundingBox2D& bounds,
    float cell_size) {

    uint32_t x = static_cast<uint32_t>((pos.x - bounds.min.x) / cell_size);
    uint32_t y = static_cast<uint32_t>((pos.y - bounds.min.y) / cell_size);

    uint32_t grid_width = static_cast<uint32_t>((bounds.max.x - bounds.min.x) / cell_size);

    return y * grid_width + x;
}

ARTEMIS_HOST_DEVICE inline float2 wrap_position(
    const float2& pos,
    const BoundingBox2D& bounds) {

    float2 size = bounds.size();
    float2 result = pos;

    while (result.x < bounds.min.x) result.x += size.x;
    while (result.x > bounds.max.x) result.x -= size.x;
    while (result.y < bounds.min.y) result.y += size.y;
    while (result.y > bounds.max.y) result.y -= size.y;

    return result;
}

// ============================================================================
// Random Number Generation (Host-side)
// ============================================================================

class RandomGenerator {
public:
    RandomGenerator(uint64_t seed = 42)
        : generator_(seed), uniform_dist_(0.0f, 1.0f) {}

    void set_seed(uint64_t seed) {
        generator_.seed(seed);
    }

    float uniform() {
        return uniform_dist_(generator_);
    }

    float uniform(float min, float max) {
        return min + (max - min) * uniform();
    }

    float normal(float mean = 0.0f, float stddev = 1.0f) {
        std::normal_distribution<float> dist(mean, stddev);
        return dist(generator_);
    }

    float2 random_point_in_box(const BoundingBox2D& box) {
        return float2(
            uniform(box.min.x, box.max.x),
            uniform(box.min.y, box.max.y)
        );
    }

    float2 random_direction() {
        float angle = uniform(0.0f, TWO_PI);
        return float2(cosf(angle), sinf(angle));
    }

private:
    std::mt19937_64 generator_;
    std::uniform_real_distribution<float> uniform_dist_;
};

} // namespace math
} // namespace artemis
