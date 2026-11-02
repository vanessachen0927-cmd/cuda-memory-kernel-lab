#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

inline void vector_add_cpu(const float* a, const float* b, float* out, std::size_t n) {
    if (!a || !b || !out) throw std::invalid_argument("null vector pointer");
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
inline bool close_enough(float got, float expected, float tol = 1e-5f) {
    float diff = got - expected;
    return (diff < 0 ? -diff : diff) <= tol;
}
