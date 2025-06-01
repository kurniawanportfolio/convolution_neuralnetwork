#pragma once


#ifndef HELPER_H
#define HELPER_H

#include <cassert>
#include <cmath>
#include <random>
#include <type_traits>

template<typename T>
inline constexpr T relu(T n)
{
	static_assert(std::is_floating_point<T>::value, "Decimal type required.");
	return std::max(static_cast<T>(0), n);
}

template<typename T>
inline constexpr T relu_derivative(T n)
{
	static_assert(std::is_floating_point<T>::value, "Decimal type required.");
	return n > 0? T(1) : T(0);
}

template<typename T>
inline T gelu(T x) {
    const T coeff = 1.702f;
    return 0.5f * x * (1.0f + tanhf(coeff * x));
}

template<typename T>
inline T gelu_derivative(T x) {
    const T coeff = 1.702f;
    T tanh_val = tanhf(coeff * x);
    return 0.5f * (1.f + tanh_val) + 0.5f * x * (1.f - tanh_val * tanh_val) * coeff;
}


// Computes softmax in a numerically stable way (avoids overflow)
template <typename T>
inline std::vector<T> get_softmax(const std::vector<T>& logits) {
    static_assert(std::is_floating_point<T>::value, "Decimal type required.");

    std::vector<T> probabilities(logits.size());

    // Find the maximum logit to avoid numerical instability
    T max_logit = *std::max_element(logits.begin(), logits.end());

    // Compute exponentials (shifted by max_logit for stability)
    T sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = exp(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }

    // Normalize to get probabilities
    for (T& prob : probabilities) {
        prob /= sum_exp;
    }
    return probabilities;
}

// Computes softmax in a numerically stable way (avoids overflow)
template <typename T>
inline void get_softmax_inp(std::vector<T>& logits) {
    static_assert(std::is_floating_point<T>::value, "Decimal type required.");

    if (logits.empty()) return;  // Safe guard

    const size_t size = logits.size();

    T* rawprob = logits.data();

    // Find the maximum logit to avoid numerical instability
    T max_logit = *std::max_element(logits.begin(), logits.end());

    // Compute exponentials (shifted by max_logit for stability)
    T sum_exp = static_cast<T>(0);
    for (size_t i = 0; i < size; ++i) {
        rawprob[i] = std::exp(rawprob[i] - max_logit);
        sum_exp += rawprob[i];
    }

    // Normalize to get probabilities
    const T inv_sum = static_cast<T>(1) / sum_exp;
    for (size_t i = 0; i < size; ++i) {
        rawprob[i] *= inv_sum;
    }
}

inline std::mt19937& global_gen() {
	static std::mt19937 gen(std::random_device{}());
	return gen;
}

inline int random_number(int n) {
	std::uniform_int_distribution<> distrib(0, n);
	return distrib(global_gen());
}

// Random number between a and b (inclusive)
inline int random_number(int a, int b) {
    std::uniform_int_distribution<> distrib(a, b);
    return distrib(global_gen());
}

template <typename T>
inline T random_number(T a, T b)
{
    std::uniform_real_distribution<T> distrib(a, b);
    return distrib(global_gen());
}

#endif