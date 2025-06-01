#pragma once

#ifndef TENSOR_MAT_H
#define TENSOR_MAT_H

#include <iostream>
#include <vector>
#include <initializer_list>
#include <type_traits>
#include <cassert>
#include <string>
#include <charconv>
#include <cstdio>
#include "helper.h"

template <typename T>
class tensor_mat;

template <typename T>
inline tensor_mat<T>& mat_init_random(tensor_mat<T>& m, T low, T high);
template<typename T>
inline tensor_mat<T> mat_mul(const tensor_mat<T>& a, const tensor_mat<T>& b);
template<typename T>
inline tensor_mat<T> mat_elem_mul(const tensor_mat<T>& a, const tensor_mat<T>& b);
template<typename T>
inline tensor_mat<T> mat_elem_add(const tensor_mat<T>& a, const tensor_mat<T>& b);
template<typename T>
inline tensor_mat<T> mat_elem_sub(const tensor_mat<T>& a, const tensor_mat<T>& b);
template<typename T>
inline tensor_mat<T> mat_transpose(const tensor_mat<T>& m);
template<typename T>
inline tensor_mat<T> mat_scale(const tensor_mat<T>& m, T n);
template<typename T>
inline tensor_mat<T>& mat_scale_inp(tensor_mat<T>& m, T n);
template<typename T>
inline tensor_mat<T> mat_apply(const tensor_mat<T>& m, T(*f)(T));
template<typename T>
inline tensor_mat<T>& mat_apply_inp(tensor_mat<T>& m, T(*f)(T));


template <typename T>
class tensor_mat {
	static_assert(std::is_floating_point<T>::value, "Decimal type required.");
private:
	std::vector<T> m_data;
	size_t m_rows = 0, m_cols = 0;
public:
	tensor_mat() = default;
	tensor_mat(const tensor_mat& other) = default;
	tensor_mat(tensor_mat&& other) noexcept = default;
	tensor_mat& operator = (const tensor_mat& other) = default;
	tensor_mat& operator = (tensor_mat&& other) noexcept = default;
	tensor_mat(size_t rows, size_t cols) : m_rows(rows), m_cols(cols)
	{
		m_data.resize(m_rows * m_cols);
	}
	tensor_mat(size_t rows, size_t cols, T value) : m_rows(rows), m_cols(cols)
	{
		m_data.resize(m_rows * m_cols, value);
	}
	tensor_mat(const std::vector<T>& data)
		: m_rows(data.empty() ? 0 : 1), m_cols(data.size()), m_data(data)
	{}
	tensor_mat(std::vector<T>&& data) noexcept
		: m_rows(m_data.empty() ? 0 : 1), m_cols(m_data.size()), m_data(std::move(data))
	{}
	tensor_mat(size_t rows, size_t cols, const std::vector<T>& data)
		: m_rows(rows), m_cols(cols), m_data(data)
	{
		assert(rows * cols == data.size() && "Matrix dimensions invalid");
	}
	tensor_mat(size_t rows, size_t cols, std::vector<T>&& data) noexcept
		: m_rows(rows), m_cols(cols), m_data(std::move(data))
	{
		assert(m_rows * m_cols == m_data.size() && "Matrix dimensions invalid");
	}
	tensor_mat(std::initializer_list<T> data)
		: m_rows(data.size() ? 1 : 0), m_cols(data.size()), m_data(data)
	{}
	tensor_mat& operator = (const std::vector<T>& data)
	{
		m_data = data;
		m_rows = m_data.size() > 0 ? 1 : 0;
		m_cols = m_data.size();
		return *this;
	}
	tensor_mat& operator = (std::vector<T>&& data) noexcept
	{
		m_data = std::move(data);
		m_rows = m_data.size() > 0 ? 1 : 0;
		m_cols = m_data.size();
		return *this;
	}
	tensor_mat& operator = (std::initializer_list<T> data)
	{
		m_data = data;
		m_rows = m_data.size() > 0 ? 1 : 0;
		m_cols = m_data.size();
		return *this;
	}
	inline tensor_mat softmax()
	{
		auto data = get_softmax(m_data);

		return tensor_mat(m_rows, m_cols, std::move(data));
	}
	inline const tensor_mat softmax() const
	{
		auto data = get_softmax(m_data);

		return tensor_mat(m_rows, m_cols, std::move(data));
	}
	inline tensor_mat operator+ (const tensor_mat& other)
	{
		return mat_elem_add(*this, other);
	}	
	inline const tensor_mat operator+ (const tensor_mat& other) const
	{
		return mat_elem_add(*this, other);
	}
	inline tensor_mat operator- (const tensor_mat& other)
	{
		return mat_elem_sub(*this, other);
	}	
	inline const tensor_mat operator- (const tensor_mat& other) const
	{
		return mat_elem_sub(*this, other);
	}
	inline tensor_mat operator* (const tensor_mat& other)
	{
		return mat_mul(*this, other);
	}	
	inline const tensor_mat operator* (const tensor_mat& other) const
	{
		return mat_mul(*this, other);
	}
	inline tensor_mat operator* (T n)
	{
		return mat_scale(*this, n);
	}	
	inline const tensor_mat operator* (T n) const
	{
		return mat_scale(*this, n);
	}
	inline tensor_mat transpose()
	{
		return mat_transpose(*this);
	}	
	inline const tensor_mat transpose() const
	{
		return mat_transpose(*this);
	}
	inline void reshape(size_t rows, size_t cols)
	{
		size_t size = rows * cols;
		if (m_data.size() != size) return;
		m_rows = rows;
		m_cols = cols;
	}
	inline void resize(size_t rows, size_t cols)
	{
		size_t size = rows * cols;
		m_rows = rows;
		m_cols = cols;
		if (m_data.size() != size) m_data.resize(size);
	}
	inline T* data() noexcept
	{
		return m_data.data();
	}
	inline const T* data() const noexcept
	{
		return m_data.data();
	}
	inline T& at(size_t row, size_t col) noexcept
	{
		return m_data[row * m_cols + col];
	}
	inline const T& at(size_t row, size_t col) const noexcept
	{
		return m_data[row * m_cols + col];
	}
	inline T& at_index(size_t index) noexcept
	{
		return m_data[index];
	}
	inline const T& at_index(size_t index) const noexcept
	{
		return m_data[index];
	}
	constexpr size_t rows() const noexcept
	{
		return m_rows;
	}
	constexpr size_t cols() const noexcept
	{
		return m_cols;
	}
	constexpr size_t size() const noexcept
	{
		return m_data.size();
	}
	friend std::ostream& operator<< (std::ostream& os, const tensor_mat<T>& m)
	{
		size_t rows = m.m_rows, cols = m.m_cols, size = m.size();
		const T* data = m.data();

		std::string out;
		out.reserve(rows * cols * 12);
		out.append("[");

		const int precision = 2;
		const int width = precision + 3; // 1 for sign, 1 for dot, 3 for digits (tweak as needed)
		char numbuf[32];

		for (int i = 0; i < rows; ++i) {
			if (i > 0) out += " ";
			out.append("[");
			for (int j = 0; j < cols; ++j) {
				auto [ptr, ec] = std::to_chars(numbuf, numbuf + sizeof(numbuf), data[i * cols + j], std::chars_format::fixed, precision);
				if (ec == std::errc()) {
					std::string s(numbuf, ptr);
					if (s.size() < static_cast<size_t>(width))
						out.append(width - s.size(), ' '); // pad left with spaces
					out.append(s);
				}
				else out.append("NaN");
				if (j < cols - 1) out.append(", ");
			}
			out.append("]");
			if (i < rows - 1) out += "\n";
		}

		out.append("]");
		os << out;

		return os;
	}
};

template <typename T>
inline tensor_mat<T>& mat_init_random(tensor_mat<T>& m, T low, T high)
{
	size_t size = m.size();

	T* rawdata = m.data();

	constexpr size_t BLOCK_SIZE = 32 * 32;

	for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
		size_t i_max = std::min(ii + BLOCK_SIZE, size);

		for (size_t i = ii; i < i_max; i++) {
#if 1
			rawdata[i] = random_number(low, high);
#else
			float rd = random_number(8) / 36.f;
			rawdata[i] = rd > 0.f ? rd : 0.1f;
#endif
		}
	}

	return m;
}
template<typename T>
inline tensor_mat<T> mat_mul(const tensor_mat<T>& a, const tensor_mat<T>& b)
{
	if (a.cols() != b.rows()) throw std::invalid_argument("Invalid argument.");

	size_t a_rows = a.rows();
	size_t a_cols = a.cols();
	size_t b_cols = b.cols();

	std::vector<T> data(a_rows * b_cols);

	const T* a_rawdata = a.data();
	const T* b_rawdata = b.data();
	T* rawdata = data.data();

	constexpr size_t BLOCK_SIZE = 32;

	for (size_t ii = 0; ii < a_rows; ii += BLOCK_SIZE) {
		for (size_t jj = 0; jj < b_cols; jj += BLOCK_SIZE) {
			for (size_t kk = 0; kk < a_cols; kk += BLOCK_SIZE) {

				size_t i_max = std::min(ii + BLOCK_SIZE, a_rows);
				size_t j_max = std::min(jj + BLOCK_SIZE, b_cols);
				size_t k_max = std::min(kk + BLOCK_SIZE, a_cols);

				for (size_t i = ii; i < i_max; i++) {
					for (size_t j = jj; j < j_max; j++) {
						T sum = rawdata[i * b_cols + j];
						for (size_t k = kk; k < k_max; k++) {
							sum += a_rawdata[i * a_cols + k] * b_rawdata[k * b_cols + j];
						}
						rawdata[i * b_cols + j] = sum;
					}
				}
			}
		}
	}

	return tensor_mat<T>(a_rows, b_cols, std::move(data));
}
template<typename T>
inline tensor_mat<T> mat_elem_mul(const tensor_mat<T>& a, const tensor_mat<T>& b)
{
	if (a.rows() != b.rows() || a.cols() != b.cols())
		throw std::invalid_argument("Invalid argument.");

	size_t size = a.size();

	std::vector<T> data(size);

	const T* a_rawdata = a.data();
	const T* b_rawdata = b.data();
	T* rawdata = data.data();

	constexpr size_t BLOCK_SIZE = 32 * 32;

	for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
		size_t i_max = std::min(ii + BLOCK_SIZE, size);

		for (size_t i = ii; i < i_max; i++) rawdata[i] = a_rawdata[i] * b_rawdata[i];
	}

	return tensor_mat<T>(a.rows(), a.cols(), std::move(data));
}
template<typename T>
inline tensor_mat<T> mat_elem_add(const tensor_mat<T>& a, const tensor_mat<T>& b)
{
	if (a.rows() != b.rows() || a.cols() != b.cols())
		throw std::invalid_argument("Invalid argument.");

	size_t size = a.size();

	std::vector<T> data(size);

	const T* a_rawdata = a.data();
	const T* b_rawdata = b.data();
	T* rawdata = data.data();

	constexpr size_t BLOCK_SIZE = 32 * 32;

	for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
		size_t i_max = std::min(ii + BLOCK_SIZE, size);

		for (size_t i = ii; i < i_max; i++) rawdata[i] = a_rawdata[i] + b_rawdata[i];
	}

	return tensor_mat<T>(a.rows(), a.cols(), std::move(data));
}
template<typename T>
inline tensor_mat<T> mat_elem_sub(const tensor_mat<T>& a, const tensor_mat<T>& b)
{
	if (a.rows() != b.rows() || a.cols() != b.cols())
		throw std::invalid_argument("Invalid argument.");

	size_t size = a.size();

	std::vector<T> data(size);

	const T* a_rawdata = a.data();
	const T* b_rawdata = b.data();
	T* rawdata = data.data();

	constexpr size_t BLOCK_SIZE = 32 * 32;

	for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
		size_t i_max = std::min(ii + BLOCK_SIZE, size);

		for (size_t i = ii; i < i_max; i++) rawdata[i] = a_rawdata[i] - b_rawdata[i];
	}

	return tensor_mat<T>(a.rows(), a.cols(), std::move(data));
}
template<typename T>
inline tensor_mat<T> mat_transpose(const tensor_mat<T>& m)
{
	size_t new_rows = m.cols(), new_cols = m.rows();

	std::vector<T> data(new_rows * new_cols);

	const T* m_rawdata = m.data();
	T* rawdata = data.data();

	constexpr size_t BLOCK_SIZE = 32;

	for (size_t ii = 0; ii < new_rows; ii += BLOCK_SIZE) {
		for (size_t jj = 0; jj < new_cols; jj += BLOCK_SIZE) {
			size_t i_max = std::min(ii + BLOCK_SIZE, new_rows);
			size_t j_max = std::min(jj + BLOCK_SIZE, new_cols);

			for (size_t i = ii; i < i_max; i++) {
				for (size_t j = jj; j < j_max; j++) {
					rawdata[i * new_cols + j] = m_rawdata[j * new_rows + i];
				}
			}
		}
	}

	return tensor_mat<T>(new_rows, new_cols, std::move(data));
}
template<typename T>
inline tensor_mat<T> mat_scale(const tensor_mat<T>& m, T n)
{
	size_t size = m.size();

	std::vector<T> data(size);

	const T* m_rawdata = m.data();

	T* rawdata = data.data();

	constexpr size_t BLOCK_SIZE = 32 * 32;

	for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
		size_t i_max = std::min(ii + BLOCK_SIZE, size);

		for (size_t i = ii; i < i_max; i++) rawdata[i] = m_rawdata[i] * n;
	}

	return tensor_mat<T>(m.rows(), m.cols(), std::move(data));
}
template<typename T>
inline tensor_mat<T>& mat_scale_inp(tensor_mat<T>& m, T n)
{
	size_t size = m.size();

	T* m_rawdata = m.data();

	constexpr size_t BLOCK_SIZE = 32 * 32;

	for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
		size_t i_max = std::min(ii + BLOCK_SIZE, size);

		for (size_t i = ii; i < i_max; i++) m_rawdata[i] *= n;
	}

	return m;
}
template<typename T>
inline tensor_mat<T> mat_apply(const tensor_mat<T>& m, T(*f)(T))
{
	size_t size = m.size();

	std::vector<T> data(size);

	const T* m_rawdata = m.data();

	T* rawdata = data.data();

	constexpr size_t BLOCK_SIZE = 32 * 32;

	for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
		size_t i_max = std::min(ii + BLOCK_SIZE, size);

		for (size_t i = ii; i < i_max; i++) rawdata[i] = f(m_rawdata[i]);
	}

	return tensor_mat<T>(m.rows(), m.cols(), std::move(data));
}
template<typename T>
inline tensor_mat<T>& mat_apply_inp(tensor_mat<T>& m, T(*f)(T))
{
	size_t size = m.size();

	T* m_rawdata = m.data();

	constexpr size_t BLOCK_SIZE = 32 * 32;

	for (size_t ii = 0; ii < size; ii += BLOCK_SIZE) {
		size_t i_max = std::min(ii + BLOCK_SIZE, size);

		for (size_t i = ii; i < i_max; i++) m_rawdata[i] = f(m_rawdata[i]);
	}

	return m;
}

#endif