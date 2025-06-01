#pragma once

#include "tensor_mat.h"

class neural_network
{
private:
	std::vector<tensor_mat<float>> m_hidden_w;
	std::vector<tensor_mat<float>> m_hidden_b;
	float lr = 0.000015f;
	float dropout = 0.f;

private: 
	void update_weight(size_t index, tensor_mat<float>& gradient, const tensor_mat<float>& zoutput);
	void update_bias(size_t index, tensor_mat<float>& gradient);
	void forward(const tensor_mat<float>& input, std::vector<tensor_mat<float>>& zoutput, bool isTraining);
	void backward(const tensor_mat<float>& input, const tensor_mat<float>& output, std::vector<tensor_mat<float>>& zoutput);
public:
	void init();
	void train(const tensor_mat<float>& input, const tensor_mat<float>& output);
	void apply_dropout(float rate);
	tensor_mat<float> feed(const tensor_mat<float>& input);
	void print();
};
