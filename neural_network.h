#pragma once

#include "tensor_mat.h"

class neural_network
{
private:
	std::vector<tensor_mat<float>> m_hidden_w;
	std::vector<tensor_mat<float>> m_hidden_b;
	float lr = 0.00001f;

private: 
	void update_weight(size_t index, tensor_mat<float>& gradient, tensor_mat<float>& zoutput);
	void update_bias(size_t index, tensor_mat<float>& gradient);
	void forward(tensor_mat<float>& input, std::vector<tensor_mat<float>>& zoutput);
	void backward(tensor_mat<float>& input, tensor_mat<float>& output, std::vector<tensor_mat<float>>& zoutput);
public:
	void init();
	void train(tensor_mat<float>& input, tensor_mat<float>& output);
};
