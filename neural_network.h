#pragma once

#include "xsor.h"

class neural_network
{
private:
	std::vector<xsor::mat<float>> m_hw;
	std::vector<xsor::mat<float>> m_hb;
	float lr = 0.00001;

private: 
	std::vector<xsor::mat<float>> forward(const xsor::mat<float>& input);
	void backward(const std::vector<xsor::mat<float>>& gradients);
public:
	void init();
	void train(xsor::mat<float>& input, xsor::mat<float>& output);
};

