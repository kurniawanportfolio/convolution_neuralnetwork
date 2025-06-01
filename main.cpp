#include <iostream>
#include <vector>
#include <chrono>
#include "neural_network.h"
#include "data_training.h"

template <typename T>
void print(const tensor_mat<T>& m) {
	auto size = m.size();
	auto data = m.data();

	for (size_t i = 0; i < size; i++) {
		std::cout << data[i] << " ";
	}

	std::cout << "\n";
}

void run() 
{
	neural_network nn;
	nn.init();
	nn.apply_dropout(0.5);

	size_t epoch = 11500;

	for (size_t i = 0; i < epoch; i++)
	{
		for (size_t j = 0; j < 10; j++) {
			nn.train(inputs[j], targets[j]);
		}
	}

	for (size_t i = 0; i < 10; i++) {
		std::cout << nn.feed(inputs[i]) << "\n";
		std::cout << nn.feed(inputs[i]).softmax() << "\n";
	}
}


int main() {
	//for (size_t i = 0; i < 9; i++) {
	//	std::cout << i / 36.f << std::endl;
	///}
	run();
	//tensor_mat<float> mat(12, 12);
	//mat_init_random(mat, .001f, .29f);
	//std::cout << mat << "\n";
	//std::cout << mat * 5 << "\n";
}