#include <iostream>
#include <vector>
#include <chrono>
#include "xsor.h"

int main() 
{
	xsor::mat<float> matrix(3, 15);
	xsor::mat_init_random(matrix, -1.5f, 3.f);
	std::cout << matrix << "\n" << xsor::mat_transpose_inp(matrix);
}
