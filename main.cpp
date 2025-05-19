#include <iostream>
#include <vector>
#include <chrono>
#include "xsor.h"


int main() 
{
	xsor::mat<double> m1 = { 1,2,3,4 };
	xsor::mat<double> m2(10, 10);

	xsor::mat_init_random(m2, -1.0, 1.0);

	std::cout << m1.softmax() << "\n";
	std::cout << m2 << "\n";
	std::cout << m2.softmax() << "\n";
}
