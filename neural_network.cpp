#include "neural_network.h"

std::vector<xsor::mat<float>> neural_network::forward(const xsor::mat<float>& input)
{
    const size_t size = m_hw.size();

    std::vector<xsor::mat<float>> zoutput(size);

    auto raw_data = zoutput.data();

    for (size_t i = 0; i < size; i++) {
        switch (i) {
        case 0:
            raw_data[i] = xsor::mat_elem_add(
                xsor::mat_mul_optimized(input, m_hw[0]),
                m_hb[0]
            );

            break;
        default:
            raw_data[i] = xsor::mat_elem_add(
                xsor::mat_mul_optimized(
                    xsor::mat_apply_activation(zoutput[i - 1], relu), 
                    m_hw[0]
                ),
                m_hb[0]
            );

            break;
        }
    }

    return zoutput;
}

void neural_network::backward(const std::vector<xsor::mat<float>>& gradients)
{

}

void neural_network::init()
{

}

void neural_network::train(xsor::mat<float>& input, xsor::mat<float>& output)
{

}
