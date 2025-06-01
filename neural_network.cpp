#include "neural_network.h"

void neural_network::update_weight(size_t index, tensor_mat<float>& gradient, const tensor_mat<float>& zoutput)
{
    auto a = zoutput * lr;
    auto dw = gradient.transpose() * a;
    m_hidden_w[index] = m_hidden_w[index] - dw;
}

void neural_network::update_bias(size_t index, tensor_mat<float>& gradient)
{
    m_hidden_b[index] = m_hidden_b[index] - (gradient * lr);
}

void neural_network::forward(const tensor_mat<float>& input, std::vector<tensor_mat<float>>& zoutput, bool isTraining)
{
    const size_t size = zoutput.size();

    zoutput[0] = (m_hidden_w[0] * input.transpose()).transpose() + m_hidden_b[0];

    for (size_t i = 1; i < size; i++) {
        zoutput[i] = (m_hidden_w[i] * mat_apply_inp(zoutput[i - 1], relu).transpose()).transpose() + m_hidden_b[i];
    }
}

void neural_network::backward(const tensor_mat<float>& input, const tensor_mat<float>& output, std::vector<tensor_mat<float>>& zoutput)
{
    size_t size = m_hidden_w.size();
    auto gradient = zoutput.back().softmax() - output;
    for (size_t i = size - 1; i > 0; i--) {
        update_weight(i, gradient, zoutput[i - 1]);
        update_bias(i, gradient);
        gradient = mat_elem_mul(gradient * m_hidden_w[i], mat_apply_inp(zoutput[i - 1], relu_derivative));
    }
    update_weight(0, gradient, input);
    update_bias(0, gradient);
}

void neural_network::init()
{
    constexpr size_t size_hidden = 3;

    m_hidden_w.resize(size_hidden);
    m_hidden_b.resize(size_hidden);

    m_hidden_w[0] = tensor_mat<float>(64, 35);
    m_hidden_w[1] = tensor_mat<float>(64, 64);
    m_hidden_w[2] = tensor_mat<float>(10, 64);

    m_hidden_b[0] = tensor_mat<float>(1, 64);
    m_hidden_b[1] = tensor_mat<float>(1, 64);
    m_hidden_b[2] = tensor_mat<float>(1, 10); 

    for (size_t i = 0; i < size_hidden; i++) {
        mat_init_random(m_hidden_w[i], .001f, 1.f);
        mat_init_random(m_hidden_b[i], .001f, 1.f);
    }
}

void neural_network::train(const tensor_mat<float>& input, const tensor_mat<float>& output)
{
    std::vector<tensor_mat<float>> zoutput(m_hidden_w.size());

    forward(input, zoutput, true);

    backward(input, output, zoutput);
}

void neural_network::apply_dropout(float rate)
{
    dropout = rate;
}

tensor_mat<float> neural_network::feed(const tensor_mat<float>& input)
{
    std::vector<tensor_mat<float>> zoutput(m_hidden_w.size());

    forward(input, zoutput, false);

    return zoutput.back();
}

void neural_network::print()
{
    size_t size = m_hidden_w.size();
    for (size_t i = 0; i < size; i++) {
        std::cout << m_hidden_w[i] << "\n" << m_hidden_b[i] << "\n";
    }
}


