#include "neural_network.h"

void neural_network::update_weight(size_t index, tensor_mat<float>& gradient, tensor_mat<float>& zoutput)
{
#if 0
    m_hidden_w[index] = mat_elem_sub(
        m_hidden_w[index],
        mat_scale(
            mat_mul(mat_transpose(zoutput), gradient),
            lr
        )
    );
#else
    const size_t rows = m_hidden_w[index].rows();
    const size_t cols = m_hidden_w[index].cols();
    float* weights = m_hidden_w[index].data();
    float* grad = gradient.data();
    float* zout = zoutput.data();
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            weights[i * cols + j] -= lr * grad[i] * relu(zout[j]);
        }
    }
#endif
}

void neural_network::update_bias(size_t index, tensor_mat<float>& gradient)
{
#if 0
    m_hidden_b[index] = mat_elem_sub(
        m_hidden_b[index],
        mat_scale(gradient, lr)
    );
#else
    const size_t size = m_hidden_b[index].size();
    float* biases = m_hidden_b[index].data();
    float* grad = gradient.data();
    for (size_t i = 0; i < size; i++) {
        biases[i] -= grad[i] * lr;
    }
#endif
}

void neural_network::forward(tensor_mat<float>& input, std::vector<tensor_mat<float>>& zoutput)
{
    const size_t size = zoutput.size();

    zoutput[0] = mat_elem_add(
        mat_mul(input, m_hidden_w[0]),
        m_hidden_b[0]
    );

    for (size_t i = 1; i < size; i++) {
        zoutput[i] = mat_elem_add(
            mat_mul(
                mat_apply(zoutput[i - 1], relu),
                m_hidden_w[i]
            ),
            m_hidden_b[i]
        );
    }
}

void neural_network::backward(tensor_mat<float>& input, tensor_mat<float>& output, std::vector<tensor_mat<float>>& zoutput)
{
    tensor_mat<float> gradient = mat_elem_sub(
        zoutput.back().softmax_inp(),
        output
    );

    for (size_t i = m_hidden_w.size() - 1; i > 0; i--) {
        update_weight(i, gradient, zoutput[i - 1]);
        update_bias(i, gradient);

        gradient = mat_transpose(
            mat_elem_mul(
                mat_mul(m_hidden_w[i], mat_transpose(gradient)),
                mat_apply_inp(zoutput[i - 1], relu_derivative)
            )
        );
    }

    update_weight(0, gradient, input);
    update_bias(0, gradient);
}

void neural_network::init()
{
    constexpr size_t size_hidden = 3;

    m_hidden_w.resize(size_hidden);
    m_hidden_b.resize(size_hidden);

    m_hidden_w[0] = tensor_mat<float>(35, 64);
    m_hidden_w[1] = tensor_mat<float>(64, 64);
    m_hidden_w[2] = tensor_mat<float>(64, 10);

    m_hidden_b[0] = tensor_mat<float>(1, 64);
    m_hidden_b[1] = tensor_mat<float>(1, 64);
    m_hidden_b[2] = tensor_mat<float>(1, 10);

    for (size_t i = 0; i < size_hidden; i++) {
        mat_init_random(m_hidden_w[i], 0.1f, 0.5f);
        mat_init_random(m_hidden_b[i], 0.1f, 0.5f);
    }
}

void neural_network::train(tensor_mat<float>& input, tensor_mat<float>& output)
{
    std::vector<tensor_mat<float>> zoutput(m_hidden_w.size());

    forward(input, zoutput);

    backward(input, output, zoutput);
}


