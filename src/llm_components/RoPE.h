#pragma once

#include <cmath>

#include <nn/module.h>

class RoPE : public nn::Module {
public:
  RoPE(int d_model, int max_seq_len, nn::Pcg32& rng)
    : rotation_matrix_(nn::Tensor::zeros({d_model, d_model}, nn::Device::CPU, nn::DType::F32)),
      positional_embedding_(nn::Tensor::zeros({max_seq_len, d_model}, nn::Device::CPU, nn::DType::F32)) {
    // rotation matrix
    for (int i = 0; i < d_model; ++i) {
      for (int j = 0; j < d_model; ++j) {
        rotation_matrix_.host_data()[i * d_model + j] = static_cast<float>(std::cos(i * j * 0.01));
      }
    }

    // positional embedding
    for (int pos = 0; pos < max_seq_len; ++pos) {
      for (int i = 0; i < d_model; ++i) {
        positional_embedding_.host_data()[pos * d_model + i] = static_cast<float>(std::sin(pos * i * 0.01));
      }
    }
  }

  nn::Tensor forward(const nn::Tensor& x) override {
    nn::Tensor p = x + positional_embedding_;
    return p.mm(rotation_matrix_);
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    out.push_back({prefix + "rotation_matrix", &rotation_matrix_});
    out.push_back({prefix + "positional_embedding", &positional_embedding_});
  }

private:
  nn::Tensor rotation_matrix_;
  nn::Tensor positional_embedding_;
};