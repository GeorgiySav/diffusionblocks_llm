#pragma once

#include <nn/module.h>

class MLP : public nn::Module {
public:
  MLP(int input_dim, int hidden_dim, int output_dim, float dropout, nn::Pcg32& rng)
    : layers_(nn::Sequential(
        nn::Linear(input_dim, hidden_dim, rng),
        nn::ReLu(),
        nn::Linear(hidden_dim, output_dim, rng),
        nn::Dropout(dropout)
      )) {}

  nn::Tensor forward(const nn::Tensor& x) override {
    return layers_.forward(x);
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    layers_.collect_named(prefix + "layers", out);
  }
private:
  nn::Sequential layers_;
};