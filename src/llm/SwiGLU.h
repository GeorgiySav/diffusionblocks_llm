#pragma once

#include <nn/module.h>

class SwiGLU : public nn::Module {
public:
  SwiGLU(int dim, nn::Pcg32& rng)
    : linear1_(dim, dim, rng),
      linear2_(dim, dim, rng) {}

  nn::Tensor forward(const nn::Tensor& x) {
    nn::Tensor x1 = linear1_.forward(x);
    nn::Tensor swish = x1 * x1.sigmoid();
    nn::Tensor swiglu = swish * linear2_.forward(x);

    return swiglu;
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    linear1_.collect_named(prefix + "linear1.", out);
    linear2_.collect_named(prefix + "linear2.", out);
  }

private:
  nn::Linear linear1_;
  nn::Linear linear2_;
};