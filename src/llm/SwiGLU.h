#pragma once

#include <nn/autograd/functions.h>
#include <nn/module.h>

class SwiGLU : public nn::Module {
public:
  SwiGLU(int in_features, int out_features, nn::Pcg32& rng, bool bias = true)
    : gate_(in_features, out_features, rng, bias),
      up_(in_features, out_features, rng, bias) {}

  nn::Tensor forward(const nn::Tensor& x) override {
    return nn::autograd::silu(gate_.forward(x)) * up_.forward(x);
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    gate_.collect_named(prefix + "gate.", out);
    up_.collect_named(prefix + "up.", out);
  }

private:
  nn::Linear gate_;
  nn::Linear up_;
};
