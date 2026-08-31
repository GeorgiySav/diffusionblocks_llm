#pragma once

#include <nn/module.h>

#include "SwiGLU.h"

class TimestepEmbedder : public nn::Module {
public:
  TimestepEmbedder(int n_embed, nn::Pcg32& rng)
    : n_embed_(n_embed),
      mlp_(nn::Sequential(
          nn::Linear(n_embed, 4 * n_embed, rng),
          SwiGLU(4 * n_embed, rng),
          nn::Linear(4 * n_embed, n_embed, rng))) {}

  nn::Tensor forward(const nn::Tensor& sigma) {
    const int half_dim = n_embed_ / 2;
    float embed_scale = std::log(10000.0f) / (half_dim - 1);

    const nn::Tensor c_noise = sigma.log() * 0.25f;

    nn::Tensor emb = (nn::Tensor::arange(half_dim, sigma.device(), nn::DType::F32) * -embed_scale)
                      .exp();                                                                           
                                   
    emb = c_noise.reshape_view({c_noise.extent(0), 1}) * emb.reshape_view({1, half_dim});
    emb = nn::cat({emb.sin(), emb.cos()}, 1);

    return mlp_.forward(emb);
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    mlp_.collect_named(prefix + "mlp.", out);
  }

private:
  const int n_embed_;
  nn::Sequential mlp_;
};