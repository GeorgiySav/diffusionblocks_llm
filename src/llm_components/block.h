#pragma once

#include <nn/module.h>

#include "attention.h"
#include "RoPE.h"
#include "mlp.h"

class Block : public nn::Module {
public:
  Block(int n_heads, int n_embed, int mlp_hidden_dim, float dropout, nn::Pcg32& rng)
    : attention_(n_heads, n_embed, dropout, rng),
      mlp_(n_embed, mlp_hidden_dim, n_embed, dropout, rng),
      norm1_(n_embed),
      norm2_(n_embed) {}

  nn::Tensor forward(const nn::Tensor& x) override {
    nn::Tensor attn_out = attention_.forward(norm1_.forward(x));
    nn::Tensor x1 = x + attn_out; // Residual connection
    nn::Tensor mlp_out = mlp_.forward(norm2_.forward(x1));
    return x1 + mlp_out; // Residual connection
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    attention_.collect_named(prefix + "attn", out);
    mlp_.collect_named(prefix + "mlp", out);
    norm1_.collect_named(prefix + "norm1", out);
    norm2_.collect_named(prefix + "norm2", out);
  }

private:
  CausalAttention attention_;
  MLP mlp_;
  nn::RMSNorm norm1_;
  nn::RMSNorm norm2_;
};