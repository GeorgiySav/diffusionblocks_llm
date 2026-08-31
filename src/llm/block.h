#pragma once

#include <nn/module.h>

#include "attention.h"
#include "mlp.h"

class Block : public nn::Module {
public:
  Block(int n_heads, int n_embed, int max_seq_len, int mlp_hidden_dim, float dropout, nn::Pcg32& rng)
    : attention_(n_heads, n_embed, max_seq_len, dropout, rng),
      mlp_(n_embed, mlp_hidden_dim, n_embed, dropout, rng),
      norm1_(n_embed),
      norm2_(n_embed) {}

  nn::Tensor forward(const nn::Tensor& x) override {
    return forward(x, nn::tril_mask(x.extent(1), x.device()), /*rope_period=*/0);
  }

  // See CausalAttention::forward for keep_mask / rope_period. Lets prelude
  // and coda layers run over the 2T-concatenated sequence with the same
  // attention pattern as the recurrent core.
  nn::Tensor forward(const nn::Tensor& x, const nn::Tensor& keep_mask, int rope_period = 0) {
    nn::Tensor attn_out = attention_.forward(norm1_.forward(x), keep_mask, rope_period);
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

  void set_training(bool on) override {
    training_ = on;
    attention_.set_training(on);
    mlp_.set_training(on);
  }

private:
  CausalAttention attention_;
  MLP mlp_;
  nn::RMSNorm norm1_;
  nn::RMSNorm norm2_;
};