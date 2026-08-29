#pragma once

#include "block.h"
#include "RoPE.h"
#include <nn/module.h>

class GPT : public nn::Module {
public:
  struct Config {
    int vocab_size;
    int n_embed;
    int n_heads;
    int n_layers;
    int mlp_hidden_dim;
    float dropout;
    int max_seq_len;
  };
  GPT(Config config, nn::Pcg32& rng)
    : token_embedding_(config.vocab_size, config.n_embed, rng),
      position_embedding_(config.n_embed, config.max_seq_len, rng),
      ln_f_(config.n_embed),
      head_(nn::Tensor::randn({config.n_embed, config.vocab_size}, rng, 0.02f)),
      config_(config) {
    for (int i = 0; i < config.n_layers; ++i) {
      blocks_.emplace_back(config.n_heads, config.n_embed, config.mlp_hidden_dim, config.dropout, rng);
    }
    head_.set_requires_grad(true);

    // TODO: weight sharing
  }

  nn::Tensor forward(const nn::Tensor& idx) {
    int B = idx.extent(0);
    int T = idx.extent(1);
    if (T > config_.max_seq_len) {
      throw std::invalid_argument("Input sequence length exceeds maximum sequence length");
    }

    nn::Tensor pos = nn::Tensor::arange(T, idx.device(), nn::DType::I32); // [T]
    nn::Tensor tok_emb = token_embedding_.forward(idx); // [B, T, n_embed]
    nn::Tensor pos_emb = position_embedding_.forward(pos); // [T, n_embed]
    nn::Tensor x = tok_emb + pos_emb; // [B, T, n_embed]

    for (Block& block : blocks_) {
      x = block.forward(x); // [B, T, n_embed]
    }

    x = ln_f_.forward(x); // [B, T, n_embed]
    nn::Tensor logits = x.mm(head_); // [B, T, vocab_size]
    return logits;
  }

private:
  nn::Embedding token_embedding_;
  RoPE position_embedding_;
  std::vector<Block> blocks_;
  nn::RMSNorm ln_f_;
  nn::Tensor head_;

  Config config_;
};