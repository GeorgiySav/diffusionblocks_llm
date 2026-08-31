#pragma once

#include "block.h"

#include <nn/module.h>
#include <nn/ops/ops.h>

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
      ln_f_(config.n_embed),
      head_(token_embedding_.weight()),
      config_(config) {
    for (int i = 0; i < config.n_layers; ++i) {
      blocks_.emplace_back(config.n_heads, config.n_embed, config.max_seq_len, config.mlp_hidden_dim, config.dropout, rng);
    }
    head_.set_requires_grad(true);
  }

  nn::Tensor forward(const nn::Tensor& idx) {
    int T = idx.extent(1);
    if (T > config_.max_seq_len) {
      throw std::invalid_argument("Input sequence length exceeds maximum sequence length");
    }

    // Matches DiffusionRecurrentLLM/DiffusionBlockedGPT's embedding handling
    // (prevents embedding collapse) -- everything else in a Block is already
    // the same building blocks (RMSNorm, CausalAttention, SwiGLU MLP,
    // tied+untied-at-init head) those models use, so this is the one real
    // architectural gap between "plain GPT" and the diffusion models' shared
    // scaffolding. Kept even though GPT has no noise to be robust to,
    // because the point of these comparisons is to isolate the training
    // *scheme* as the difference, not incidental embedding-scale effects.
    nn::Tensor x = l2_normalize(token_embedding_.forward(idx)); // [B, T, n_embed]

    for (Block& block : blocks_) {
      x = block.forward(x); // [B, T, n_embed]
    }

    x = ln_f_.forward(x); // [B, T, n_embed]
    nn::Tensor logits = x.mm(head_, /*transB=*/true); // [B, T, vocab_size]
    return logits;
  }

  nn::Tensor generate(const nn::Tensor& idx, int max_new_tokens, int topk, nn::Pcg32& rng) {
    nn::Tensor x = idx; // [B, T]
    for (int i = 0; i < max_new_tokens; ++i) {
      nn::Tensor logits = forward(x); // [B, T, V]
      nn::Tensor last = logits.slice_view(1, logits.extent(1) - 1, 1); // [B, 1, V]
      last = last.reshape_view({last.extent(0), last.extent(2)}); // [B, V]

      nn::Tensor probs = last.softmax(); // [B, V]
      nn::Tensor values, indices;
      nn::ops::topk_rows(probs, topk, values, indices);

      // Sample from the top-k probabilities
      const nn::Tensor local = nn::ops::multinomial(values);   // [B], index into values' columns
      const nn::Tensor picked = nn::ops::gather_rows(indices, local); // [B]

      x = nn::cat({x, picked.reshape_view({picked.extent(0), 1})}, 1); // [B, T+1]
    }
    return x;
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    token_embedding_.collect_named(prefix + "token_embedding.", out);
    for (size_t i = 0; i < blocks_.size(); ++i) {
      blocks_[i].collect_named(prefix + "blocks." + std::to_string(i) + ".", out);
    }
    ln_f_.collect_named(prefix + "ln_f.", out);
    out.push_back({prefix + "head", &head_});
  }

  void set_training(bool on) override {
    training_ = on;
    for (Block& block : blocks_) block.set_training(on);
  }

private:
  static nn::Tensor l2_normalize(const nn::Tensor& x, float eps = 1e-6f) {
    const int last = x.rank() - 1;
    nn::Tensor norm = x.pow(2.0f).sum(last, /*keepdim=*/true).sqrt();
    return x / (norm + eps);
  }

  nn::Embedding token_embedding_;
  std::vector<Block> blocks_;
  nn::RMSNorm ln_f_;
  nn::Tensor head_;

  Config config_;
};