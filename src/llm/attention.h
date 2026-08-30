#pragma once

#include <nn/autograd/functions.h>
#include <nn/module.h>

#include "RoPE.h"

class CausalAttention : public nn::Module {

public:
  CausalAttention(int n_heads, int n_embed, int max_seq_len, float dropout, nn::Pcg32& rng)
    : n_heads_(n_heads),
      n_embed_(n_embed),
      dropout_(dropout),
      c_attn_(nn::Linear(n_embed, 3 * n_embed, rng)),
      c_proj_(nn::Linear(n_embed,     n_embed, rng)),
      rope_(n_embed / n_heads, max_seq_len) {
    if (n_embed % n_heads != 0) {
      throw std::invalid_argument("CausalAttention: n_embed must be divisible by n_heads");
    }
  };

  nn::Tensor forward(const nn::Tensor& x) override {
    const int B = x.extent(0); // batch
    const int T = x.extent(1); // time
    const int C = x.extent(2); // channels

    const int H = n_heads_; // heads
    const int D = n_embed_ / n_heads_; // head dimension

    // split into Q, K, V
    nn::Tensor qkv = c_attn_.forward(x); // [B, T, 3 * C]
    nn::Tensor q = qkv.slice_view(2, 0, C); // [B, T, C]
    nn::Tensor k = qkv.slice_view(2, C, C); // [B, T, C]
    nn::Tensor v = qkv.slice_view(2, 2 * C, C); // [B, T, C]

    // Reshape for multi-head attention
    q = q.reshape_view({B, T, H, D}).transpose_view(1, 2).contiguous(); // [B, H, T, D]
    k = k.reshape_view({B, T, H, D}).transpose_view(1, 2).contiguous(); // [B, H, T, D]
    v = v.reshape_view({B, T, H, D}).transpose_view(1, 2).contiguous(); // [B, H, T, D]

    // Rotary position embeddings on Q and K only (never V).
    q = rope_.apply(q); // [B, H, T, D]
    k = rope_.apply(k); // [B, H, T, D]

    // Scaled dot-product attention
    nn::Tensor attn_scores =
        q.mm(k.transpose_view(2, 3).contiguous()) / std::sqrt(static_cast<float>(D)); // [B, H, T, T]

    // Causal mask: position i may only attend to j <= i. tril_mask(T) is 1
    // where j <= i (keep) and 0 where j > i (future); masked_fill wants the
    // opposite convention (1 == replace), hence the "1 - ...".
    nn::Tensor future_mask = 1.0f - nn::tril_mask(T, x.device()); // [T, T]
    attn_scores = nn::masked_fill(attn_scores, future_mask, -1e9f);

    nn::Tensor attn = attn_scores.softmax(); // [B, H, T, T]
    nn::Tensor out = attn.mm(v); // [B, H, T, D]

    // Reshape back to [B, T, C]
    out = out.transpose_view(1, 2).reshape_view({B, T, C}); // [B, T, C]
    return c_proj_.forward(out);
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    c_attn_.collect_named(prefix + "c_attn", out);
    c_proj_.collect_named(prefix + "c_proj", out);
  }
private:
  int   n_heads_;
  int   n_embed_;
  float dropout_;

  nn::Linear c_attn_;
  nn::Linear c_proj_;
  RoPE rope_;
};