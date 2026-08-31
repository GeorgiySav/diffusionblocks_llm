#pragma once

#include <vector>

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
    return forward(x, nn::tril_mask(x.extent(1), x.device()), /*rope_period=*/0);
  }

  nn::Tensor forward(const nn::Tensor& x, const nn::Tensor& keep_mask, int rope_period = 0) {
    const int B = x.extent(0); // batch
    const int L = x.extent(1); // time (T, or 2T under the concatenation trick)
    const int C = x.extent(2); // channels

    const int H = n_heads_; // heads
    const int D = n_embed_ / n_heads_; // head dimension

    nn::Tensor qkv = c_attn_.forward(x); // [B, L, 3 * C]
    nn::Tensor q = qkv.slice(2, 0, C); // [B, L, C]
    nn::Tensor k = qkv.slice(2, C, C); // [B, L, C]
    nn::Tensor v = qkv.slice(2, 2 * C, C); // [B, L, C]

    // Reshape for multi-head attention
    q = q.reshape({B, L, H, D}).transpose(1, 2).contiguous(); // [B, H, L, D]
    k = k.reshape({B, L, H, D}).transpose(1, 2).contiguous(); // [B, H, L, D]
    v = v.reshape({B, L, H, D}).transpose(1, 2).contiguous(); // [B, H, L, D]

    // Rotary position embeddings on Q and K only (never V).
    q = apply_rope(q, rope_period); // [B, H, L, D]
    k = apply_rope(k, rope_period); // [B, H, L, D]

    nn::Tensor out = nn::autograd::scaled_dot_product_attention(q, k, v, keep_mask, dropout_,
                                                                 /*is_causal=*/false, training()); // [B, H, L, D]

    // Reshape back to [B, L, C]
    out = out.transpose(1, 2).reshape({B, L, C}); // [B, L, C]
    return c_proj_.forward(out);
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    c_attn_.collect_named(prefix + "c_attn", out);
    c_proj_.collect_named(prefix + "c_proj", out);
  }
private:
  // x: [B, H, L, D]. Applies rope_ (which only knows positions 0..period-1)
  // independently to each period-length chunk along the sequence axis, so
  // position ids wrap back to 0 at each chunk boundary.
  nn::Tensor apply_rope(const nn::Tensor& x, int rope_period) const {
    const int L = x.extent(2);
    if (rope_period <= 0 || rope_period >= L) {
      return rope_.apply(x);
    }

    std::vector<nn::Tensor> chunks;
    for (int start = 0; start < L; start += rope_period) {
      chunks.push_back(rope_.apply(x.slice(2, start, rope_period)));
    }
    return nn::cat(chunks, 2);
  }

  int   n_heads_;
  int   n_embed_;
  float dropout_;

  nn::Linear c_attn_;
  nn::Linear c_proj_;
  RoPE rope_;
};