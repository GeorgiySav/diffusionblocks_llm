#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/module.h>

#include "RoPE.h"

// Llama-style self-attention
class LlamaAttention : public nn::Module {
public:
  LlamaAttention(int hidden_size, int num_attention_heads, int num_key_value_heads,
                 int max_seq_len, float rope_theta, float dropout, nn::Pcg32& rng)
    : num_heads_(num_attention_heads),
      num_kv_heads_(num_key_value_heads),
      head_dim_(hidden_size / num_attention_heads),
      dropout_(dropout),
      q_proj_(hidden_size, num_attention_heads * (hidden_size / num_attention_heads), rng,
              /*bias=*/false),
      k_proj_(hidden_size, num_key_value_heads * (hidden_size / num_attention_heads), rng,
              /*bias=*/false),
      v_proj_(hidden_size, num_key_value_heads * (hidden_size / num_attention_heads), rng,
              /*bias=*/false),
      o_proj_(hidden_size, hidden_size, rng, /*bias=*/false),
      rope_(hidden_size / num_attention_heads, max_seq_len, rope_theta) {
    if (num_attention_heads <= 0 || hidden_size % num_attention_heads != 0) {
      throw std::invalid_argument("LlamaAttention: hidden_size must be divisible by "
                                  "num_attention_heads");
    }
    if (num_key_value_heads <= 0 || num_attention_heads % num_key_value_heads != 0) {
      throw std::invalid_argument("LlamaAttention: num_attention_heads must be divisible by "
                                  "num_key_value_heads");
    }
  }

  nn::Tensor forward(const nn::Tensor& x) override {
    const int B = x.extent(0);
    const int L = x.extent(1);
    const int C = x.extent(2);
    const int H = num_heads_;
    const int Hkv = num_kv_heads_;
    const int D = head_dim_;

    // [B, L, H*D] -> [B, H, L, D]; K/V carry Hkv heads instead of H.
    nn::Tensor q = split_heads(q_proj_.forward(x), B, L, H, D);
    nn::Tensor k = split_heads(k_proj_.forward(x), B, L, Hkv, D);
    nn::Tensor v = split_heads(v_proj_.forward(x), B, L, Hkv, D);

    q = rope_.apply(q);
    k = rope_.apply(k);

    // Broadcast each K/V head across the Q heads that share it.
    k = repeat_kv(k, B, Hkv, L, D, H / Hkv);
    v = repeat_kv(v, B, Hkv, L, D, H / Hkv);

    nn::Tensor out = nn::autograd::scaled_dot_product_attention(
        q, k, v, /*mask=*/nn::Tensor(), dropout_, /*is_causal=*/true, training()); // [B,H,L,D]

    out = out.transpose(1, 2).reshape({B, L, C}); // [B, L, C]
    return o_proj_.forward(out);
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    q_proj_.collect_named(prefix + "q_proj.", out);
    k_proj_.collect_named(prefix + "k_proj.", out);
    v_proj_.collect_named(prefix + "v_proj.", out);
    o_proj_.collect_named(prefix + "o_proj.", out);
  }

private:
  static nn::Tensor split_heads(const nn::Tensor& proj, int B, int L, int heads, int D) {
    return proj.reshape({B, L, heads, D}).transpose(1, 2).contiguous(); // [B, heads, L, D]
  }

  // [B, Hkv, L, D] -> [B, Hkv*rep, L, D]
  static nn::Tensor repeat_kv(const nn::Tensor& x, int B, int Hkv, int L, int D, int rep) {
    if (rep == 1) return x;
    return x.reshape({B, Hkv, 1, L, D})
            .expand({B, Hkv, rep, L, D})
            .contiguous()
            .reshape({B, Hkv * rep, L, D});
  }

  int num_heads_;
  int num_kv_heads_;
  int head_dim_;
  float dropout_;

  nn::Linear q_proj_, k_proj_, v_proj_, o_proj_;
  RoPE rope_;
};
