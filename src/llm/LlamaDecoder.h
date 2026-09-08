#pragma once

#include <string>
#include <vector>

#include <nn/module.h>

#include "LlamaAttention.h"
#include "SwiGLU.h"

class LlamaDecoderLayer : public nn::Module {
public:
  LlamaDecoderLayer(int hidden_size, int intermediate_size, int num_attention_heads,
                    int num_key_value_heads, int max_seq_len, float rope_theta, float dropout,
                    nn::Pcg32& rng)
    : input_layernorm_(hidden_size),
      post_attention_layernorm_(hidden_size),
      self_attn_(hidden_size, num_attention_heads, num_key_value_heads, max_seq_len, rope_theta,
                 dropout, rng),
      mlp_(hidden_size, intermediate_size, rng, /*bias=*/false),
      down_proj_(intermediate_size, hidden_size, rng, /*bias=*/false) {}

  nn::Tensor forward(const nn::Tensor& x) override {
    nn::Tensor h = x + self_attn_.forward(input_layernorm_.forward(x));
    return h + down_proj_.forward(mlp_.forward(post_attention_layernorm_.forward(h)));
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    input_layernorm_.collect_named(prefix + "input_layernorm.", out);
    post_attention_layernorm_.collect_named(prefix + "post_attention_layernorm.", out);
    self_attn_.collect_named(prefix + "self_attn.", out);
    mlp_.collect_named(prefix + "mlp.", out);
    down_proj_.collect_named(prefix + "down_proj.", out);
  }

  void set_training(bool on) override {
    training_ = on;
    input_layernorm_.set_training(on);
    post_attention_layernorm_.set_training(on);
    self_attn_.set_training(on);
    mlp_.set_training(on);
    down_proj_.set_training(on);
  }

private:
  nn::RMSNorm input_layernorm_;
  nn::RMSNorm post_attention_layernorm_;
  LlamaAttention self_attn_;
  SwiGLU mlp_;            // gate_proj / up_proj
  nn::Linear down_proj_;
};
