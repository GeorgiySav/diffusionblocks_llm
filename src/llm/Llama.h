#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nn/module.h>

#include "LlamaDecoder.h"

struct LlamaConfig {
  int vocab_size = 0;
  int hidden_size = 512;
  int intermediate_size = 1376;
  int num_hidden_layers = 12;
  int num_attention_heads = 8;
  int num_key_value_heads = 8;
  int max_position_embeddings = 512;
  float rope_theta = 10000.0f;
  float dropout = 0.0f;

  bool tie_word_embeddings = true;
};

class Llama : public nn::Module {
public:
  Llama(const LlamaConfig& config, nn::Pcg32& rng)
    : config_(config),
      embed_tokens_(config.vocab_size, config.hidden_size, rng),
      norm_(config.hidden_size) {
    if (config.vocab_size <= 0) {
      throw std::invalid_argument("Llama: vocab_size must be positive");
    }
    if (config.num_hidden_layers <= 0) {
      throw std::invalid_argument("Llama: num_hidden_layers must be positive");
    }

    if (!config.tie_word_embeddings) {
      lm_head_ = std::make_unique<nn::Linear>(config.hidden_size, config.vocab_size, rng,
                                              /*bias=*/false);
    }

    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
      layers_.push_back(std::make_unique<LlamaDecoderLayer>(
          config.hidden_size, config.intermediate_size, config.num_attention_heads,
          config.num_key_value_heads, config.max_position_embeddings, config.rope_theta,
          config.dropout, rng));
    }
  }

  const LlamaConfig& config() const { return config_; }

  // input_ids: [B, L] I32 -> logits [B, L, vocab_size]
  nn::Tensor forward(const nn::Tensor& input_ids) override {
    const int L = input_ids.extent(1);
    if (L > config_.max_position_embeddings) {
      throw std::invalid_argument("Llama::forward: sequence length exceeds "
                                  "max_position_embeddings");
    }

    nn::Tensor h = embed_tokens_.forward(input_ids); // [B, L, C]
    for (auto& layer : layers_) {
      h = layer->forward(h);
    }
    return output_logits(norm_.forward(h));
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    embed_tokens_.collect_named(prefix + "embed_tokens.", out);
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->collect_named(prefix + "layers." + std::to_string(i) + ".", out);
    }
    norm_.collect_named(prefix + "norm.", out);
    if (lm_head_) lm_head_->collect_named(prefix + "lm_head.", out);
  }

  void set_training(bool on) override {
    training_ = on;
    embed_tokens_.set_training(on);
    for (auto& layer : layers_) layer->set_training(on);
    norm_.set_training(on);
    if (lm_head_) lm_head_->set_training(on);
  }

private:
  nn::Tensor output_logits(const nn::Tensor& hidden_states) {
    if (lm_head_) return lm_head_->forward(hidden_states);
    // the [V, C] table is the transpose of what a [C, V] projection wants
    return hidden_states.mm(embed_tokens_.weight(), /*transB=*/true); // [B, L, V]
  }

  LlamaConfig config_;

  nn::Embedding embed_tokens_;
  std::vector<std::unique_ptr<LlamaDecoderLayer>> layers_;
  nn::RMSNorm norm_;
  std::unique_ptr<nn::Linear> lm_head_; // null when tied to embed_tokens_
};
