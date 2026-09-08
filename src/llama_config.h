#pragma once

#include <cstdint>
#include <string>

#include "dataloader.h"
#include "llm/Llama.h"

// Configuration for the Llama-style causal LM
struct LlamaTrainConfig {
  // Model.
  int vocab_size = 4096;
  int hidden_size = 640;
  int num_attention_heads = 10;  // head_dim 64
  int num_key_value_heads = 10;  // == heads: plain MHA. Lower this for GQA.
  int num_hidden_layers = 12;
  int intermediate_size = 1728;  // 8/3 * 640 = 1707, rounded up to a multiple of 32
  int context_length = 256;
  float rope_theta = 10000.0f;
  float dropout = 0.0f;
  bool tie_word_embeddings = true;

  // Optimization.
  int batch_size = 256;
  int micro_batch_size = 32;
  int64_t max_steps = 18700;
  int64_t warmup_steps = 200;
  float peak_lr = 3e-4f;
  float min_lr = 3e-5f;
  float weight_decay = 0.1f;
  float grad_clip = 1.0f;

  // Generation.
  int max_new_tokens = 200;
  float temperature = 0.8f;
  int top_k = 40;

  // Logging / eval / checkpointing.
  int64_t log_interval = 50;
  int64_t eval_interval = 400;
  int64_t eval_iters = 100;
  int64_t checkpoint_interval = 400;

  std::string checkpoint_path = "checkpoints/llama.ckpt";

  std::string train_tokens_path = llm::data::kTinyStoriesTrainTokens;
  std::string valid_tokens_path = llm::data::kTinyStoriesValidTokens;
  std::string tokenizer_path = "data/TinyStories/tokenizer-4096.json";

  uint64_t seed = 42;
};

inline LlamaConfig llama_model_config_from(const LlamaTrainConfig& cfg) {
  return LlamaConfig{
      .vocab_size = cfg.vocab_size,
      .hidden_size = cfg.hidden_size,
      .intermediate_size = cfg.intermediate_size,
      .num_hidden_layers = cfg.num_hidden_layers,
      .num_attention_heads = cfg.num_attention_heads,
      .num_key_value_heads = cfg.num_key_value_heads,
      .max_position_embeddings = cfg.context_length,
      .rope_theta = cfg.rope_theta,
      .dropout = cfg.dropout,
      .tie_word_embeddings = cfg.tie_word_embeddings,
  };
}
