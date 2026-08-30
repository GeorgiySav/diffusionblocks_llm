#pragma once

#include <cstdint>
#include <string>

#include "dataloader.h"

struct TrainConfig {
  // Model.
  int vocab_size = llm::data::kGPT2VocabSize; // matches the tokenization script
  int n_embed = 512;
  int n_heads = 16;
  int n_layers = 8;
  int mlp_hidden_dim = 4 * n_embed;
  float dropout = 0.1f;
  int context_length = 256;

  // Optimization.
  int batch_size = 32;
  int64_t max_steps = 3000;
  int64_t warmup_steps = 100;
  float peak_lr = 3e-4f;
  float min_lr = 3e-5f;
  float weight_decay = 0.1f;
  float grad_clip = 1.0f;

  // Logging / eval / checkpointing.
  int64_t log_interval = 10;
  int64_t eval_interval = 250;
  int64_t eval_iters = 50;
  int64_t checkpoint_interval = 1000;
  std::string checkpoint_path = "checkpoints/tinystories_gpt.ckpt";

  uint64_t seed = 42;
};
