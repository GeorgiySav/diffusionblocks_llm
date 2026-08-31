#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "dataloader.h"

// DiffusionRecurrent: one shared "core" block applied once per step (see
// src/llm/diffusion_llm.h) -- the recurrent-depth adaptation from Shing et
// al. 2026 Section 5.5.
// DiffusionBlocks: n_layers distinct transformer layers partitioned into
// n_diffusion_blocks independently-trained blocks, one sampled per step
// (see src/llm/diffusion_blocks_gpt.h) -- the general block-wise-training
// scheme the same paper is actually named after (their Section 3).
enum class ModelKind { DiffusionRecurrent, GPT, DiffusionBlocks };

inline ModelKind model_kind_from_string(const std::string& s) {
  if (s == "diffusion") return ModelKind::DiffusionRecurrent;
  if (s == "gpt") return ModelKind::GPT;
  if (s == "diffusion_blocks") return ModelKind::DiffusionBlocks;
  throw std::invalid_argument(
      "unknown model kind \"" + s + "\" (expected \"diffusion\", \"gpt\", or \"diffusion_blocks\")");
}

struct TrainConfig {
  ModelKind model = ModelKind::DiffusionRecurrent;

  // Model.
  int vocab_size = 4096; //llm::data::kGPT2VocabSize;
  int n_embed = 256;
  int n_heads = 8;
  int mlp_hidden_dim = 4 * n_embed;
  float dropout = 0.1f;
  int context_length = 256;

  int n_layers = 8; // GPT and DiffusionBlocks: total transformer depth

  // DiffusionRecurrentLLM only
  int n_prelude_layers = 2;
  int n_coda_layers = 2;

  // DiffusionBlocks only: independently-trained blocks n_layers is split
  // into (must divide n_layers evenly) -- see
  // src/llm/diffusion_blocks_gpt.h.
  int n_diffusion_blocks = 4;

  // EDM's sigma_data
  float sigma_data = 1.0f / std::sqrt(n_embed);
  // Clamp on the EDM loss weight
  float max_loss_weight = 20.0f;

  // Optimization.
  int batch_size = 512; // effective/statistical batch size
  // Actual per-forward/backward batch size
  int micro_batch_size = 32;
  int64_t max_steps = 20000;
  // Wall-clock safety cap, in addition to max_steps: 0 means no cap. Checked
  // once per step, so a run stops at most one step late, not mid-step.
  // Useful when you want "at most N minutes" regardless of how per-step
  // throughput estimates compare to actual (machine load, thermal
  // throttling, etc.) -- max_steps still sizes the LR schedule.
  double max_seconds = 0.0;
  int64_t warmup_steps = 300;
  float peak_lr = 3e-4f;
  float min_lr = 3e-5f;
  float weight_decay = 0.1f;
  float grad_clip = 1.0f;

  // Logging / eval / checkpointing.
  int64_t log_interval = 10;
  int64_t eval_interval = 250;
  int64_t eval_iters = 50;
  int64_t checkpoint_interval = 1000;

  std::string checkpoint_path = "checkpoints/tinystories.ckpt";

  // Which pre-tokenized corpus to train on -- see dataloader.h. Point both
  // at llm::data::kTinyShakespeare{Train,Valid}Tokens for a quick end-to-end
  // validation run (full epochs in seconds) instead of a full TinyStories run.
  std::string train_tokens_path = llm::data::kTinyStoriesTrainTokens;
  std::string valid_tokens_path = llm::data::kTinyStoriesValidTokens;

  uint64_t seed = 42;
};
