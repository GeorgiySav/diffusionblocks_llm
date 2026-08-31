#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nn/nn.h>

#include "dataloader.h"
#include "llm/diffusion_blocks_gpt.h"
#include "llm/diffusion_llm.h"
#include "llm/gpt.h"
#include "tokenizer.h"
#include "train_config.h"

namespace llm::generate {

namespace detail {

inline nn::Tensor encode_prompt(const std::string& prompt, const std::string& encoding, nn::Device device) {
  llm::Tokenizer tokenizer(encoding);
  std::printf("Encoding prompt: \"%s\"\n", prompt.c_str());
  const std::vector<int32_t> prompt_ids = tokenizer.encode(prompt);
  if (prompt_ids.empty()) {
    throw std::runtime_error("generate: prompt encoded to zero tokens");
  }

  nn::Tensor idx = nn::Tensor::from_i32(
      std::span<const int32_t>(prompt_ids.data(), prompt_ids.size()),
      {1, static_cast<int>(prompt_ids.size())});
  return idx.to(device);
}

inline std::string decode_ids(const nn::Tensor& out, const std::string& encoding) {
  llm::Tokenizer tokenizer(encoding);
  const nn::Tensor cpu_out = out.to(nn::Device::CPU);
  const std::vector<int32_t> out_ids(cpu_out.host_data_i32(), cpu_out.host_data_i32() + cpu_out.numel());
  return tokenizer.decode(out_ids);
}

inline void run_generate_gpt(const TrainConfig& cfg, const std::string& checkpoint_path,
                              const std::string& prompt, const std::string& encoding,
                              int max_new_tokens, int topk) {
  const nn::Device device =
      nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Generating on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);
  GPT::Config model_config{
      .vocab_size = cfg.vocab_size,
      .n_embed = cfg.n_embed,
      .n_heads = cfg.n_heads,
      .n_layers = cfg.n_layers,
      .mlp_hidden_dim = cfg.mlp_hidden_dim,
      .dropout = cfg.dropout,
      .max_seq_len = cfg.context_length,
  };
  GPT model(model_config, rng);
  model.to(device);

  std::printf("Loading weights from %s\n", checkpoint_path.c_str());
  nn::io::load_weights(checkpoint_path, model);
  model.eval();

  nn::Tensor idx = encode_prompt(prompt, encoding, device);

  nn::Tensor out;
  {
    nn::autograd::NoGradScope no_grad;
    out = model.generate(idx, max_new_tokens, topk, rng); // [1, T + max_new_tokens]
  }

  std::printf("\n--- Generated text ---\n%s\n", decode_ids(out, encoding).c_str());
}

inline void run_generate_diffusion(const TrainConfig& cfg, const std::string& checkpoint_path,
                                    const std::string& prompt, const std::string& encoding,
                                    int max_new_tokens, int topk, int num_sampling_steps) {
  const nn::Device device =
      nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Generating on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);
  DiffusionRecurrentLLM::Config model_config{
      .vocab_size = cfg.vocab_size,
      .n_embed = cfg.n_embed,
      .n_heads = cfg.n_heads,
      .mlp_hidden_dim = cfg.mlp_hidden_dim,
      .dropout = cfg.dropout,
      .max_seq_len = cfg.context_length,
      .n_prelude_layers = cfg.n_prelude_layers,
      .n_coda_layers = cfg.n_coda_layers,
      .sigma_data = cfg.sigma_data,
      .max_loss_weight = cfg.max_loss_weight,
  };
  DiffusionRecurrentLLM model(model_config, rng);
  model.to(device);

  std::printf("Loading weights from %s\n", checkpoint_path.c_str());
  nn::io::load_weights(checkpoint_path, model);
  model.eval();

  nn::Tensor idx = encode_prompt(prompt, encoding, device);

  nn::Tensor out;
  {
    nn::autograd::NoGradScope no_grad;
    out = model.generate(idx, max_new_tokens, num_sampling_steps, topk, rng); // [1, T + max_new_tokens]
  }

  std::printf("\n--- Generated text ---\n%s\n", decode_ids(out, encoding).c_str());
}

inline void run_generate_diffusion_blocks(const TrainConfig& cfg, const std::string& checkpoint_path,
                                           const std::string& prompt, const std::string& encoding,
                                           int max_new_tokens, int topk, int num_sampling_steps) {
  const nn::Device device =
      nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Generating on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);
  DiffusionBlockedGPT::Config model_config{
      .vocab_size = cfg.vocab_size,
      .n_embed = cfg.n_embed,
      .n_heads = cfg.n_heads,
      .mlp_hidden_dim = cfg.mlp_hidden_dim,
      .dropout = cfg.dropout,
      .max_seq_len = cfg.context_length,
      .n_layers = cfg.n_layers,
      .n_blocks = cfg.n_diffusion_blocks,
      .sigma_data = cfg.sigma_data,
      .max_loss_weight = cfg.max_loss_weight,
  };
  DiffusionBlockedGPT model(model_config, rng);
  model.to(device);

  std::printf("Loading weights from %s\n", checkpoint_path.c_str());
  nn::io::load_weights(checkpoint_path, model);
  model.eval();

  nn::Tensor idx = encode_prompt(prompt, encoding, device);

  nn::Tensor out;
  {
    nn::autograd::NoGradScope no_grad;
    out = model.generate(idx, max_new_tokens, num_sampling_steps, topk, rng); // [1, T + max_new_tokens]
  }

  std::printf("\n--- Generated text ---\n%s\n", decode_ids(out, encoding).c_str());
}

}  // namespace detail

inline const char* model_kind_name(ModelKind k) {
  switch (k) {
    case ModelKind::GPT:                return "gpt";
    case ModelKind::DiffusionRecurrent: return "diffusion";
    case ModelKind::DiffusionBlocks:    return "diffusion_blocks";
  }
  return "unknown";
}

inline void run_generate(int argc, char** argv) {
  TrainConfig cfg;

  if (argc > 1) cfg.model = model_kind_from_string(argv[1]); // "gpt", "diffusion", or "diffusion_blocks"
  const std::string prompt = (argc > 2) ? argv[2] : "Once upon a time";
  const std::string checkpoint_path = (argc > 3) ? argv[3] : cfg.checkpoint_path;
  if (argc > 4) cfg.vocab_size = std::atoi(argv[4]);
  const std::string encoding =
      (argc > 5) ? argv[5] : llm::data::encoding_for_vocab_size(cfg.vocab_size);
  const int num_sampling_steps = (argc > 6) ? std::atoi(argv[6]) : 16; // diffusion / diffusion_blocks only
  const int max_new_tokens = 200;
  const int topk = 40;

  std::printf("TrainConfig:\n");
  std::printf("  model: %s\n", model_kind_name(cfg.model));
  std::printf("  vocab_size: %d\n", cfg.vocab_size);
  std::printf("  encoding: %s\n", encoding.c_str());
  std::printf("  n_embed: %d\n", cfg.n_embed);
  std::printf("  n_heads: %d\n", cfg.n_heads);
  std::printf("  mlp_hidden_dim: %d\n", cfg.mlp_hidden_dim);
  std::printf("  dropout: %.2f\n", cfg.dropout);
  std::printf("  context_length: %d\n", cfg.context_length);

  switch (cfg.model) {
    case ModelKind::GPT:
      std::printf("  n_layers: %d\n", cfg.n_layers);
      detail::run_generate_gpt(cfg, checkpoint_path, prompt, encoding, max_new_tokens, topk);
      return;
    case ModelKind::DiffusionRecurrent:
      std::printf("  n_prelude_layers: %d\n", cfg.n_prelude_layers);
      std::printf("  n_coda_layers: %d\n", cfg.n_coda_layers);
      std::printf("  sigma_data: %.4f\n", cfg.sigma_data);
      std::printf("  max_loss_weight: %.2f\n", cfg.max_loss_weight);
      std::printf("  num_sampling_steps: %d\n", num_sampling_steps);
      detail::run_generate_diffusion(cfg, checkpoint_path, prompt, encoding, max_new_tokens, topk,
                                      num_sampling_steps);
      return;
    case ModelKind::DiffusionBlocks:
      std::printf("  n_layers: %d\n", cfg.n_layers);
      std::printf("  n_diffusion_blocks: %d\n", cfg.n_diffusion_blocks);
      std::printf("  sigma_data: %.4f\n", cfg.sigma_data);
      std::printf("  max_loss_weight: %.2f\n", cfg.max_loss_weight);
      std::printf("  num_sampling_steps: %d\n", num_sampling_steps);
      detail::run_generate_diffusion_blocks(cfg, checkpoint_path, prompt, encoding, max_new_tokens, topk,
                                             num_sampling_steps);
      return;
  }
  throw std::invalid_argument("run_generate: unknown ModelKind");
}

}  // namespace llm::generate
