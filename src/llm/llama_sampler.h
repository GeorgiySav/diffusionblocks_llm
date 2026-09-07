#pragma once

#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nn/nn.h>

#include "llama_config.h"
#include "llm/Llama.h"
#include "llm/causal_lm.h"
#include "tokenizer.h"

namespace llm::llama_generate {

// Generates from a checkpoint using cfg's architecture, which must match what
// the checkpoint was trained with: the weights carry no metadata to check it.
inline void run_generate(const LlamaTrainConfig& cfg, const std::string& checkpoint_path,
                         const std::string& prompt, const std::string& encoding,
                         int max_new_tokens, float temperature, int top_k) {
  const nn::Device device = nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Generating on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);
  Llama model(llama_model_config_from(cfg), rng);
  model.to(device);

  std::printf("Loading weights from %s\n", checkpoint_path.c_str());
  nn::io::load_weights(checkpoint_path, model);
  model.eval();

  llm::Tokenizer tokenizer(encoding);
  std::printf("Encoding prompt: \"%s\"\n", prompt.c_str());
  const std::vector<int32_t> prompt_ids = tokenizer.encode(prompt);
  if (prompt_ids.empty()) {
    throw std::runtime_error("generate: prompt encoded to zero tokens");
  }
  nn::Tensor idx = nn::Tensor::from_i32(
                       std::span<const int32_t>(prompt_ids.data(), prompt_ids.size()),
                       {1, static_cast<int>(prompt_ids.size())})
                       .to(device);

  const nn::Tensor out = llm::causal::generate(model, idx, max_new_tokens, temperature, top_k);

  const nn::Tensor cpu_out = out.to(nn::Device::CPU);
  const std::vector<int32_t> out_ids(cpu_out.host_data_i32(),
                                     cpu_out.host_data_i32() + cpu_out.numel());
  std::printf("\n--- Generated text ---\n%s\n", tokenizer.decode(out_ids).c_str());
}

// CLI: generate_llama [prompt] [checkpoint] [temperature] [top_k] [tokenizer.json]
inline void run_generate_cli(int argc, char** argv) {
  LlamaTrainConfig cfg;

  const std::string prompt = (argc > 1) ? argv[1] : "Once upon a time";
  const std::string checkpoint_path = (argc > 2) ? argv[2] : cfg.checkpoint_path;
  const float temperature = (argc > 3) ? float(std::atof(argv[3])) : cfg.temperature;
  const int top_k = (argc > 4) ? std::atoi(argv[4]) : cfg.top_k;
  const std::string encoding = (argc > 5) ? argv[5] : cfg.tokenizer_path;

  std::printf("Llama:\n");
  std::printf("  vocab_size: %d\n", cfg.vocab_size);
  std::printf("  encoding: %s\n", encoding.c_str());
  std::printf("  hidden_size: %d\n", cfg.hidden_size);
  std::printf("  num_hidden_layers: %d\n", cfg.num_hidden_layers);
  std::printf("  heads: %d q / %d kv\n", cfg.num_attention_heads, cfg.num_key_value_heads);
  std::printf("  intermediate_size: %d\n", cfg.intermediate_size);
  std::printf("  context_length: %d\n", cfg.context_length);
  std::printf("  temperature: %.2f\n", temperature);
  std::printf("  top_k: %d%s\n", top_k, top_k == 1 ? " (greedy)" : "");

  run_generate(cfg, checkpoint_path, prompt, encoding, cfg.max_new_tokens, temperature, top_k);
}

}  // namespace llm::llama_generate
