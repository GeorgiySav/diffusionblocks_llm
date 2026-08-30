#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include <nn/nn.h>

#include "dataloader.h"
#include "llm/gpt.h"
#include "tokenizer.h"
#include "train_config.h"

namespace {

void run_generate(int argc, char** argv) {
  TrainConfig cfg;

  const std::string checkpoint_path = (argc > 2) ? argv[2] : cfg.checkpoint_path;
  const std::string prompt = (argc > 1) ? argv[1] : "Once upon a time";
  if (argc > 3) cfg.vocab_size = std::atoi(argv[3]);
  const std::string encoding =
      (argc > 4) ? argv[4] : llm::data::encoding_for_vocab_size(cfg.vocab_size);
  const int max_new_tokens = 200;
  const int topk = 40;

  std::printf("TrainConfig:\n");
  std::printf("  vocab_size: %d\n", cfg.vocab_size);
  std::printf("  encoding: %s\n", encoding.c_str());
  std::printf("  n_embed: %d\n", cfg.n_embed);
  std::printf("  n_heads: %d\n", cfg.n_heads);
  std::printf("  n_layers: %d\n", cfg.n_layers);
  std::printf("  mlp_hidden_dim: %d\n", cfg.mlp_hidden_dim);
  std::printf("  dropout: %.2f\n", cfg.dropout);
  std::printf("  context_length: %d\n", cfg.context_length);

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

  llm::Tokenizer tokenizer(encoding);

  std::printf("Encoding prompt: \"%s\"\n", prompt.c_str());
  const std::vector<int32_t> prompt_ids = tokenizer.encode(prompt);
  if (prompt_ids.empty()) {
    throw std::runtime_error("generate: prompt encoded to zero tokens");
  }

  nn::Tensor idx = nn::Tensor::from_i32(
      std::span<const int32_t>(prompt_ids.data(), prompt_ids.size()),
      {1, static_cast<int>(prompt_ids.size())});
  idx = idx.to(device);

  nn::Tensor out;
  {
    nn::autograd::NoGradScope no_grad;
    out = model.generate(idx, max_new_tokens, topk, rng);  // [1, T + max_new_tokens]
  }
  out = out.to(nn::Device::CPU);

  const std::vector<int32_t> out_ids(out.host_data_i32(), out.host_data_i32() + out.numel());
  const std::string text = tokenizer.decode(out_ids);

  std::printf("\n--- Generated text ---\n%s\n", text.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // flush progress even when piped

  try {
    run_generate(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "generate failed: %s\n", e.what());
    return 1;
  }

  return 0;
}
