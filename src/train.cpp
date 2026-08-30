#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

#include <nn/nn.h>

#include "dataloader.h"
#include "llm/gpt.h"
#include "train_config.h"

namespace {

namespace fs = std::filesystem;

// One cross-entropy forward pass; shared by the train step and eval loop.
// Returns the (possibly tape-recorded) scalar loss tensor so the train step
// can still call .backward() on it.
nn::Tensor compute_loss(GPT& model, const nn::Tensor& xb, const nn::Tensor& yb) {
  nn::Tensor logits = model.forward(xb);  // [B, T, V]
  const int B = logits.extent(0);
  const int T = logits.extent(1);
  const int V = logits.extent(2);

  return nn::cross_entropy(logits.reshape({B * T, V}), yb.reshape_view({B * T}));
}

// Averages loss over `iters` batches with no gradient tracking.
float estimate_loss(GPT& model, nn::data::DataLoader<>& loader, int64_t iters) {
  nn::autograd::NoGradScope no_grad;
  model.eval();

  double total = 0.0;
  for (int64_t i = 0; i < iters; ++i) {
    if (!loader.has_next()) loader.reset();
    auto [xb, yb] = loader.next();
    total += compute_loss(model, xb, yb).item();
  }

  model.train();
  return iters > 0 ? float(total / double(iters)) : 0.0f;
}

void run_training(const TrainConfig& cfg) {

  const nn::Device device =
      nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Training on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);
  
  auto train_loader = llm::data::create_tiny_stories_loader(
      llm::data::kTinyStoriesTrainTokens, cfg.batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);
  auto valid_loader = llm::data::create_tiny_stories_loader(
      llm::data::kTinyStoriesValidTokens, cfg.batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);


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

  std::vector<nn::Tensor*> params = model.parameters();
  nn::optim::AdamW opt(params, cfg.peak_lr, cfg.weight_decay);
  nn::optim::Schedule schedule(cfg.peak_lr, cfg.max_steps, cfg.warmup_steps,
                                nn::optim::Decay::Cosine, cfg.min_lr);

  int64_t start_step = 0;
  if (fs::exists(cfg.checkpoint_path)) {
    start_step = nn::io::load_checkpoint(cfg.checkpoint_path, model, opt);
    std::printf("Resumed from %s at step %lld\n", cfg.checkpoint_path.c_str(),
                static_cast<long long>(start_step));
  } else {
    fs::path parent = fs::path(cfg.checkpoint_path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
  }

  for (int64_t step = start_step; step < cfg.max_steps; ++step) {
    opt.set_lr(schedule(step));

    if (!train_loader.has_next()) train_loader.reset();
    auto [xb, yb] = train_loader.next();

    opt.zero_grad();
    float train_loss;
    {
      nn::autograd::GradScope grad;
      nn::Tensor loss = compute_loss(model, xb, yb);
      train_loss = loss.item();
      loss.backward();
    }
    nn::optim::clip_grad_norm(params, cfg.grad_clip);
    opt.step();

    if (step % cfg.log_interval == 0) {
      std::printf("step %6lld | lr %.2e | train loss %.4f\n",
                  static_cast<long long>(step), opt.lr(), train_loss);
    }

    if (step > 0 && step % cfg.eval_interval == 0) {
      float val_loss = estimate_loss(model, valid_loader, cfg.eval_iters);
      std::printf("step %6lld | val loss %.4f\n",
                  static_cast<long long>(step), val_loss);
    }

    if (step > 0 && step % cfg.checkpoint_interval == 0) {
      nn::io::save_checkpoint(cfg.checkpoint_path, model, opt, step);
      std::printf("step %6lld | saved checkpoint to %s\n",
                  static_cast<long long>(step), cfg.checkpoint_path.c_str());
    }
  }

  nn::io::save_checkpoint(cfg.checkpoint_path, model, opt, cfg.max_steps);
  std::printf("training complete; final checkpoint at %s\n",
              cfg.checkpoint_path.c_str());
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // flush progress even when piped

  try {
    run_training(TrainConfig{});
  } catch (const std::exception& e) {
    std::fprintf(stderr, "training failed: %s\n", e.what());
    return 1;
  }

  return 0;
}
