#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nn/nn.h>

#include "dataloader.h"
#include "llama_config.h"
#include "llm/Llama.h"
#include "llm/causal_lm.h"

namespace llm::llama_train {

namespace detail {

inline float estimate_loss(Llama& model, nn::data::DataLoader<>& loader, int64_t iters) {
  nn::autograd::NoGradScope no_grad;
  model.eval();

  nn::Tensor total;
  for (int64_t i = 0; i < iters; ++i) {
    if (!loader.has_next()) loader.reset();
    auto [xb, yb] = loader.next();
    nn::Tensor loss = llm::causal::forward_loss(model, xb, yb);
    total = total.defined() ? total + loss : loss;
  }

  model.train();
  if (iters <= 0) return 0.0f;
  return float(total.item() / double(iters));
}

}  // namespace detail

inline void run_training(const LlamaTrainConfig& cfg) {
  namespace fs = std::filesystem;

  if (cfg.micro_batch_size <= 0 || cfg.micro_batch_size > cfg.batch_size ||
      cfg.batch_size % cfg.micro_batch_size != 0) {
    throw std::invalid_argument(
        "run_training: micro_batch_size must be a positive divisor of batch_size (got "
        "micro_batch_size=" + std::to_string(cfg.micro_batch_size) +
        ", batch_size=" + std::to_string(cfg.batch_size) + ")");
  }

  const nn::Device device = nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Training Llama on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);

  auto train_loader = llm::data::create_tiny_stories_loader(
      cfg.train_tokens_path, cfg.micro_batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);
  auto valid_loader = llm::data::create_tiny_stories_loader(
      cfg.valid_tokens_path, cfg.micro_batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);

  Llama model(llama_model_config_from(cfg), rng);
  model.to(device);

  std::vector<nn::Tensor*> params = model.parameters();
  int64_t param_count = 0;
  for (const nn::Tensor* p : params) param_count += p->numel();
  std::printf("parameters: %.2fM across %zu tensors\n", double(param_count) / 1e6, params.size());

  const int64_t accum_steps = cfg.batch_size / cfg.micro_batch_size;

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

  const std::string best_path = cfg.checkpoint_path + ".best";
  float best_val = std::numeric_limits<float>::infinity();

  int64_t step = start_step;
  for (; step < cfg.max_steps; ++step) {
    opt.set_lr(schedule(step));
    opt.zero_grad();

    const bool logging = (step % cfg.log_interval == 0);

    nn::Tensor loss_sum;
    for (int64_t micro = 0; micro < accum_steps; ++micro) {
      if (!train_loader.has_next()) train_loader.reset();
      auto [xb, yb] = train_loader.next();

      nn::autograd::GradScope grad;
      nn::Tensor loss;
      {
        nn::Autocast autocast;
        loss = llm::causal::forward_loss(model, xb, yb);
      }
      if (logging) {
        nn::autograd::NoGradScope no_grad;
        loss_sum = loss_sum.defined() ? loss_sum + loss : loss;
      }

      (loss / float(accum_steps)).backward();
    }

    nn::optim::clip_grad_norm(params, cfg.grad_clip);
    opt.step();

    if (logging) {
      const float train_loss = float(loss_sum.item() / double(accum_steps));
      std::printf("step %6lld | lr %.2e | train loss %.4f\n",
                  static_cast<long long>(step), opt.lr(), train_loss);
    }

    if (step > 0 && step % cfg.eval_interval == 0) {
      const float val = detail::estimate_loss(model, valid_loader, cfg.eval_iters);
      std::printf("step %6lld | val loss %.4f\n", static_cast<long long>(step), val);
      if (val < best_val) {
        best_val = val;
        nn::io::save_checkpoint(best_path, model, opt, step);
        std::printf("step %6lld | new best val loss %.4f -> %s\n",
                    static_cast<long long>(step), best_val, best_path.c_str());
      }
    }

    if (step > 0 && step % cfg.checkpoint_interval == 0) {
      nn::io::save_checkpoint(cfg.checkpoint_path, model, opt, step);
      std::printf("step %6lld | saved checkpoint to %s\n",
                  static_cast<long long>(step), cfg.checkpoint_path.c_str());
    }
  }

  nn::io::save_checkpoint(cfg.checkpoint_path, model, opt, step);
  std::printf("training complete; final checkpoint at %s\n", cfg.checkpoint_path.c_str());
  if (best_val < std::numeric_limits<float>::infinity()) {
    std::printf("best val loss %.4f at %s\n", best_val, best_path.c_str());
  }
}

}  // namespace llm::llama_train
