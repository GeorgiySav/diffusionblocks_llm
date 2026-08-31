#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <nn/nn.h>

#include "dataloader.h"
#include "llm/diffusion_blocks_gpt.h"
#include "llm/diffusion_llm.h"
#include "llm/gpt.h"
#include "train_config.h"

namespace llm::train {

namespace detail {

struct LossResult {
  nn::Tensor loss;
  nn::Tensor aux;  // left unsynced (no .item()) so callers can accumulate across
                   // micro-batches/eval-iters and read the host value once
  bool has_aux = false;
  const char* aux_label = nullptr;
};

template <typename Model>
using LossFn = LossResult (*)(Model&, const nn::Tensor& /*xb*/, const nn::Tensor& /*yb*/, nn::Pcg32&);

inline LossResult gpt_loss(GPT& model, const nn::Tensor& xb, const nn::Tensor& yb, nn::Pcg32&) {
  nn::Tensor logits = model.forward(xb); // [B, T, V]
  const int B = logits.extent(0);
  const int T = logits.extent(1);
  const int V = logits.extent(2);
  return {nn::cross_entropy(logits.reshape({B * T, V}), yb.reshape_view({B * T}))};
}

inline LossResult diffusion_loss(DiffusionRecurrentLLM& model, const nn::Tensor& xb,
                                  const nn::Tensor&, nn::Pcg32& rng) {
  DiffusionRecurrentLLM::Output out = model.forward_loss(xb, rng); // denoises xb in place; no next-token shift
  return {out.loss, out.unweighted_loss, /*has_aux=*/true, "raw ce"};
}

inline LossResult diffusion_blocks_loss(DiffusionBlockedGPT& model, const nn::Tensor& xb,
                                         const nn::Tensor&, nn::Pcg32& rng) {
  DiffusionBlockedGPT::Output out = model.forward_loss(xb, rng); // denoises xb in place; no next-token shift
  return {out.loss, out.unweighted_loss, /*has_aux=*/true, "raw ce"};
}

struct EvalResult {
  float loss = 0.0f;
  float aux = 0.0f;
  bool has_aux = false;
  const char* aux_label = nullptr;
};

template <typename Model>
EvalResult estimate_loss(Model& model, nn::data::DataLoader<>& loader, int64_t iters, nn::Pcg32& rng,
                          LossFn<Model> compute_loss) {
  nn::autograd::NoGradScope no_grad;
  model.eval();

  // Accumulated as tensors and synced to host once at the end, not once per
  // iteration -- each .item() is a GPU pipeline drain.
  nn::Tensor total, total_aux;
  bool has_aux = false;
  const char* aux_label = nullptr;
  for (int64_t i = 0; i < iters; ++i) {
    if (!loader.has_next()) loader.reset();
    auto [xb, yb] = loader.next();
    LossResult r = compute_loss(model, xb, yb, rng);
    total = total.defined() ? total + r.loss : r.loss;
    if (r.has_aux) {
      has_aux = true;
      aux_label = r.aux_label;
      total_aux = total_aux.defined() ? total_aux + r.aux : r.aux;
    }
  }

  model.train();
  const float avg_loss = iters > 0 ? float(total.item() / double(iters)) : 0.0f;
  const float avg_aux = (has_aux && iters > 0) ? float(total_aux.item() / double(iters)) : 0.0f;
  return {avg_loss, avg_aux, has_aux, aux_label};
}

// block_wise: true only for DiffusionBlockedGPT, where a given step only
// ever runs one sampled block, so most parameters go untouched. Passed
// straight to AdamW::zero_grad's set_to_none -- see its doc comment in the
// nn library for why plain (in-place) zeroing would be wrong there
// specifically, and why every other model should keep it false.
template <typename Model>
void run_training_loop(const TrainConfig& cfg, Model& model, nn::Pcg32& rng,
                        nn::data::DataLoader<>& train_loader, nn::data::DataLoader<>& valid_loader,
                        LossFn<Model> compute_loss, bool block_wise = false) {
  namespace fs = std::filesystem;

  if (cfg.micro_batch_size <= 0 || cfg.micro_batch_size > cfg.batch_size ||
      cfg.batch_size % cfg.micro_batch_size != 0) {
    throw std::invalid_argument(
        "run_training: micro_batch_size must be a positive divisor of batch_size (got "
        "micro_batch_size=" + std::to_string(cfg.micro_batch_size) +
        ", batch_size=" + std::to_string(cfg.batch_size) + ")");
  }
  // Gradients accumulate over this many micro-batches (each already a
  // dataloader-sized batch of cfg.micro_batch_size) before every optimizer
  // step, so peak memory tracks micro_batch_size while training dynamics
  // (gradient noise, the LR schedule) still track the larger batch_size.
  const int64_t accum_steps = cfg.batch_size / cfg.micro_batch_size;

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

  const auto training_start = std::chrono::steady_clock::now();

  int64_t step = start_step;
  for (; step < cfg.max_steps; ++step) {
    if (cfg.max_seconds > 0.0) {
      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - training_start).count();
      if (elapsed >= cfg.max_seconds) {
        std::printf("step %6lld | stopping: max_seconds (%.0fs) reached\n",
                    static_cast<long long>(step), cfg.max_seconds);
        break;
      }
    }
    opt.set_lr(schedule(step));
    opt.zero_grad(/*set_to_none=*/block_wise);

    // Accumulated as tensors across micro-batches and synced to host only
    // once, after every kernel for this step (including opt.step()) is
    // already enqueued -- a .item() call is a GPU pipeline drain, so doing
    // it once at the end instead of once per micro-batch (and skipping the
    // logging accumulation itself past the tape via NoGradScope) keeps the
    // GPU fed instead of round-tripping to host after every micro-batch.
    nn::Tensor train_loss_sum, train_aux_sum;
    bool has_aux = false;
    const char* aux_label = nullptr;

    for (int64_t micro = 0; micro < accum_steps; ++micro) {
      if (!train_loader.has_next()) train_loader.reset();
      auto [xb, yb] = train_loader.next();

      nn::autograd::GradScope grad;
      LossResult result = compute_loss(model, xb, yb, rng);
      {
        nn::autograd::NoGradScope no_grad;
        train_loss_sum = train_loss_sum.defined() ? train_loss_sum + result.loss : result.loss;
        if (result.has_aux) {
          has_aux = true;
          aux_label = result.aux_label;
          train_aux_sum = train_aux_sum.defined() ? train_aux_sum + result.aux : result.aux;
        }
      }
      // Scaled so the accum_steps accumulated backward() calls (Tensor's
      // backward accumulates into leaf grads rather than overwriting them --
      // that's the entire mechanism this relies on) sum to the gradient of
      // the mean loss over the full effective batch_size, matching what one
      // non-accumulated backward() over that many examples would produce.
      (result.loss / float(accum_steps)).backward();
    }

    nn::optim::clip_grad_norm(params, cfg.grad_clip);
    // For block-wise models, whichever blocks no micro-batch touched this
    // step still have an undefined gradient (freed by zero_grad's
    // set_to_none above and never repopulated) -- step() already skips
    // those on its own, so plain step() is correct here regardless of
    // block_wise.
    opt.step();

    const float train_loss = float(train_loss_sum.item() / double(accum_steps));
    const float train_aux = has_aux ? float(train_aux_sum.item() / double(accum_steps)) : 0.0f;

    if (step % cfg.log_interval == 0) {
      if (has_aux) {
        std::printf("step %6lld | lr %.2e | train loss %.4f | train %s %.4f\n",
                    static_cast<long long>(step), opt.lr(), train_loss, aux_label, train_aux);
      } else {
        std::printf("step %6lld | lr %.2e | train loss %.4f\n",
                    static_cast<long long>(step), opt.lr(), train_loss);
      }
    }

    if (step > 0 && step % cfg.eval_interval == 0) {
      EvalResult val = estimate_loss(model, valid_loader, cfg.eval_iters, rng, compute_loss);
      if (val.has_aux) {
        std::printf("step %6lld | val loss %.4f | val %s %.4f\n",
                    static_cast<long long>(step), val.loss, val.aux_label, val.aux);
      } else {
        std::printf("step %6lld | val loss %.4f\n",
                    static_cast<long long>(step), val.loss);
      }
    }

    if (step > 0 && step % cfg.checkpoint_interval == 0) {
      nn::io::save_checkpoint(cfg.checkpoint_path, model, opt, step);
      std::printf("step %6lld | saved checkpoint to %s\n",
                  static_cast<long long>(step), cfg.checkpoint_path.c_str());
    }
  }

  nn::io::save_checkpoint(cfg.checkpoint_path, model, opt, step);
  std::printf("training complete; final checkpoint at %s\n",
              cfg.checkpoint_path.c_str());
}

inline void run_training_gpt(const TrainConfig& cfg) {
  const nn::Device device =
      nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Training GPT on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);

  // micro_batch_size, not batch_size: each loader batch feeds one
  // forward/backward micro-step (see run_training_loop's gradient
  // accumulation), so it's the size that actually has to fit in memory.
  auto train_loader = llm::data::create_tiny_stories_loader(
      cfg.train_tokens_path, cfg.micro_batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);
  auto valid_loader = llm::data::create_tiny_stories_loader(
      cfg.valid_tokens_path, cfg.micro_batch_size, cfg.context_length,
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

  run_training_loop<GPT>(cfg, model, rng, train_loader, valid_loader, gpt_loss);
}

inline void run_training_diffusion(const TrainConfig& cfg) {
  const nn::Device device =
      nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Training DiffusionRecurrentLLM on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);

  // micro_batch_size, not batch_size: each loader batch feeds one
  // forward/backward micro-step (see run_training_loop's gradient
  // accumulation), so it's the size that actually has to fit in memory.
  auto train_loader = llm::data::create_tiny_stories_loader(
      cfg.train_tokens_path, cfg.micro_batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);
  auto valid_loader = llm::data::create_tiny_stories_loader(
      cfg.valid_tokens_path, cfg.micro_batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);

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

  run_training_loop<DiffusionRecurrentLLM>(cfg, model, rng, train_loader, valid_loader, diffusion_loss);
}

inline void run_training_diffusion_blocks(const TrainConfig& cfg) {
  const nn::Device device =
      nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Training DiffusionBlockedGPT on device: %s\n", nn::device_name(device));

  nn::Pcg32 rng(cfg.seed);

  // micro_batch_size, not batch_size: each loader batch feeds one
  // forward/backward micro-step (see run_training_loop's gradient
  // accumulation), so it's the size that actually has to fit in memory.
  auto train_loader = llm::data::create_tiny_stories_loader(
      cfg.train_tokens_path, cfg.micro_batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);
  auto valid_loader = llm::data::create_tiny_stories_loader(
      cfg.valid_tokens_path, cfg.micro_batch_size, cfg.context_length,
      rng, /*shuffle=*/true, /*drop_last=*/true, /*stride=*/cfg.context_length, device);

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

  run_training_loop<DiffusionBlockedGPT>(cfg, model, rng, train_loader, valid_loader,
                                          diffusion_blocks_loss, /*block_wise=*/true);
}

}  // namespace detail

inline void run_training(const TrainConfig& cfg) {
  switch (cfg.model) {
    case ModelKind::GPT:               detail::run_training_gpt(cfg); return;
    case ModelKind::DiffusionRecurrent: detail::run_training_diffusion(cfg); return;
    case ModelKind::DiffusionBlocks:    detail::run_training_diffusion_blocks(cfg); return;
  }
  throw std::invalid_argument("run_training: unknown ModelKind");
}

}  // namespace llm::train
