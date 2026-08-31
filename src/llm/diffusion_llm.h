#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/module.h>
#include <nn/ops/ops.h>

#include "block.h"
#include "diffusion_mask.h"
#include "timestep_embedder.h"

class DiffusionRecurrentLLM : public nn::Module {
public:
  struct Config {
    int vocab_size;
    int n_embed;
    int n_heads;
    int mlp_hidden_dim;
    float dropout;
    int max_seq_len;        // T: clean-context length, before 2T concatenation
    int n_prelude_layers = 2;
    int n_coda_layers = 2;

    // EDM's sigma_data
    float sigma_data = 0.5f;

    // Clamp for the EDM loss weight
    float max_loss_weight = 20.0f;
  };

  struct Output {
    nn::Tensor loss;    // scalar, EDM-weighted -- what backward() will see
    nn::Tensor logits;  // [B, T, vocab_size], denoised-Z logits
    nn::Tensor unweighted_loss;  // diagnostic-only plain mean CE, not synced to host here
  };

  DiffusionRecurrentLLM(Config config, nn::Pcg32& rng)
    : token_embedding_(config.vocab_size, config.n_embed, rng),
      timestep_embedder_(config.n_embed, rng),
      core_(config.n_heads, config.n_embed, config.max_seq_len, config.mlp_hidden_dim,
            config.dropout, rng),
      ln_f_(config.n_embed),
      head_(token_embedding_.weight()),
      config_(config) {
    for (int i = 0; i < config.n_prelude_layers; ++i) {
      prelude_.emplace_back(config.n_heads, config.n_embed, config.max_seq_len,
                             config.mlp_hidden_dim, config.dropout, rng);
    }
    for (int i = 0; i < config.n_coda_layers; ++i) {
      coda_.emplace_back(config.n_heads, config.n_embed, config.max_seq_len,
                          config.mlp_hidden_dim, config.dropout, rng);
    }
    head_.set_requires_grad(true);
  }

  Output forward_loss(const nn::Tensor& idx, nn::Pcg32& rng) {
    const int B = idx.extent(0);
    const int T = idx.extent(1);
    if (T > config_.max_seq_len) {
      throw std::invalid_argument("DiffusionRecurrentLLM: sequence length exceeds maximum sequence length");
    }
    const nn::Device device = idx.device();

    nn::Tensor X = token_embedding_.forward(idx); // [B, T, C]
    X = l2_normalize(X); // prevents embedding collapse

    nn::Tensor sigma = sample_sigma(B, rng, device); // [B], EDM log-normal schedule
    nn::Tensor eps = nn::Tensor::randn({B, T, config_.n_embed}, rng, 1.0f, device);
    nn::Tensor Z = X + sigma.reshape_view({B, 1, 1}) * eps; // [B, T, C]

    nn::Tensor c_in = c_in_scale(sigma, config_.sigma_data); // [B]
    nn::Tensor Z_scaled = Z * c_in.reshape_view({B, 1, 1}); // [B, T, C]

    nn::Tensor h = nn::cat({X, Z_scaled}, 1); // [B, 2T, C]
    const nn::Tensor keep_mask = build_diffusion_keep_mask(T, device); // [2T, 2T]
    // Noise-level conditioning injected once (broadcast over the 2T axis)
    // instead of per-block AdaLN modulation -- cheaper and simpler, at the
    // cost of the per-layer adaptive control AdaLN-Zero gives a DiT.
    nn::Tensor cond = timestep_embedder_.forward(sigma); // [B, C]
    h = h + cond.reshape({B, 1, config_.n_embed});

    for (Block& block : prelude_) h = block.forward(h, keep_mask, T);
    h = core_.forward(h, keep_mask, T);
    for (Block& block : coda_) h = block.forward(h, keep_mask, T);

    h = ln_f_.forward(h); // [B, 2T, C]

    nn::Tensor z_out = h.slice(1, T, T).contiguous(); // [B, T, C], the denoised Z half
    nn::Tensor logits = z_out.mm(head_, /*transB=*/true); // [B, T, vocab_size]

    CeResult ce = weighted_cross_entropy(logits, idx, sigma);

    return {ce.loss, logits, ce.unweighted};
  }


  nn::Tensor generate(const nn::Tensor& prompt_idx, int max_new_tokens, int num_sampling_steps,
                       int topk, nn::Pcg32& rng, float sigma_min = 0.01f, float sigma_max = 10.0f,
                       float rho = 7.0f) {
    const int B = prompt_idx.extent(0);
    const int P = prompt_idx.extent(1);
    if (P < 1) {
      throw std::invalid_argument("DiffusionRecurrentLLM::generate: prompt must have at least one token");
    }
    const nn::Device device = prompt_idx.device();
    const int C = config_.n_embed;
    const std::vector<float> sigmas = karras_sigmas(num_sampling_steps, sigma_min, sigma_max, rho);

    nn::Tensor clean_embeds = l2_normalize(token_embedding_.forward(prompt_idx)); // [B, P, C]

    nn::Tensor out_ids = prompt_idx; // [B, P] -> [B, P + max_new_tokens]

    for (int t = 0; t < max_new_tokens; ++t) {
      const int ctx_len = clean_embeds.extent(1);
      const int T_cur = ctx_len + 1; // clean context, plus the slot being generated
      if (T_cur > config_.max_seq_len) {
        throw std::invalid_argument("DiffusionRecurrentLLM::generate: sequence length exceeds maximum sequence length");
      }

      const nn::Tensor placeholder = nn::Tensor::zeros({B, 1, C}, device); // never attended to
      const nn::Tensor X_branch = nn::cat({clean_embeds, placeholder}, 1); // [B, T_cur, C]
                                                                          
                                                          
      const nn::Tensor zero_prefix = nn::Tensor::zeros({B, T_cur - 1, C}, device);
      const nn::Tensor keep_mask = build_diffusion_keep_mask(T_cur, device);

      // EDM initialization: pure noise at the top of the schedule.
      nn::Tensor z_new = sigmas[0] * nn::Tensor::randn({B, C}, rng, 1.0f, device);
      nn::Tensor denoised = z_new;

      for (int s = 0; s < num_sampling_steps; ++s) {
        const float sigma_cur = sigmas[s];
        const float sigma_next = sigmas[s + 1];                                                                            
                                                                             
        const float c_in = 1.0f / std::sqrt(sigma_cur * sigma_cur + config_.sigma_data * config_.sigma_data);
        const nn::Tensor z_slot = (z_new * c_in).reshape_view({B, 1, C});
        const nn::Tensor Z_branch = nn::cat({zero_prefix, z_slot}, 1);

        nn::Tensor h = nn::cat({X_branch, Z_branch}, 1); // [B, 2 * T_cur, C]
        nn::Tensor cond = timestep_embedder_.forward(nn::Tensor::full({B}, sigma_cur, device));
        h = h + cond.reshape({B, 1, C});

        for (Block& block : prelude_) h = block.forward(h, keep_mask, T_cur);
        h = core_.forward(h, keep_mask, T_cur);
        for (Block& block : coda_) h = block.forward(h, keep_mask, T_cur);

        h = ln_f_.forward(h);
        denoised = h.slice_view(1, 2 * T_cur - 1, 1).contiguous().reshape_view({B, C});

        // First-order Euler step along the reverse ODE towards sigma_next.
        const nn::Tensor d = (z_new - denoised) / sigma_cur;
        z_new = z_new + (sigma_next - sigma_cur) * d;
      }

      nn::Tensor logits = denoised.mm(head_, /*transB=*/true); // [B, vocab_size]
      nn::Tensor probs = logits.softmax();
      nn::Tensor values, indices;
      nn::ops::topk_rows(probs, topk, values, indices);
      const nn::Tensor local = nn::ops::multinomial(values);
      const nn::Tensor picked = nn::ops::gather_rows(indices, local); // [B]

      nn::Tensor new_embed = l2_normalize(token_embedding_.forward(picked.reshape_view({B, 1})));
      clean_embeds = nn::cat({clean_embeds, new_embed}, 1);
      out_ids = nn::cat({out_ids, picked.reshape_view({B, 1})}, 1);
    }

    return out_ids;
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    token_embedding_.collect_named(prefix + "token_embedding.", out);
    for (size_t i = 0; i < prelude_.size(); ++i) {
      prelude_[i].collect_named(prefix + "prelude." + std::to_string(i) + ".", out);
    }
    timestep_embedder_.collect_named(prefix + "timestep_embedder.", out);
    core_.collect_named(prefix + "core.", out);
    for (size_t i = 0; i < coda_.size(); ++i) {
      coda_[i].collect_named(prefix + "coda." + std::to_string(i) + ".", out);
    }
    ln_f_.collect_named(prefix + "ln_f.", out);
    out.push_back({prefix + "head", &head_});
  }

  void set_training(bool on) override {
    training_ = on;
    for (Block& block : prelude_) block.set_training(on);
    core_.set_training(on);
    for (Block& block : coda_) block.set_training(on);
  }

private:
  static nn::Tensor l2_normalize(const nn::Tensor& x, float eps = 1e-6f) {
    const int last = x.rank() - 1;
    nn::Tensor norm = x.pow(2.0f).sum(last, /*keepdim=*/true).sqrt();
    return x / (norm + eps);
  }

  // EDM log-normal noise schedule: ln(sigma) ~ N(-1.2, 1.2^2).
  static nn::Tensor sample_sigma(int B, nn::Pcg32& rng, nn::Device device) {
    nn::Tensor h(nn::Shape({B}), nn::Device::CPU, nn::DType::F32);
    float* data = h.host_data();
    for (int i = 0; i < B; ++i) {
      data[i] = std::exp(-1.2f + 1.2f * rng.next_normal());
    }
    return h.to(device);
  }

  static std::vector<float> karras_sigmas(int n_steps, float sigma_min, float sigma_max, float rho) {
    if (n_steps < 1) {
      throw std::invalid_argument("DiffusionRecurrentLLM::generate: num_sampling_steps must be >= 1");
    }
    std::vector<float> sigmas(n_steps + 1);
    const float min_inv_rho = std::pow(sigma_min, 1.0f / rho);
    const float max_inv_rho = std::pow(sigma_max, 1.0f / rho);
    for (int i = 0; i < n_steps; ++i) {
      const float t = (n_steps > 1) ? float(i) / float(n_steps - 1) : 0.0f;
      sigmas[i] = std::pow(max_inv_rho + t * (min_inv_rho - max_inv_rho), rho);
    }
    sigmas[n_steps] = 0.0f;
    return sigmas;
  }

  static nn::Tensor c_in_scale(const nn::Tensor& sigma, float sigma_data) {
    return (sigma.pow(2.0f) + sigma_data * sigma_data).rsqrt(); // [B]
  }

  static nn::Tensor edm_loss_weight(const nn::Tensor& sigma, float sigma_data, float max_weight) {
    nn::Tensor w = (sigma.pow(2.0f) + sigma_data * sigma_data) /
                   (sigma * sigma_data).pow(2.0f);
    return w.clamp_max(max_weight); // [B]
  }

  struct CeResult {
    nn::Tensor loss;        // EDM-weighted, for backward()
    nn::Tensor unweighted;  // diagnostic-only plain mean CE, see Output::unweighted_loss --
                            // left as a Tensor (no .item() here) so callers can accumulate
                            // it across micro-batches and sync to host once, not per call
  };

  CeResult weighted_cross_entropy(const nn::Tensor& logits, const nn::Tensor& idx,
                                   const nn::Tensor& sigma) const {
    const int B = logits.extent(0);
    const int T = logits.extent(1);
    const int V = logits.extent(2);

    nn::Tensor logits_flat = logits.reshape({B * T, V});
    nn::Tensor idx_flat = idx.reshape_view({B * T});

    nn::Tensor w = edm_loss_weight(sigma, config_.sigma_data, config_.max_loss_weight); // [B]
    nn::Tensor weights = w.reshape_view({B, 1}).expand_view({B, T}).reshape_view({B * T});

    nn::Tensor loss = nn::cross_entropy(logits_flat, idx_flat, weights);

    nn::Tensor unweighted;
    {
      nn::autograd::NoGradScope no_grad;
      unweighted = nn::cross_entropy(logits_flat, idx_flat);
    }

    return {loss, unweighted};
  }

  nn::Embedding token_embedding_;
  std::vector<Block> prelude_;
  TimestepEmbedder timestep_embedder_;
  Block core_;
  std::vector<Block> coda_;
  nn::RMSNorm ln_f_;
  nn::Tensor head_;

  Config config_;
};
