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

// An autoregressive LLM trained via genuine block-wise-independent
// DiffusionBlocks training (Shing et al., "DiffusionBlocks: Block-wise
// Neural Network Training via Diffusion Interpretation," ICLR 2026),
// distinct from DiffusionRecurrentLLM's single-shared-core recurrent-depth
// adaptation of the same paper (their Section 5.5): here there are
// n_layers DISTINCT (non-shared) transformer layers, partitioned into
// n_blocks contiguous groups. Each group is assigned its own slice of the
// EDM noise range via equi-probability partitioning (Section 3.3) and is a
// complete, standalone denoiser for that range -- at every training step,
// ONE block is sampled uniformly at random and only its (+ the always-on
// embedding/head/final-norm/timestep-embedder) parameters receive a
// gradient (Section 3.2, Figure 3's training algorithm), exactly the
// "genuinely independent block training" the paper is about. This means
// forward/backward memory scales with n_layers/n_blocks, not n_layers.
//
// The autoregressive adaptation (their Section 5.4 / Appendix E.4) uses
// the same "concatenate noisy and clean sequences under a modified causal
// mask" trick as DiffusionRecurrentLLM's 2T concatenation (build in
// diffusion_mask.h) to let a noisy token attend to clean past context
// without leaking its own answer, and the same cross-entropy-in-embedding-
// space objective (their Appendix B) instead of L2 in pixel space.
class DiffusionBlockedGPT : public nn::Module {
public:
  struct Config {
    int vocab_size;
    int n_embed;
    int n_heads;
    int mlp_hidden_dim;
    float dropout;
    int max_seq_len;   // T: clean-context length, before 2T concatenation

    int n_layers = 8;  // L: total transformer depth, across every block
    int n_blocks = 4;  // B: independently-trained blocks; must divide n_layers

    // EDM's sigma_data (see DiffusionRecurrentLLM::Config::sigma_data for
    // the same caveat: 0.5 is Karras et al.'s image-domain default).
    float sigma_data = 0.5f;
    // Clamp for the EDM loss weight -- see DiffusionRecurrentLLM::Config.
    float max_loss_weight = 20.0f;
  };

  struct Output {
    nn::Tensor loss;    // scalar, EDM-weighted -- what backward() should run on
    nn::Tensor logits;  // [B, T, vocab_size], denoised-Z logits
    // Diagnostic only, no gradient: see DiffusionRecurrentLLM::Output. Left
    // as a Tensor (not synced to host here) so callers can accumulate it
    // across micro-batches and sync once, not per call.
    nn::Tensor unweighted_loss;
  };

  DiffusionBlockedGPT(Config config, nn::Pcg32& rng)
    : token_embedding_(config.vocab_size, config.n_embed, rng),
      timestep_embedder_(config.n_embed, rng),
      ln_f_(config.n_embed),
      head_(token_embedding_.weight()),
      config_(config) {
    if (config.n_blocks < 1 || config.n_layers < config.n_blocks ||
        config.n_layers % config.n_blocks != 0) {
      throw std::invalid_argument(
          "DiffusionBlockedGPT: n_blocks must be a positive divisor of n_layers");
    }
    const int layers_per_block = config.n_layers / config.n_blocks;
    blocks_.resize(config.n_blocks);
    for (int b = 0; b < config.n_blocks; ++b) {
      blocks_[b].reserve(layers_per_block);
      for (int i = 0; i < layers_per_block; ++i) {
        blocks_[b].emplace_back(config.n_heads, config.n_embed, config.max_seq_len,
                                 config.mlp_hidden_dim, config.dropout, rng);
      }
    }
    q_boundaries_.resize(config.n_blocks + 1);
    sigma_boundaries_.resize(config.n_blocks + 1);
    equi_probability_boundaries(config.n_blocks, q_boundaries_, sigma_boundaries_);
    head_.set_requires_grad(true);
  }

  // idx: [B, T] token ids, used both as the sequence to denoise and as the
  // cross-entropy targets -- see DiffusionRecurrentLLM::forward_loss.
  Output forward_loss(const nn::Tensor& idx, nn::Pcg32& rng) {
    const int B = idx.extent(0);
    const int T = idx.extent(1);
    if (T > config_.max_seq_len) {
      throw std::invalid_argument("DiffusionBlockedGPT: sequence length exceeds maximum sequence length");
    }
    const nn::Device device = idx.device();

    // "During training, blocks are sampled uniformly at random for each
    // iteration" (Shing et al. 2026, Appendix E).
    const int b = int(rng.next_uint32() % uint32_t(config_.n_blocks));
    nn::Tensor sigma = sample_sigma_in_block(b, B, rng, device); // [B]

    nn::Tensor X = token_embedding_.forward(idx); // [B, T, C]
    X = l2_normalize(X); // prevents embedding collapse

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

    // Only block b's layers run: everything else in the model this step
    // stays exactly as it was after the previous step touched it.
    for (Block& layer : blocks_[b]) h = layer.forward(h, keep_mask, T);

    h = ln_f_.forward(h); // [B, 2T, C]
    nn::Tensor z_out = h.slice(1, T, T).contiguous(); // [B, T, C], the denoised Z half
    nn::Tensor logits = z_out.mm(head_, /*transB=*/true); // [B, T, vocab_size]

    CeResult ce = weighted_cross_entropy(logits, idx, sigma);

    return {ce.loss, logits, ce.unweighted};
  }

  // Autoregressive sampling: generates max_new_tokens tokens after
  // prompt_idx ([B, P], P >= 1). Mirrors DiffusionRecurrentLLM::generate's
  // reverse-ODE structure, except each Euler step picks whichever block
  // owns the current noise level (Figure 3's inference algorithm: "Select
  // block b where sigma_i in [sigma_b, sigma_{b-1}]") instead of always
  // using one shared core.
  nn::Tensor generate(const nn::Tensor& prompt_idx, int max_new_tokens, int num_sampling_steps,
                       int topk, nn::Pcg32& rng, float sigma_min = -1.0f, float sigma_max = -1.0f,
                       float rho = 7.0f) {
    const int B = prompt_idx.extent(0);
    const int P = prompt_idx.extent(1);
    if (P < 1) {
      throw std::invalid_argument("DiffusionBlockedGPT::generate: prompt must have at least one token");
    }
    const nn::Device device = prompt_idx.device();
    const int C = config_.n_embed;
    // Default to the same range the blocks were partitioned over, so the
    // inference schedule actually covers what each block was trained on.
    const std::vector<float> sigmas = karras_sigmas(
        num_sampling_steps, sigma_min > 0.0f ? sigma_min : kSigmaMin,
        sigma_max > 0.0f ? sigma_max : kSigmaMax, rho);

    nn::Tensor clean_embeds = l2_normalize(token_embedding_.forward(prompt_idx)); // [B, P, C]
    nn::Tensor out_ids = prompt_idx; // [B, P] -> [B, P + max_new_tokens]

    for (int t = 0; t < max_new_tokens; ++t) {
      const int ctx_len = clean_embeds.extent(1);
      const int T_cur = ctx_len + 1;
      if (T_cur > config_.max_seq_len) {
        throw std::invalid_argument("DiffusionBlockedGPT::generate: sequence length exceeds maximum sequence length");
      }

      const nn::Tensor placeholder = nn::Tensor::zeros({B, 1, C}, device);
      const nn::Tensor X_branch = nn::cat({clean_embeds, placeholder}, 1); // [B, T_cur, C]
      const nn::Tensor zero_prefix = nn::Tensor::zeros({B, T_cur - 1, C}, device);
      const nn::Tensor keep_mask = build_diffusion_keep_mask(T_cur, device);

      nn::Tensor z_new = sigmas[0] * nn::Tensor::randn({B, C}, rng, 1.0f, device);
      nn::Tensor denoised = z_new;

      for (int s = 0; s < num_sampling_steps; ++s) {
        const float sigma_cur = sigmas[s];
        const float sigma_next = sigmas[s + 1];
        const int b = block_for_sigma(sigma_cur);

        const float c_in = 1.0f / std::sqrt(sigma_cur * sigma_cur + config_.sigma_data * config_.sigma_data);
        const nn::Tensor z_slot = (z_new * c_in).reshape_view({B, 1, C});
        const nn::Tensor Z_branch = nn::cat({zero_prefix, z_slot}, 1);

        nn::Tensor h = nn::cat({X_branch, Z_branch}, 1);
        nn::Tensor cond = timestep_embedder_.forward(nn::Tensor::full({B}, sigma_cur, device));
        h = h + cond.reshape({B, 1, C});

        for (Block& layer : blocks_[b]) h = layer.forward(h, keep_mask, T_cur);

        h = ln_f_.forward(h);
        denoised = h.slice_view(1, 2 * T_cur - 1, 1).contiguous().reshape_view({B, C});

        const nn::Tensor d = (z_new - denoised) / sigma_cur;
        z_new = z_new + (sigma_next - sigma_cur) * d;
      }

      nn::Tensor logits = denoised.mm(head_, /*transB=*/true);
      nn::Tensor probs = logits.softmax();
      nn::Tensor values, indices;
      nn::ops::topk_rows(probs, topk, values, indices);
      const nn::Tensor local = nn::ops::multinomial(values);
      const nn::Tensor picked = nn::ops::gather_rows(indices, local);

      nn::Tensor new_embed = l2_normalize(token_embedding_.forward(picked.reshape_view({B, 1})));
      clean_embeds = nn::cat({clean_embeds, new_embed}, 1);
      out_ids = nn::cat({out_ids, picked.reshape_view({B, 1})}, 1);
    }

    return out_ids;
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    token_embedding_.collect_named(prefix + "token_embedding.", out);
    timestep_embedder_.collect_named(prefix + "timestep_embedder.", out);
    for (size_t b = 0; b < blocks_.size(); ++b) {
      for (size_t i = 0; i < blocks_[b].size(); ++i) {
        blocks_[b][i].collect_named(
            prefix + "blocks." + std::to_string(b) + "." + std::to_string(i) + ".", out);
      }
    }
    ln_f_.collect_named(prefix + "ln_f.", out);
    out.push_back({prefix + "head", &head_});
  }

  void set_training(bool on) override {
    training_ = on;
    for (auto& block : blocks_) {
      for (Block& layer : block) layer.set_training(on);
    }
  }

private:
  static nn::Tensor l2_normalize(const nn::Tensor& x, float eps = 1e-6f) {
    const int last = x.rank() - 1;
    nn::Tensor norm = x.pow(2.0f).sum(last, /*keepdim=*/true).sqrt();
    return x / (norm + eps);
  }

  static std::vector<float> karras_sigmas(int n_steps, float sigma_min, float sigma_max, float rho) {
    if (n_steps < 1) {
      throw std::invalid_argument("DiffusionBlockedGPT::generate: num_sampling_steps must be >= 1");
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
    nn::Tensor unweighted;  // diagnostic-only plain mean CE, kept as a Tensor -- see Output::unweighted_loss
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

  // --- Equi-probability block partitioning (Shing et al. 2026, Section
  // 3.3): boundaries chosen so each block covers an equal share of the
  // EDM log-normal training distribution's cumulative probability mass,
  // not a uniform slice of sigma or log-sigma -- concentrating more
  // blocks' capacity on the intermediate noise levels where most learning
  // happens, which their ablation (Table 7) shows matters a lot. ---

  // Karras et al. EDM defaults, matching what this paper uses throughout
  // (Appendix E): log-normal training distribution log(sigma) ~
  // N(kPMean, kPStd^2) over the range [kSigmaMin, kSigmaMax].
  static constexpr float kPMean = -1.2f;
  static constexpr float kPStd = 1.2f;
  static constexpr float kSigmaMin = 0.002f;
  static constexpr float kSigmaMax = 80.0f;

  // Standard normal CDF.
  static float norm_cdf(float x) {
    return 0.5f * std::erfc(-x * 0.7071067811865476f); // 1/sqrt(2)
  }

  // Standard normal PDF.
  static float norm_pdf(float x) {
    return 0.3989422804014327f * std::exp(-0.5f * x * x); // 1/sqrt(2*pi)
  }

  // Inverse standard normal CDF (probit). Bisection to bracket the root
  // (norm_cdf is monotonic, so this always converges) followed by a few
  // Newton polish steps for precision -- deliberately not a rational
  // approximation's magic coefficients, since this only ever runs a
  // handful of times (at construction, computing block boundaries) and a
  // wrong constant here would silently mis-partition every block.
  static float norm_ppf(float p) {
    if (!(p > 0.0f) || !(p < 1.0f)) {
      throw std::invalid_argument("norm_ppf: p must be in (0, 1)");
    }
    float lo = -10.0f, hi = 10.0f;
    for (int i = 0; i < 100; ++i) {
      const float mid = 0.5f * (lo + hi);
      if (norm_cdf(mid) < p) lo = mid; else hi = mid;
    }
    float x = 0.5f * (lo + hi);
    for (int i = 0; i < 4; ++i) {
      const float deriv = norm_pdf(x);
      if (deriv < 1e-12f) break;
      x -= (norm_cdf(x) - p) / deriv;
    }
    return x;
  }

  // Fills q_boundaries (length n_blocks+1, the underlying cumulative-
  // probability fractions) and sigma_boundaries (their sigma values) with
  // n_blocks+1 equi-probability boundaries spanning [kSigmaMin, kSigmaMax],
  // both increasing. Block b (0-indexed) owns [sigma_boundaries[b],
  // sigma_boundaries[b+1]].
  static void equi_probability_boundaries(int n_blocks, std::vector<float>& q_boundaries,
                                           std::vector<float>& sigma_boundaries) {
    const float q_min = norm_cdf((std::log(kSigmaMin) - kPMean) / kPStd);
    const float q_max = norm_cdf((std::log(kSigmaMax) - kPMean) / kPStd);
    for (int b = 0; b <= n_blocks; ++b) {
      const float q_b = q_min + (float(b) / float(n_blocks)) * (q_max - q_min);
      q_boundaries[b] = q_b;
      sigma_boundaries[b] = std::exp(kPMean + kPStd * norm_ppf(q_b));
    }
  }

  // Exact inverse-CDF sampling of per-example EDM noise levels, restricted
  // to block b's equi-probability range -- costs one norm_ppf per example,
  // not a rejection-sampling loop, regardless of how narrow the range is.
  nn::Tensor sample_sigma_in_block(int b, int B, nn::Pcg32& rng, nn::Device device) const {
    nn::Tensor h(nn::Shape({B}), nn::Device::CPU, nn::DType::F32);
    float* data = h.host_data();
    const float q_lo = q_boundaries_[b], q_hi = q_boundaries_[b + 1];
    for (int i = 0; i < B; ++i) {
      const float u = q_lo + rng.next_uniform() * (q_hi - q_lo);
      data[i] = std::exp(kPMean + kPStd * norm_ppf(u));
    }
    return h.to(device);
  }

  // The block whose range contains sigma (scans from the highest block
  // down; n_blocks is always small, so a linear scan is fine).
  int block_for_sigma(float sigma) const {
    for (int b = config_.n_blocks - 1; b >= 0; --b) {
      if (sigma >= sigma_boundaries_[b]) return b;
    }
    return 0;
  }

  nn::Embedding token_embedding_;
  TimestepEmbedder timestep_embedder_;
  std::vector<std::vector<Block>> blocks_; // [n_blocks][layers_per_block]
  nn::RMSNorm ln_f_;
  nn::Tensor head_;

  std::vector<float> q_boundaries_;      // length n_blocks + 1, increasing
  std::vector<float> sigma_boundaries_;  // length n_blocks + 1, increasing

  Config config_;
};
