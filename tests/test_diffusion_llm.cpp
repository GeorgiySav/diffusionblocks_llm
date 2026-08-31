#include "test_harness.h"

#include <cmath>
#include <typeinfo>

#include <nn/nn.h>

#include "llm/attention.h"
#include "llm/diffusion_llm.h"
#include "llm/diffusion_mask.h"

int main(int argc, char** argv) {
  try { return nn::test::run_all(argc, argv); }
  catch (const std::exception& e) {
    std::printf("\n!! Uncaught %s: %s\n", typeid(e).name(), e.what());
    return 2;
  }
}

namespace {

nn::Tensor make_token_ids(int B, int T, int vocab_size, nn::Pcg32& rng) {
  nn::Tensor h(nn::Shape({B, T}), nn::Device::CPU, nn::DType::I32);
  int32_t* data = h.host_data_i32();
  for (int i = 0; i < B * T; ++i) {
    data[i] = int32_t(rng.next_uint32() % uint32_t(vocab_size));
  }
  return h;
}

}  // namespace

// --- 2T mask: no-leakage checks --------------------------------------------

NN_TEST(diffusion_mask_x_is_plain_causal) {
  const int T = 4;
  const int L = 2 * T;
  nn::Tensor mask = build_diffusion_keep_mask(T, nn::Device::CPU);
  const float* data = mask.host_data();

  for (int row = 0; row < T; ++row) {
    for (int col = 0; col < L; ++col) {
      const float expected = (col < T && col <= row) ? 1.0f : 0.0f;
      NN_CHECK(data[row * L + col] == expected);
    }
  }
}

NN_TEST(diffusion_mask_z_sees_only_strictly_past_clean_tokens_and_itself) {
  const int T = 4;
  const int L = 2 * T;
  nn::Tensor mask = build_diffusion_keep_mask(T, nn::Device::CPU);
  const float* data = mask.host_data();

  for (int i = 0; i < T; ++i) {
    const int row = T + i;
    for (int col = 0; col < L; ++col) {
      const bool expect_keep = (col == row) || (col < i); // col < i implies col < T
      NN_CHECK(data[row * L + col] == (expect_keep ? 1.0f : 0.0f));
    }
  }
}

// Restates the leakage property the whole 2T trick depends on: a noisy row
// Z_i must not see any other noisy token, and must not see X_i itself or any
// later clean token (either would hand it the answer it is supposed to
// denoise).
NN_TEST(diffusion_mask_blocks_other_noisy_and_future_or_self_clean_tokens) {
  const int T = 5;
  const int L = 2 * T;
  nn::Tensor mask = build_diffusion_keep_mask(T, nn::Device::CPU);
  const float* data = mask.host_data();

  for (int i = 0; i < T; ++i) {
    const int row = T + i;
    for (int j = 0; j < T; ++j) {
      if (j != i) {
        NN_CHECK(data[row * L + (T + j)] == 0.0f); // other noisy tokens Z_j
      }
      if (j >= i) {
        NN_CHECK(data[row * L + j] == 0.0f); // X_i itself, or a future clean token
      }
    }
  }
}

// --- CausalAttention: keep_mask is actually enforced -----------------------

// The mask tests above check build_diffusion_keep_mask's bit pattern in
// isolation; these check that CausalAttention actually enforces it in a real
// forward pass, by perturbing every column a row must not see and requiring
// that row's output not move. This is exactly the kind of property that can
// silently break when the attention primitive underneath changes (e.g.
// swapping in a library's scaled_dot_product_attention) without anyone
// touching build_diffusion_keep_mask itself -- a keep_mask that's computed
// correctly but never threaded into the actual attention call would pass
// every test above and still leak.
NN_TEST(causal_attention_plain_causal_row_ignores_blocked_columns) {
  nn::Pcg32 rng(123);
  CausalAttention attn(/*n_heads=*/2, /*n_embed=*/8, /*max_seq_len=*/8, /*dropout=*/0.0f, rng);
  attn.eval();

  const int T = 4, L = 2 * T;
  const nn::Tensor keep_mask = build_diffusion_keep_mask(T, nn::Device::CPU);
  const float* mask_data = keep_mask.host_data();

  const nn::Tensor x = nn::Tensor::randn({1, L, 8}, rng, 1.0f);
  nn::Tensor out_a, out_b;
  {
    nn::autograd::NoGradScope no_grad;
    out_a = attn.forward(x, keep_mask, /*rope_period=*/T);
  }

  // Row 0 (X_0, plain causal) may only see column 0. Blow up every other
  // column and check row 0 of the output is unaffected.
  nn::Tensor x2 = x.clone();
  float* x2_data = x2.host_data();
  for (int col = 0; col < L; ++col) {
    if (mask_data[0 * L + col] != 0.0f) continue;
    for (int c = 0; c < 8; ++c) x2_data[col * 8 + c] += 1000.0f;
  }
  {
    nn::autograd::NoGradScope no_grad;
    out_b = attn.forward(x2, keep_mask, /*rope_period=*/T);
  }

  const float* a = out_a.host_data();
  const float* b = out_b.host_data();
  for (int c = 0; c < 8; ++c) {
    NN_CHECK(std::abs(a[c] - b[c]) < 1e-3f);
  }
}

NN_TEST(causal_attention_noisy_row_ignores_blocked_columns) {
  nn::Pcg32 rng(124);
  CausalAttention attn(/*n_heads=*/2, /*n_embed=*/8, /*max_seq_len=*/8, /*dropout=*/0.0f, rng);
  attn.eval();

  const int T = 4, L = 2 * T;
  const int i = 2, row = T + i; // Z_2: may see only X_0, X_1, and itself.
  const nn::Tensor keep_mask = build_diffusion_keep_mask(T, nn::Device::CPU);
  const float* mask_data = keep_mask.host_data();

  const nn::Tensor x = nn::Tensor::randn({1, L, 8}, rng, 1.0f);
  nn::Tensor out_a, out_b;
  {
    nn::autograd::NoGradScope no_grad;
    out_a = attn.forward(x, keep_mask, /*rope_period=*/T);
  }

  nn::Tensor x2 = x.clone();
  float* x2_data = x2.host_data();
  for (int col = 0; col < L; ++col) {
    if (mask_data[row * L + col] != 0.0f) continue;
    for (int c = 0; c < 8; ++c) x2_data[col * 8 + c] += 1000.0f;
  }
  {
    nn::autograd::NoGradScope no_grad;
    out_b = attn.forward(x2, keep_mask, /*rope_period=*/T);
  }

  const float* a = out_a.host_data() + row * 8;
  const float* b = out_b.host_data() + row * 8;
  for (int c = 0; c < 8; ++c) {
    NN_CHECK(std::abs(a[c] - b[c]) < 1e-3f);
  }
}

// --- DiffusionRecurrentLLM: forward pass + loss shape ----------------------

NN_TEST(diffusion_llm_forward_loss_produces_finite_scalar_and_backprops) {
  nn::Pcg32 rng(1234);

  DiffusionRecurrentLLM::Config config{
      .vocab_size = 37,
      .n_embed = 16,
      .n_heads = 4,
      .mlp_hidden_dim = 32,
      .dropout = 0.0f,
      .max_seq_len = 6,
      .n_prelude_layers = 1,
      .n_coda_layers = 1,
  };
  DiffusionRecurrentLLM model(config, rng);

  const int B = 3, T = 6;
  nn::Tensor idx = make_token_ids(B, T, config.vocab_size, rng);

  DiffusionRecurrentLLM::Output out;
  {
    nn::autograd::GradScope grad;
    out = model.forward_loss(idx, rng);

    NN_CHECK(out.loss.rank() == 0);
    NN_CHECK(out.logits.extent(0) == B);
    NN_CHECK(out.logits.extent(1) == T);
    NN_CHECK(out.logits.extent(2) == config.vocab_size);

    const float loss_value = out.loss.item();
    NN_CHECK(std::isfinite(loss_value));

    out.loss.backward();
  }

  std::vector<nn::Tensor*> params = model.parameters();
  NN_CHECK(!params.empty());
  const float grad_norm = nn::optim::grad_norm(params);
  NN_CHECK(std::isfinite(grad_norm));
  NN_CHECK(grad_norm > 0.0f);
}

// A single forward/backward pass right after construction can miss a
// connectivity bug that only shows up once every module has actually been
// exercised -- in particular, a raw slice_view/reshape_view silently
// severing the tape (Tensor::view_like does not carry over AutogradMeta, so
// the "_view" family must never sit on a path back to a trainable weight --
// see attention.h / block.h / diffusion_llm.h). timestep_embedder_'s
// conditioning path is exactly this shape (a reshape between its output and
// where it's added into h), which is why this test caught a real regression
// once: an earlier version used cond.reshape_view(...) there and every
// timestep_embedder_ parameter came back with a zero gradient. A few
// optimizer steps before checking also lets the model move off its initial
// (near-identity-ish) parameters, so the check isn't just validating
// gradients at initialization.
NN_TEST(diffusion_llm_gradients_reach_every_parameter_after_warmup) {
  nn::Pcg32 rng(4321);

  DiffusionRecurrentLLM::Config config{
      .vocab_size = 37,
      .n_embed = 16,
      .n_heads = 4,
      .mlp_hidden_dim = 32,
      .dropout = 0.0f,
      .max_seq_len = 6,
      .n_prelude_layers = 1,
      .n_coda_layers = 1,
  };
  DiffusionRecurrentLLM model(config, rng);

  const int B = 3, T = 6;
  nn::Tensor idx = make_token_ids(B, T, config.vocab_size, rng);

  std::vector<nn::Tensor*> params = model.parameters();
  nn::optim::AdamW opt(params, /*lr=*/0.1f);

  for (int step = 0; step < 3; ++step) {
    nn::autograd::GradScope grad;
    opt.zero_grad();
    model.forward_loss(idx, rng).loss.backward();
    opt.step();
  }

  opt.zero_grad();
  {
    nn::autograd::GradScope grad;
    model.forward_loss(idx, rng).loss.backward();
  }

  std::vector<nn::NamedTensor> named_params = model.named_parameters();
  NN_CHECK(!named_params.empty());
  for (const nn::NamedTensor& p : named_params) {
    const float g_norm = p.t->grad().pow(2.0f).sum().sqrt().item();
    if (!(std::isfinite(g_norm) && g_norm > 0.0f)) {
      std::printf("  zero/non-finite gradient for parameter \"%s\" (norm=%f)\n",
                  p.name.c_str(), g_norm);
    }
    NN_CHECK(std::isfinite(g_norm));
    NN_CHECK(g_norm > 0.0f);
  }
}

NN_TEST(diffusion_llm_rejects_sequences_longer_than_max_seq_len) {
  nn::Pcg32 rng(7);

  DiffusionRecurrentLLM::Config config{
      .vocab_size = 16,
      .n_embed = 8,
      .n_heads = 2,
      .mlp_hidden_dim = 16,
      .dropout = 0.0f,
      .max_seq_len = 4,
      .n_prelude_layers = 1,
      .n_coda_layers = 1,
  };
  DiffusionRecurrentLLM model(config, rng);

  nn::Tensor idx = make_token_ids(1, 5, config.vocab_size, rng); // 5 > max_seq_len
  NN_CHECK_THROWS(model.forward_loss(idx, rng), std::invalid_argument);
}

// --- DiffusionRecurrentLLM: generation (reverse-ODE sampling) --------------

// max_new_tokens is large enough that the clean-context length (P + tokens
// generated so far) crosses 8 partway through: nn::cat caps out at 8 tensors
// at a time (the tape records a node's inputs inline), which an
// accumulate-as-separate-chunks-then-cat-them-all approach would blow past
// after a handful of tokens even though generation itself never gets close
// to max_seq_len.
NN_TEST(diffusion_llm_generate_produces_valid_tokens_of_the_right_shape) {
  nn::Pcg32 rng(99);

  DiffusionRecurrentLLM::Config config{
      .vocab_size = 23,
      .n_embed = 16,
      .n_heads = 4,
      .mlp_hidden_dim = 32,
      .dropout = 0.0f,
      .max_seq_len = 20,
      .n_prelude_layers = 1,
      .n_coda_layers = 1,
  };
  DiffusionRecurrentLLM model(config, rng);
  model.eval();

  const int B = 2, P = 3, max_new_tokens = 10, num_sampling_steps = 3, topk = 5;
  nn::Tensor prompt = make_token_ids(B, P, config.vocab_size, rng);

  nn::Tensor out;
  {
    nn::autograd::NoGradScope no_grad;
    out = model.generate(prompt, max_new_tokens, num_sampling_steps, topk, rng);
  }

  NN_CHECK(out.extent(0) == B);
  NN_CHECK(out.extent(1) == P + max_new_tokens);

  // The prompt itself must be echoed back unchanged at the front.
  const int32_t* prompt_data = prompt.host_data_i32();
  const int32_t* out_data = out.host_data_i32();
  for (int b = 0; b < B; ++b) {
    for (int p = 0; p < P; ++p) {
      NN_CHECK(out_data[b * out.extent(1) + p] == prompt_data[b * P + p]);
    }
  }

  // Every generated id must be a valid vocabulary index.
  for (int64_t i = 0; i < out.numel(); ++i) {
    NN_CHECK(out_data[i] >= 0 && out_data[i] < config.vocab_size);
  }
}

NN_TEST(diffusion_llm_generate_rejects_empty_prompt) {
  nn::Pcg32 rng(11);

  DiffusionRecurrentLLM::Config config{
      .vocab_size = 10,
      .n_embed = 8,
      .n_heads = 2,
      .mlp_hidden_dim = 16,
      .dropout = 0.0f,
      .max_seq_len = 8,
      .n_prelude_layers = 1,
      .n_coda_layers = 1,
  };
  DiffusionRecurrentLLM model(config, rng);

  nn::Tensor empty_prompt = make_token_ids(1, 0, config.vocab_size, rng);
  NN_CHECK_THROWS(model.generate(empty_prompt, 1, 2, 3, rng), std::invalid_argument);
}
