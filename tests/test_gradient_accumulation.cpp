#include "test_harness.h"

#include <cmath>
#include <vector>

#include <nn/nn.h>

#include "llm/gpt.h"

// main() lives in test_diffusion_llm.cpp -- llm_tests is one binary over
// every tests/*.cpp file sharing the NN_TEST registry.

namespace {

nn::Tensor make_token_ids(int B, int T, int vocab_size, nn::Pcg32& rng) {
  nn::Tensor h(nn::Shape({B, T}), nn::Device::CPU, nn::DType::I32);
  int32_t* data = h.host_data_i32();
  for (int i = 0; i < B * T; ++i) {
    data[i] = int32_t(rng.next_uint32() % uint32_t(vocab_size));
  }
  return h;
}

nn::Tensor gpt_loss(GPT& model, const nn::Tensor& xb, const nn::Tensor& yb) {
  nn::Tensor logits = model.forward(xb);
  const int B = logits.extent(0);
  const int T = logits.extent(1);
  const int V = logits.extent(2);
  return nn::cross_entropy(logits.reshape({B * T, V}), yb.reshape_view({B * T}));
}

}  // namespace

// Restates the exact mechanism llm::train::run_training_loop's gradient
// accumulation depends on (see src/llm/trainer.h): summing
// (loss_i / accum_steps).backward() over accum_steps micro-batches must
// equal one backward() over the mean loss of the whole concatenated batch.
// If Tensor::backward() ever stopped accumulating into leaf grads (rather
// than overwriting them), gradient accumulation would silently produce
// wrong gradients -- smaller effective batches with no error -- rather than
// failing loudly, so this is worth pinning down directly.
NN_TEST(gradient_accumulation_matches_single_full_batch_backward) {
  nn::Pcg32 data_rng(7);

  GPT::Config config{
      .vocab_size = 23,
      .n_embed = 16,
      .n_heads = 4,
      .n_layers = 2,
      .mlp_hidden_dim = 32,
      .dropout = 0.0f, // deterministic: no dropout noise to desync the two paths
      .max_seq_len = 8,
  };

  const int B = 4, T = 8;
  nn::Tensor idx = make_token_ids(B, T, config.vocab_size, data_rng);
  nn::Tensor targets = make_token_ids(B, T, config.vocab_size, data_rng);

  // Path A: one backward() over the full batch.
  nn::Pcg32 rng_a(42);
  GPT model_a(config, rng_a);
  {
    nn::autograd::GradScope grad;
    gpt_loss(model_a, idx, targets).backward();
  }

  // Path B: two backward() calls, each over half the batch with its loss
  // scaled by 1/2, accumulating into the same grads -- exactly what
  // accum_steps=2 does in run_training_loop. Same seed as model_a, so its
  // initialization (and thus its forward pass on identical inputs) is
  // bit-for-bit identical.
  nn::Pcg32 rng_b(42);
  GPT model_b(config, rng_b);
  {
    nn::autograd::GradScope grad;
    nn::Tensor idx0 = idx.slice_view(0, 0, 2), targets0 = targets.slice_view(0, 0, 2);
    nn::Tensor idx1 = idx.slice_view(0, 2, 2), targets1 = targets.slice_view(0, 2, 2);
    (gpt_loss(model_b, idx0, targets0) / 2.0f).backward();
    (gpt_loss(model_b, idx1, targets1) / 2.0f).backward();
  }

  std::vector<nn::NamedTensor> params_a = model_a.named_parameters();
  std::vector<nn::NamedTensor> params_b = model_b.named_parameters();
  NN_CHECK(!params_a.empty());
  NN_CHECK(params_a.size() == params_b.size());
  for (size_t i = 0; i < params_a.size(); ++i) {
    NN_CHECK(params_a[i].name == params_b[i].name);
    const float diff = (params_a[i].t->grad() - params_b[i].t->grad()).pow(2.0f).sum().sqrt().item();
    if (!(diff < 1e-3f)) {
      std::printf("  gradient mismatch for parameter \"%s\" (||diff||=%f)\n",
                  params_a[i].name.c_str(), diff);
    }
    NN_CHECK(diff < 1e-3f);
  }
}
