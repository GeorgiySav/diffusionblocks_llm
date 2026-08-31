#include "test_harness.h"

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include <nn/nn.h>

#include "llm/diffusion_blocks_gpt.h"

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

// Parses the block index out of a collect_named-style name containing
// "blocks.<b>.<layer>...."; returns -1 for a name with no such component
// (the always-shared parts: embedding, head, ln_f, timestep embedder).
int block_index_of(const std::string& name) {
  const std::string marker = "blocks.";
  const size_t pos = name.find(marker);
  if (pos == std::string::npos) return -1;
  const size_t start = pos + marker.size();
  const size_t end = name.find('.', start);
  return std::stoi(name.substr(start, end - start));
}

}  // namespace

NN_TEST(diffusion_blocks_gpt_rejects_a_block_count_that_does_not_divide_n_layers) {
  nn::Pcg32 rng(1);
  DiffusionBlockedGPT::Config config{
      .vocab_size = 16,
      .n_embed = 8,
      .n_heads = 2,
      .mlp_hidden_dim = 16,
      .dropout = 0.0f,
      .max_seq_len = 8,
      .n_layers = 5,
      .n_blocks = 2,
  };
  NN_CHECK_THROWS(DiffusionBlockedGPT(config, rng), std::invalid_argument);
}

NN_TEST(diffusion_blocks_gpt_forward_loss_produces_finite_scalar_and_backprops) {
  nn::Pcg32 rng(1234);

  DiffusionBlockedGPT::Config config{
      .vocab_size = 23,
      .n_embed = 16,
      .n_heads = 4,
      .mlp_hidden_dim = 32,
      .dropout = 0.0f,
      .max_seq_len = 6,
      .n_layers = 4,
      .n_blocks = 2,
  };
  DiffusionBlockedGPT model(config, rng);

  const int B = 3, T = 6;
  nn::Tensor idx = make_token_ids(B, T, config.vocab_size, rng);

  // As run_training_loop does for block-wise models: free every gradient
  // first, so afterwards "defined" really means "backward() touched this."
  model.zero_grad(/*set_to_none=*/true);

  DiffusionBlockedGPT::Output out;
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

  bool any_active = false;
  for (nn::NamedTensor& p : model.named_parameters()) {
    if (p.t->meta() && p.t->meta()->grad.defined()) { any_active = true; break; }
  }
  NN_CHECK(any_active);
}

// The whole point of DiffusionBlocks: only the sampled block's parameters
// (plus the always-on shared components) ever end up with a gradient after
// a call -- never a *different* block's. This is exactly the mechanism
// run_training_loop relies on for correct block-wise optimizer updates: it
// frees every gradient via zero_grad(set_to_none=true) before each step, so
// AdamW::step()'s existing "skip if grad undefined" check does the right
// thing on its own (see Optimizer::zero_grad's doc comment in the nn
// library). If block selection silently regressed to "always block 0" or
// "everything active," training would still run and loss would still go
// down -- nothing would look obviously broken without a test that checks
// this directly.
NN_TEST(diffusion_blocks_gpt_active_params_never_include_a_different_block) {
  nn::Pcg32 rng(7);

  DiffusionBlockedGPT::Config config{
      .vocab_size = 23,
      .n_embed = 16,
      .n_heads = 4,
      .mlp_hidden_dim = 32,
      .dropout = 0.0f,
      .max_seq_len = 6,
      .n_layers = 6,
      .n_blocks = 3,
  };
  DiffusionBlockedGPT model(config, rng);

  const int B = 2, T = 6;
  nn::Tensor idx = make_token_ids(B, T, config.vocab_size, rng);

  std::vector<nn::NamedTensor> named = model.named_parameters();
  NN_CHECK(!named.empty());

  std::set<int> blocks_seen;
  for (int trial = 0; trial < 30; ++trial) {
    // As run_training_loop does for block-wise models: free every gradient
    // first, so afterwards "defined" really means "this call's backward
    // touched it," not leftover state from a previous trial's block.
    model.zero_grad(/*set_to_none=*/true);
    {
      nn::autograd::GradScope grad;
      DiffusionBlockedGPT::Output out = model.forward_loss(idx, rng);
      out.loss.backward();
    }

    // head_ ties to token_embedding_'s weight storage (see
    // DiffusionBlockedGPT's constructor), so both named entries share one
    // AutogradMeta and therefore one grad tensor -- checking definedness
    // through it needs no manual deduplication, unlike a raw Tensor*
    // comparison would.
    const auto is_active = [](const nn::NamedTensor& p) {
      return p.t->meta() && p.t->meta()->grad.defined();
    };

    int selected = -1;
    for (nn::NamedTensor& p : named) {
      const int b = block_index_of(p.name);
      if (b >= 0 && is_active(p)) { selected = b; break; }
    }
    NN_CHECK(selected >= 0);
    blocks_seen.insert(selected);

    for (nn::NamedTensor& p : named) {
      const int b = block_index_of(p.name);
      if (b < 0) {
        NN_CHECK(is_active(p)); // shared components: always active
      } else if (b == selected) {
        NN_CHECK(is_active(p)); // the sampled block: active
      } else {
        NN_CHECK(!is_active(p)); // every other block: not active
      }
    }
  }
  // With n_blocks=3 and 30 independent uniform draws, seeing fewer than
  // all 3 distinct blocks has probability under 1e-4 -- this would fail
  // loudly if selection collapsed to a fixed block instead of sampling.
  NN_CHECK(blocks_seen.size() == 3u);
}

NN_TEST(diffusion_blocks_gpt_rejects_sequences_longer_than_max_seq_len) {
  nn::Pcg32 rng(7);

  DiffusionBlockedGPT::Config config{
      .vocab_size = 16,
      .n_embed = 8,
      .n_heads = 2,
      .mlp_hidden_dim = 16,
      .dropout = 0.0f,
      .max_seq_len = 4,
      .n_layers = 4,
      .n_blocks = 2,
  };
  DiffusionBlockedGPT model(config, rng);

  nn::Tensor idx = make_token_ids(1, 5, config.vocab_size, rng); // 5 > max_seq_len
  NN_CHECK_THROWS(model.forward_loss(idx, rng), std::invalid_argument);
}

NN_TEST(diffusion_blocks_gpt_generate_produces_valid_tokens_of_the_right_shape) {
  nn::Pcg32 rng(99);

  DiffusionBlockedGPT::Config config{
      .vocab_size = 23,
      .n_embed = 16,
      .n_heads = 4,
      .mlp_hidden_dim = 32,
      .dropout = 0.0f,
      .max_seq_len = 12,
      .n_layers = 4,
      .n_blocks = 2,
  };
  DiffusionBlockedGPT model(config, rng);
  model.eval();

  const int B = 2, P = 3, max_new_tokens = 4, num_sampling_steps = 6, topk = 5;
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

NN_TEST(diffusion_blocks_gpt_generate_rejects_empty_prompt) {
  nn::Pcg32 rng(11);

  DiffusionBlockedGPT::Config config{
      .vocab_size = 10,
      .n_embed = 8,
      .n_heads = 2,
      .mlp_hidden_dim = 16,
      .dropout = 0.0f,
      .max_seq_len = 8,
      .n_layers = 2,
      .n_blocks = 2,
  };
  DiffusionBlockedGPT model(config, rng);

  nn::Tensor empty_prompt = make_token_ids(1, 0, config.vocab_size, rng);
  NN_CHECK_THROWS(model.generate(empty_prompt, 1, 2, 3, rng), std::invalid_argument);
}
