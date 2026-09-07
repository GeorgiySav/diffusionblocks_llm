#include "test_harness.h"

#include <cmath>
#include <string>
#include <typeinfo>
#include <vector>

#include <nn/nn.h>

#include "llm/Llama.h"
#include "llm/causal_lm.h"

// llm_tests is one binary over every tests/*.cpp, sharing the NN_TEST
// registry, so main() lives here and the other files hold only tests.
int main(int argc, char** argv) {
  try { return nn::test::run_all(argc, argv); }
  catch (const std::exception& e) {
    std::printf("\n!! Uncaught %s: %s\n", typeid(e).name(), e.what());
    return 2;
  }
}

namespace {

LlamaConfig small_config() {
  LlamaConfig cfg;
  cfg.vocab_size = 64;
  cfg.hidden_size = 32;
  cfg.intermediate_size = 64;
  cfg.num_hidden_layers = 3;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 4;
  cfg.max_position_embeddings = 16;
  cfg.dropout = 0.0f;
  return cfg;
}

nn::Tensor make_token_ids(int B, int L, int vocab_size, nn::Pcg32& rng) {
  nn::Tensor h(nn::Shape({B, L}), nn::Device::CPU, nn::DType::I32);
  int32_t* data = h.host_data_i32();
  for (int i = 0; i < B * L; ++i) {
    data[i] = int32_t(rng.next_uint32() % uint32_t(vocab_size));
  }
  return h;
}

bool all_finite(const nn::Tensor& t) {
  const nn::Tensor cpu = t.to(nn::Device::CPU);
  const float* d = cpu.host_data();
  for (int64_t i = 0; i < cpu.numel(); ++i) {
    if (!std::isfinite(d[i])) return false;
  }
  return true;
}

}  // namespace

// ---- shape and plumbing ------------------------------------------------

NN_TEST(forward_returns_logits_for_every_position) {
  nn::Pcg32 rng(1);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  model.eval();

  const int B = 2, L = 8;
  nn::Tensor ids = make_token_ids(B, L, cfg.vocab_size, rng);
  nn::Tensor logits = model.forward(ids);

  NN_CHECK(logits.rank() == 3);
  NN_CHECK(logits.extent(0) == B);
  NN_CHECK(logits.extent(1) == L);
  NN_CHECK(logits.extent(2) == cfg.vocab_size);
  NN_CHECK(all_finite(logits));
}

NN_TEST(sequences_longer_than_the_context_window_are_rejected) {
  nn::Pcg32 rng(2);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);

  nn::Tensor too_long = make_token_ids(1, cfg.max_position_embeddings + 1, cfg.vocab_size, rng);
  NN_CHECK_THROWS(model.forward(too_long), std::invalid_argument);
}

// A stray bias would be a silent architectural deviation nothing else in the
// suite would catch, so it is asserted directly on the parameter names.
NN_TEST(no_parameter_is_a_bias) {
  nn::Pcg32 rng(3);
  Llama model(small_config(), rng);

  for (const nn::NamedTensor& p : model.named_parameters()) {
    const std::string& name = p.name;
    const bool is_bias = name.size() >= 2 && name.compare(name.size() - 2, 2, ".b") == 0;
    if (is_bias) {
      ::nn::test::report(__FILE__, __LINE__, "unexpected bias parameter: " + name);
    }
  }
}

NN_TEST(tying_the_head_removes_a_parameter_tensor) {
  nn::Pcg32 rng(4);
  LlamaConfig tied = small_config();
  tied.tie_word_embeddings = true;
  LlamaConfig untied = small_config();
  untied.tie_word_embeddings = false;

  Llama tied_model(tied, rng);
  Llama untied_model(untied, rng);

  NN_CHECK(untied_model.named_parameters().size() ==
           tied_model.named_parameters().size() + 1);
}

// ---- causality ---------------------------------------------------------
//
// Position t's logits must be a function of tokens 0..t only. A backwards
// leak collapses training loss and the model cannot generate at all.
NN_TEST(logits_at_each_position_ignore_later_tokens) {
  nn::Pcg32 rng(5);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  model.eval();

  const int B = 1, L = 8;
  nn::Tensor ids = make_token_ids(B, L, cfg.vocab_size, rng);
  nn::Tensor baseline = model.forward(ids).to(nn::Device::CPU);

  // Change only the final token.
  nn::Tensor perturbed(nn::Shape({B, L}), nn::Device::CPU, nn::DType::I32);
  const int32_t* src = ids.to(nn::Device::CPU).host_data_i32();
  int32_t* dst = perturbed.host_data_i32();
  for (int i = 0; i < L; ++i) dst[i] = src[i];
  dst[L - 1] = int32_t((dst[L - 1] + 1) % cfg.vocab_size);
  nn::Tensor after = model.forward(perturbed).to(nn::Device::CPU);

  const int V = cfg.vocab_size;
  // Positions 0..L-2 must be bit-for-bit unaffected.
  for (int t = 0; t < L - 1; ++t) {
    for (int v = 0; v < V; ++v) {
      const int64_t i = int64_t(t) * V + v;
      NN_CHECK_CLOSE(baseline.host_data()[i], after.host_data()[i], 1e-6);
    }
  }

  // ...and the final position must actually change, or the check above would
  // pass for a model that ignores its input entirely.
  bool last_changed = false;
  for (int v = 0; v < V; ++v) {
    const int64_t i = int64_t(L - 1) * V + v;
    if (std::fabs(baseline.host_data()[i] - after.host_data()[i]) > 1e-5) last_changed = true;
  }
  NN_CHECK(last_changed);
}

// ---- grouped-query attention -------------------------------------------

NN_TEST(grouped_query_attention_runs_at_every_valid_head_ratio) {
  // 4 query heads: 4 kv heads is MHA, 1 is multi-query, 2 is in between.
  for (int kv_heads : {4, 2, 1}) {
    nn::Pcg32 rng(6);
    LlamaConfig cfg = small_config();
    cfg.num_key_value_heads = kv_heads;
    Llama model(cfg, rng);
    model.eval();

    nn::Tensor ids = make_token_ids(2, 6, cfg.vocab_size, rng);
    nn::Tensor logits = model.forward(ids);
    NN_CHECK(logits.extent(2) == cfg.vocab_size);
    NN_CHECK(all_finite(logits));
  }
}

NN_TEST(head_counts_that_do_not_divide_are_rejected) {
  nn::Pcg32 rng(7);

  LlamaConfig bad_kv = small_config();
  bad_kv.num_key_value_heads = 3; // 4 query heads is not a multiple of 3
  NN_CHECK_THROWS(Llama(bad_kv, rng), std::invalid_argument);

  LlamaConfig bad_width = small_config();
  bad_width.hidden_size = 30; // not divisible by 4 heads
  NN_CHECK_THROWS(Llama(bad_width, rng), std::invalid_argument);

  LlamaConfig bad_vocab = small_config();
  bad_vocab.vocab_size = 0;
  NN_CHECK_THROWS(Llama(bad_vocab, rng), std::invalid_argument);
}

// Sharing K/V across query heads must actually save parameters.
NN_TEST(grouped_query_attention_has_fewer_parameters_than_multi_head) {
  nn::Pcg32 rng(8);
  LlamaConfig mha = small_config();
  LlamaConfig mqa = small_config();
  mqa.num_key_value_heads = 1;

  const auto count = [](Llama& m) {
    int64_t n = 0;
    for (const nn::Tensor* p : m.parameters()) n += p->numel();
    return n;
  };

  Llama mha_model(mha, rng);
  Llama mqa_model(mqa, rng);
  NN_CHECK(count(mqa_model) < count(mha_model));
}

// ---- objective ---------------------------------------------------------
//
// TokenDataset hands over y already shifted, so forward_loss must not shift
// again. An off-by-one here looks like fast convergence, not an error.
NN_TEST(forward_loss_scores_targets_against_matching_positions) {
  nn::Pcg32 rng(9);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  model.eval();

  const int B = 2, L = 6;
  nn::Tensor ids = make_token_ids(B, L, cfg.vocab_size, rng);
  nn::Tensor targets = make_token_ids(B, L, cfg.vocab_size, rng);

  nn::Tensor loss = llm::causal::forward_loss(model, ids, targets);
  NN_CHECK(loss.rank() == 0);

  // Recomputed from the model's own logits, to pin the flattening.
  nn::Tensor logits = model.forward(ids);
  const int V = cfg.vocab_size;
  nn::Tensor expected = nn::cross_entropy(logits.reshape({B * L, V}),
                                          targets.contiguous().reshape_view({B * L}));
  NN_CHECK_CLOSE(loss.item(), expected.item(), 1e-5);

  // An untrained model is near-uniform, so the loss should sit near ln(V).
  NN_CHECK_CLOSE(loss.item(), std::log(double(V)), 0.25);
}

NN_TEST(forward_loss_rejects_mismatched_shapes) {
  nn::Pcg32 rng(10);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);

  nn::Tensor ids = make_token_ids(2, 6, cfg.vocab_size, rng);
  nn::Tensor wrong = make_token_ids(2, 5, cfg.vocab_size, rng);
  NN_CHECK_THROWS(llm::causal::forward_loss(model, ids, wrong), std::invalid_argument);
}

NN_TEST(gradients_reach_every_parameter) {
  nn::Pcg32 rng(11);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  model.train();

  nn::Tensor ids = make_token_ids(2, 6, cfg.vocab_size, rng);
  nn::Tensor targets = make_token_ids(2, 6, cfg.vocab_size, rng);

  {
    nn::autograd::GradScope grad;
    llm::causal::forward_loss(model, ids, targets).backward();
  }

  // set_requires_grad allocates a zero buffer eagerly, so .defined() proves
  // nothing; the question is whether anything was accumulated.
  for (const nn::NamedTensor& p : model.named_parameters()) {
    NN_CHECK(p.t->grad().defined());
    if (!p.t->grad().defined()) continue;
    const nn::Tensor g = p.t->grad().to(nn::Device::CPU);
    double magnitude = 0.0;
    for (int64_t i = 0; i < g.numel(); ++i) magnitude += std::fabs(double(g.host_data()[i]));
    if (!(magnitude > 0.0)) {
      ::nn::test::report(__FILE__, __LINE__, "no gradient reached " + p.name);
    }
  }
}

// Catches sign errors, detached graphs, and optimizer misconfiguration that
// every shape-level test above would pass.
NN_TEST(loss_decreases_when_overfitting_a_single_batch) {
  nn::Pcg32 rng(12);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  model.train();

  nn::Tensor ids = make_token_ids(2, 6, cfg.vocab_size, rng);
  nn::Tensor targets = make_token_ids(2, 6, cfg.vocab_size, rng);

  std::vector<nn::Tensor*> params = model.parameters();
  nn::optim::AdamW opt(params, 3e-3f, 0.0f);

  double first = 0.0, last = 0.0;
  for (int i = 0; i < 20; ++i) {
    opt.zero_grad();
    nn::Tensor loss;
    {
      nn::autograd::GradScope grad;
      loss = llm::causal::forward_loss(model, ids, targets);
      loss.backward();
    }
    opt.step();
    const double value = loss.item();
    if (i == 0) first = value;
    last = value;
  }

  NN_CHECK(last < first);
  NN_CHECK(std::isfinite(last));
}

// ---- generation --------------------------------------------------------

NN_TEST(generate_appends_exactly_the_requested_number_of_tokens) {
  nn::Pcg32 rng(13);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  model.eval();

  nn::Tensor prompt = make_token_ids(1, 3, cfg.vocab_size, rng);
  nn::Tensor out = llm::causal::generate(model, prompt, /*max_new_tokens=*/5,
                                         /*temperature=*/1.0f, /*top_k=*/8);
  NN_CHECK(out.extent(0) == 1);
  NN_CHECK(out.extent(1) == 3 + 5);

  // The prompt must survive unchanged at the front.
  for (int i = 0; i < 3; ++i) {
    NN_CHECK(out.to(nn::Device::CPU).host_data_i32()[i] ==
             prompt.to(nn::Device::CPU).host_data_i32()[i]);
  }
}

// Past the context window the prompt is cropped rather than generation
// failing, so the sequence keeps growing.
NN_TEST(generate_slides_the_window_instead_of_overflowing_it) {
  nn::Pcg32 rng(14);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  model.eval();

  const int prompt_len = cfg.max_position_embeddings - 2;
  nn::Tensor prompt = make_token_ids(1, prompt_len, cfg.vocab_size, rng);
  const int new_tokens = 6; // takes it well past the window

  nn::Tensor out = llm::causal::generate(model, prompt, new_tokens, 1.0f, 4);
  NN_CHECK(out.extent(1) == prompt_len + new_tokens);
  NN_CHECK(out.extent(1) > cfg.max_position_embeddings);
}

NN_TEST(generate_with_top_k_one_is_deterministic) {
  nn::Pcg32 rng(15);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  model.eval();

  nn::Tensor prompt = make_token_ids(1, 3, cfg.vocab_size, rng);

  // Greedy decoding never consults the sampler, so two calls must agree even
  // though nn::ops::multinomial has advanced its own internal stream between
  // them. This is the only decoding mode that is reproducible at all.
  nn::Tensor a = llm::causal::generate(model, prompt, 6, 1.0f, /*top_k=*/1);
  nn::Tensor b = llm::causal::generate(model, prompt, 6, 1.0f, /*top_k=*/1);

  const nn::Tensor a_cpu = a.to(nn::Device::CPU);
  const nn::Tensor b_cpu = b.to(nn::Device::CPU);
  NN_CHECK(a_cpu.numel() == b_cpu.numel());
  for (int64_t i = 0; i < a_cpu.numel(); ++i) {
    NN_CHECK(a_cpu.host_data_i32()[i] == b_cpu.host_data_i32()[i]);
  }
}

NN_TEST(generate_rejects_a_nonpositive_temperature) {
  nn::Pcg32 rng(16);
  LlamaConfig cfg = small_config();
  Llama model(cfg, rng);
  nn::Tensor prompt = make_token_ids(1, 3, cfg.vocab_size, rng);
  NN_CHECK_THROWS(llm::causal::generate(model, prompt, 2, 0.0f, 4), std::invalid_argument);
}
