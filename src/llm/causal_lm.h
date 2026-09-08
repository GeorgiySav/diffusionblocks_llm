#pragma once

#include <algorithm>
#include <stdexcept>

#include <nn/autograd/functions.h>
#include <nn/ops/ops.h>

#include "Llama.h"

namespace llm::causal {

// Mean cross-entropy of predicting targets from input_ids.
inline nn::Tensor forward_loss(Llama& model, const nn::Tensor& input_ids,
                               const nn::Tensor& targets) {
  if (input_ids.rank() != 2 || targets.rank() != 2) {
    throw std::invalid_argument("causal::forward_loss: expected [B, L] input_ids and targets");
  }
  if (input_ids.extent(0) != targets.extent(0) || input_ids.extent(1) != targets.extent(1)) {
    throw std::invalid_argument("causal::forward_loss: input_ids and targets must have the "
                                "same shape");
  }
  const int B = input_ids.extent(0);
  const int L = input_ids.extent(1);

  nn::Tensor logits = model.forward(input_ids); // [B, L, V]
  const int V = logits.extent(2);

  const int M = B * L;
  return nn::cross_entropy(logits.reshape({M, V}), targets.contiguous().reshape_view({M}));
}

inline nn::Tensor generate(Llama& model, const nn::Tensor& prompt_idx, int max_new_tokens,
                           float temperature, int top_k) {
  if (prompt_idx.rank() != 2 || prompt_idx.extent(1) < 1) {
    throw std::invalid_argument("causal::generate: prompt must be [B, P] with P >= 1");
  }
  if (temperature <= 0.0f) {
    throw std::invalid_argument("causal::generate: temperature must be positive");
  }
  const int B = prompt_idx.extent(0);
  const int V = model.config().vocab_size;
  const int max_len = model.config().max_position_embeddings;
  const int k = std::min(std::max(top_k, 1), V);

  nn::autograd::NoGradScope no_grad;
  nn::Tensor ids = prompt_idx;

  for (int t = 0; t < max_new_tokens; ++t) {
    const int L = ids.extent(1);
    const nn::Tensor window = (L <= max_len) ? ids : ids.slice(1, L - max_len, max_len);

    nn::Tensor logits = model.forward(window); // [B, W, V]
    const int W = window.extent(1);
    nn::Tensor last = logits.slice(1, W - 1, 1).contiguous().reshape_view({B, V});

    if (temperature != 1.0f) last = last * (1.0f / temperature);

    nn::Tensor probs = last.softmax();
    nn::Tensor values, indices;
    nn::ops::topk_rows(probs, k, values, indices);
    const nn::Tensor local = nn::ops::multinomial(values);
    const nn::Tensor picked = nn::ops::gather_rows(indices, local); // [B]

    ids = nn::cat({ids, picked.reshape_view({B, 1})}, 1);
  }

  return ids;
}

}  // namespace llm::causal
