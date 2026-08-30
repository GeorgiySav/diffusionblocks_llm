#pragma once

#include <cmath>
#include <stdexcept>

#include <nn/autograd/functions.h>
#include <nn/core/tensor.h>

class RoPE {
public:
  RoPE(int head_dim, int max_seq_len, float base = 10000.0f)
    : head_dim_(head_dim) {
    if (head_dim <= 0 || head_dim % 2 != 0) {
      throw std::invalid_argument("RoPE: head_dim must be even and positive");
    }

    cos_ = nn::Tensor::zeros({max_seq_len, head_dim});
    sin_ = nn::Tensor::zeros({max_seq_len, head_dim});

    const int half = head_dim / 2;
    float* cos_data = cos_.host_data();
    float* sin_data = sin_.host_data();
    for (int pos = 0; pos < max_seq_len; ++pos) {
      for (int i = 0; i < half; ++i) {
        const float theta = std::pow(base, -2.0f * float(i) / float(head_dim));
        const float angle = float(pos) * theta;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        // Both halves share the same angle per pair, matching the rotate-half layout below.
        cos_data[pos * head_dim + i]        = c;
        cos_data[pos * head_dim + half + i] = c;
        sin_data[pos * head_dim + i]        = s;
        sin_data[pos * head_dim + half + i] = s;
      }
    }
  }

  // x: [..., T, head_dim] (Q or K, already split per-head, e.g. [B, H, T,
  // head_dim]). Returns the rotated tensor at the same shape.
  nn::Tensor apply(const nn::Tensor& x) const {
    const int rank = x.rank();
    const int T = x.extent(rank - 2);
    if (x.extent(rank - 1) != head_dim_) {
      throw std::invalid_argument("RoPE::apply: last axis must equal head_dim");
    }
    if (T > cos_.extent(0)) {
      throw std::invalid_argument("RoPE::apply: sequence length exceeds max_seq_len");
    }

    // Constant tables: no grad needed, so a plain (non-differentiable) view
    // sliced to this call's length and moved to x's device.
    const nn::Tensor cos_t = cos_.slice_view(0, 0, T).to(x.device());  // [T, D]
    const nn::Tensor sin_t = sin_.slice_view(0, 0, T).to(x.device());  // [T, D]

    const int half = head_dim_ / 2;
    const nn::Tensor x1 = x.slice(rank - 1, 0, half);         // [..., D/2]
    const nn::Tensor x2 = x.slice(rank - 1, half, half);      // [..., D/2]
    const nn::Tensor rotated = nn::cat({-x2, x1}, rank - 1);  // [..., D]

    return x * cos_t + rotated * sin_t;
  }

private:
  int head_dim_;
  nn::Tensor cos_, sin_;
};
