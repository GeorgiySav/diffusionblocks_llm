#pragma once

#include <cmath>
#include <stdexcept>

#include <nn/autograd/functions.h>
#include <nn/core/tensor.h>

class RoPE {
public:
  RoPE(int head_dim, int max_seq_len, float base = 10000.0f)
    : head_dim_(head_dim), max_seq_len_(max_seq_len), base_(base) {
    if (head_dim <= 0 || head_dim % 2 != 0) {
      throw std::invalid_argument("RoPE: head_dim must be even and positive");
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
    if (T > max_seq_len_) {
      throw std::invalid_argument("RoPE::apply: sequence length exceeds max_seq_len");
    }

    if (!cos_dev_.defined() || cos_dev_.device() != x.device()) {
      build_tables(x.device());
    }
    const nn::Tensor cos_t = cos_dev_.slice_view(0, 0, T);  // [T, D]
    const nn::Tensor sin_t = sin_dev_.slice_view(0, 0, T);  // [T, D]

    const int half = head_dim_ / 2;
    const nn::Tensor x1 = x.slice(rank - 1, 0, half);         // [..., D/2]
    const nn::Tensor x2 = x.slice(rank - 1, half, half);      // [..., D/2]
    const nn::Tensor rotated = nn::cat({-x2, x1}, rank - 1);  // [..., D]

    return x * cos_t + rotated * sin_t;
  }

private:
  void build_tables(nn::Device device) const {
    nn::autograd::NoGradScope no_grad; // constant tables, never backpropagated through

    const int half = head_dim_ / 2;
    // theta_i = base^(-2i/D) = exp(-2i/D * ln(base)), i in [0, half).
    const float scale = -2.0f * std::log(base_) / float(head_dim_);
    const nn::Tensor theta = (nn::Tensor::arange(half, device, nn::DType::F32) * scale).exp(); // [half]

    // angle[pos, i] = pos * theta_i, an outer product via broadcasting.
    const nn::Tensor pos = nn::Tensor::arange(max_seq_len_, device, nn::DType::F32); // [max_seq_len]
    const nn::Tensor angle = pos.reshape_view({max_seq_len_, 1}) *
                              theta.reshape_view({1, half}); // [max_seq_len, half]

    // Both halves of head_dim share the same angle per pair, matching the
    // rotate-half layout in apply().
    const nn::Tensor c = angle.cos(), s = angle.sin();
    cos_dev_ = nn::cat({c, c}, 1); // [max_seq_len, head_dim]
    sin_dev_ = nn::cat({s, s}, 1); // [max_seq_len, head_dim]
  }

  int head_dim_;
  int max_seq_len_;
  float base_;
  mutable nn::Tensor cos_dev_, sin_dev_;
};
