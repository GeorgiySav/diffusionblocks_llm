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

  // x: [..., T, head_dim] (Q or K, already split per-head), positions 0..T-1.
  // Returns the rotated tensor at the same shape.
  nn::Tensor apply(const nn::Tensor& x) const {
    const int T = check(x);
    build_tables_if_needed(x.device());
    return rotate(x, cos_dev_.slice_view(0, 0, T), sin_dev_.slice_view(0, 0, T));
  }

private:
  int check(const nn::Tensor& x) const {
    const int rank = x.rank();
    const int T = x.extent(rank - 2);
    if (x.extent(rank - 1) != head_dim_) {
      throw std::invalid_argument("RoPE::apply: last axis must equal head_dim");
    }
    if (T > max_seq_len_) {
      throw std::invalid_argument("RoPE::apply: sequence length exceeds max_seq_len");
    }
    return T;
  }

  // cos_t / sin_t are [T, head_dim] and broadcast across the leading axes.
  nn::Tensor rotate(const nn::Tensor& x, const nn::Tensor& cos_t, const nn::Tensor& sin_t) const {
    const int rank = x.rank();
    const int half = head_dim_ / 2;
    const nn::Tensor x1 = x.slice(rank - 1, 0, half);         // [..., D/2]
    const nn::Tensor x2 = x.slice(rank - 1, half, half);      // [..., D/2]
    const nn::Tensor rotated = nn::cat({-x2, x1}, rank - 1);  // [..., D]

    return x * cos_t + rotated * sin_t;
  }

  void build_tables_if_needed(nn::Device device) const {
    if (cos_dev_.defined() && cos_dev_.device() == device) return;

    nn::autograd::NoGradScope no_grad; // constant tables

    const int half = head_dim_ / 2;
    // theta_i = base^(-2i/D) = exp(-2i/D * ln(base)), i in [0, half).
    const float scale = -2.0f * std::log(base_) / float(head_dim_);
    const nn::Tensor theta = (nn::Tensor::arange(half, device, nn::DType::F32) * scale).exp(); // [half]

    // angle[pos, i] = pos * theta_i
    const nn::Tensor pos = nn::Tensor::arange(max_seq_len_, device, nn::DType::F32); // [max_seq_len]
    const nn::Tensor angle = pos.reshape_view({max_seq_len_, 1}) *
                             theta.reshape_view({1, half}); // [max_seq_len, half]

    const nn::Tensor c = angle.cos(), s = angle.sin();
    cos_dev_ = nn::cat({c, c}, 1); // [max_seq_len, head_dim]
    sin_dev_ = nn::cat({s, s}, 1); // [max_seq_len, head_dim]
  }

  int head_dim_;
  int max_seq_len_;
  float base_;
  mutable nn::Tensor cos_dev_, sin_dev_;
};
