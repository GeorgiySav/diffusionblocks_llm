#pragma once

#include <stdexcept>

#include <nn/core/tensor.h>

inline nn::Tensor build_diffusion_keep_mask(int T, nn::Device device) {
  if (T < 0) {
    throw std::invalid_argument("build_diffusion_keep_mask: T must not be negative");
  }

  const int L = 2 * T;
  nn::Tensor h(nn::Shape({L, L}), nn::Device::CPU, nn::DType::F32);
  float* data = h.host_data();
  for (int row = 0; row < L; ++row) {
    for (int col = 0; col < L; ++col) {
      bool keep;
      if (row < T) {
        keep = (col <= row) && (col < T);
      } else {
        const int i = row - T;
        keep = (col == row) || (col < i);
      }
      data[size_t(row) * L + col] = keep ? 1.0f : 0.0f;
    }
  }
  return h.to(device);
}
