#include <cstdio>

#include <nn/nn.h>

// Sanity check that the nn-library dependency links and runs; replace with
// the real model.
int main() {
  nn::Pcg32 rng(0);
  const nn::Tensor x = nn::Tensor::randn({2, 3}, rng, 1.0f);
  std::printf("nn-library linked OK. device: %s\n",
              nn::device_name(x.device()));
  std::printf("%s\n", x.str().c_str());
  return 0;
}
