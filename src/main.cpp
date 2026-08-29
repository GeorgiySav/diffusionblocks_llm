#include <cstdio>

#include <nn/nn.h>
#include "dataloader.h"

int main() {
  nn::Pcg32 rng(0);
  const nn::Tensor x = nn::Tensor::randn({2, 3}, rng, 1.0f);
  std::printf("nn-library linked OK. device: %s\n",
              nn::device_name(x.device()));
  std::printf("%s\n", x.str().c_str());

  try {
    // Example: Create a TinyStories dataloader.
    // Requires running `py scripts/tokenize_tiny_stories.py` once beforehand
    // to produce data/TinyStories/train.bin from the raw .txt file.
    const int batch_size = 32;
    const int context_length = 256;

    auto loader = llm::data::create_tiny_stories_loader(
        llm::data::kTinyStoriesTrainTokens,
        batch_size,
        context_length,
        rng);

    std::printf("\nTinyStories DataLoader initialized:\n");
    std::printf("  Batches per epoch: %d\n", loader.batches_per_epoch());
    std::printf("  Batch size: %d\n", batch_size);
    std::printf("  Context length: %d\n", context_length);

    if (loader.has_next()) {
      auto [input, target] = loader.next();
      std::printf("  Input shape: [%d, %d]\n", input.extent(0), input.extent(1));
      std::printf("  Target shape: [%d, %d]\n", target.extent(0), target.extent(1));
    }
  } catch (const std::exception& e) {
    std::printf("Warning: Could not load TinyStories (expected if data file missing): %s\n", e.what());
  }

  return 0;
}
