#pragma once

#include <memory>
#include <string>

#include <nn/data/dataloader.h>
#include <nn/data/token_dataset.h>
#include <nn/nn.h>

namespace llm::data {

// Produced by scripts/tokenize_tiny_stories.py.
inline constexpr const char* kTinyStoriesTrainTokens = "data/TinyStories/train.bin";
inline constexpr const char* kTinyStoriesValidTokens = "data/TinyStories/valid.bin";

// A TokenDataset backed by a memory-mapped, pre-tokenized .bin file (see
// scripts/tokenize_tiny_stories.py).
class MappedTokenDataset : private nn::data::MappedTokens,
                            public nn::data::TokenDataset {
 public:
  MappedTokenDataset(const std::string& bin_path, int window, int stride = 1)
      : nn::data::MappedTokens(bin_path),
        nn::data::TokenDataset(nn::data::MappedTokens::tokens(), window, stride) {}
};

inline std::shared_ptr<MappedTokenDataset> create_tiny_stories_dataset(
    const std::string& bin_path,
    int context_length,
    int stride = 1) {
  return std::make_shared<MappedTokenDataset>(bin_path, context_length, stride);
}

inline nn::data::DataLoader<> create_tiny_stories_loader(
    const std::string& bin_path,
    int batch_size,
    int context_length,
    nn::Pcg32& rng,
    bool shuffle = true,
    bool drop_last = false,
    int stride = 1,
    nn::Device device = nn::Device::CPU) {
  auto dataset = create_tiny_stories_dataset(bin_path, context_length, stride);
  return nn::data::DataLoader<>(dataset, batch_size, rng, shuffle, drop_last, device);
}

}  // namespace llm::data
