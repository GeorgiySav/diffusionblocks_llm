#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <nn/data/dataloader.h>
#include <nn/data/token_dataset.h>
#include <nn/nn.h>

namespace llm::data {

// Produced by scripts/tokenize_tiny_stories.py (run once before training).
inline constexpr const char* kTinyStoriesTrainTokens = "data/TinyStories/train.bin";
inline constexpr const char* kTinyStoriesValidTokens = "data/TinyStories/valid.bin";

// The first 20M tokens of kTinyStoriesTrainTokens (~4% of the full ~482M),
// for a quality-vs-speed middle ground between TinyShakespeare (too small
// and repetitive to judge story quality) and the full corpus (hours per
// epoch): still tens of thousands of distinct stories, but small enough for
// several epochs within a short, fixed time budget. Pair with the full
// kTinyStoriesValidTokens -- it's already small, no need for its own slice.
inline constexpr const char* kTinyStoriesSmallTrainTokens = "data/TinyStories/train_small.bin";

// A much smaller corpus (~1MB) for quickly validating a model/trainer change
// end-to-end -- full epochs in seconds rather than TinyStories' minutes, at
// the cost of being too small and repetitive to judge final quality from.
// Tokenized with the same tokenizer-4096.json as TinyStories (see
// TrainConfig::train_tokens_path), so it's a drop-in swap for whichever
// architecture's config points at it.
inline constexpr const char* kTinyShakespeareTrainTokens = "data/TinyShakespeare/train.bin";
inline constexpr const char* kTinyShakespeareValidTokens = "data/TinyShakespeare/valid.bin";

inline constexpr int kCl100kVocabSize = 100277;
inline constexpr int kGPT2VocabSize = 50257;

// Convenience default for `generate`'s encoding argument when training used
// one of tiktoken's two built-in encodings this project knows the vocab
// size of. A custom-trained tokenizer (see scripts/train_tokenizer.py) has
// no fixed vocab size to guess from -- pass its tokenizer.json path
// explicitly as `generate`'s encoding argument instead of relying on this.
inline std::string encoding_for_vocab_size(int vocab_size) {
  if (vocab_size == kCl100kVocabSize) return "cl100k_base";
  if (vocab_size == kGPT2VocabSize) return "gpt2";
  throw std::invalid_argument(
      "encoding_for_vocab_size: no known tiktoken encoding for vocab_size " +
      std::to_string(vocab_size) +
      " -- if this is a custom-trained tokenizer (scripts/train_tokenizer.py), "
      "pass its tokenizer.json path explicitly as generate's encoding argument");
}

// A TokenDataset backed by a memory-mapped, pre-tokenized .bin file (see
// scripts/tokenize_tiny_stories.py). MappedTokens is listed before
// TokenDataset in the base-specifier list so it is constructed first --
// base classes initialize in list order, not declaration order of data
// members -- letting its span be passed straight into TokenDataset's
// constructor and stay valid for the object's whole lifetime.
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
