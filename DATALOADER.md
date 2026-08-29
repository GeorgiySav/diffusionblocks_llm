# TinyStories DataLoader

A dataloader for the TinyStories dataset compatible with the nn library.

Tokenization happens offline in Python (via the `tiktoken` package, which has
no C++ counterpart); training reads the resulting binary token file through
the nn library's own `nn::data::MappedTokens` + `nn::data::TokenDataset` +
`nn::data::DataLoader`.

## Workflow

1. **Tokenize once** (Python):

   ```bash
   pip install -r scripts/requirements.txt
   py scripts/tokenize_tiny_stories.py
   ```

   This reads `data/TinyStories/TinyStories-{train,valid}.txt` and writes
   `data/TinyStories/{train,valid}.bin` -- flat little-endian `int32` token
   id arrays, encoded with tiktoken's `cl100k_base` encoding (the `<|endoftext|>`
   story separator already in the `.txt` files is preserved as its special
   token, id 100257).

   To tokenize a single file, or with a different encoding:

   ```bash
   py scripts/tokenize_tiny_stories.py --input path/to/in.txt --output path/to/out.bin --encoding p50k_base
   ```

2. **Load in C++** (`src/dataloader.h`):

   ```cpp
   #include "dataloader.h"

   nn::Pcg32 rng(42);
   auto loader = llm::data::create_tiny_stories_loader(
       llm::data::kTinyStoriesTrainTokens,  // "data/TinyStories/train.bin"
       /*batch_size=*/32,
       /*context_length=*/256,
       rng);

   while (loader.has_next()) {
     auto [input, target] = loader.next();  // I32 tensors, [batch_size, context_length]
     // target[b, t] == input[b, t + 1] (next-token prediction)
   }
   ```

## API Reference

### `MappedTokenDataset`

`nn::data::TokenDataset` backed by a memory-mapped `.bin` file, so the corpus
doesn't need to fit in RAM and there's no separate buffer to keep alive.

```cpp
MappedTokenDataset(const std::string& bin_path, int window, int stride = 1);
```

### `create_tiny_stories_dataset(bin_path, context_length, stride = 1)`

Returns a `shared_ptr<MappedTokenDataset>`, usable anywhere an
`nn::data::Dataset<2>` is expected.

### `create_tiny_stories_loader(bin_path, batch_size, context_length, rng, shuffle = true, drop_last = false, stride = 1, device = nn::Device::CPU)`

Returns a ready-to-use `nn::data::DataLoader<>`.

**Parameters:**

- `bin_path`: path to a `.bin` file produced by the tokenize script
- `batch_size`: sequences per batch
- `context_length`: window size (tokens per sequence)
- `rng`: shared `nn::Pcg32` used for shuffling
- `shuffle` / `drop_last` / `device`: forwarded to `nn::data::DataLoader`
- `stride`: step between consecutive windows (1 = maximum overlap; use
  `context_length` for non-overlapping windows)

## Notes

- `input`/`target` are `I32` tensors of shape `[batch_size, context_length]`,
  with `target` the same token window shifted by one -- the standard
  next-token prediction setup, matching `nn::data::TokenDataset`.
- Re-run the tokenize script whenever the `.txt` source files change; the
  `.bin` files are a build artifact, not something to hand-edit.

## Data Download

Download TinyStories from <https://huggingface.co/datasets/roneneldan/TinyStories>
and place the files at:

- `data/TinyStories/TinyStories-train.txt`
- `data/TinyStories/TinyStories-valid.txt`
