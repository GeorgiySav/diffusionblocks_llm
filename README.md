# LLM

Two autoregressive language models trained on TinyStories, sharing one
training/generation harness:

- **GPT**: a plain decoder-only transformer (`src/llm/gpt.h`).
- **DiffusionRecurrentLLM**: a recurrent-depth model trained via the
  DiffusionBlocks recipe (`src/llm/diffusion_llm.h`) -- depth recurrence is
  mapped onto a reverse diffusion ODE and trained in a single forward pass
  per step at one sampled noise level, following Karras et al.'s EDM
  preconditioning and DiT's AdaLN-Zero conditioning.

Built on top of [nn library](../nn%20library), consumed as a sibling checkout
rather than a submodule: `CMakeLists.txt` points at `../nn library` via
`NN_LIBRARY_DIR` and pulls it in with `add_subdirectory`. Because it's a plain,
separate git repo and not nested inside this one, changes to the library are
just `cd "../nn library" && git commit` as usual -- nothing here needs to be
updated or re-pinned.

If the checkout lives somewhere else (a different machine, CI), point at it
explicitly:

```bash
cmake -S . -B build -DNN_LIBRARY_DIR=/path/to/nn-library
```

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces `train`, `generate`, and `llm_tests` (see `CMakeLists.txt` --
every top-level `src/*.cpp` file becomes its own executable, named after the
file). `-DNN_WITH_CUDA=OFF` forces a CPU-only build if no CUDA toolchain is
available; **always build in a Release-family config** (`Release`,
`RelWithDebInfo`) for real training runs -- a Debug build synchronizes the
GPU after every single kernel launch for error-checking, which is
dramatically slower.

## Data

See [DATALOADER.md](DATALOADER.md) for tokenizing TinyStories into the
`.bin` files training reads.

## Train / generate

Both binaries take the model kind (`gpt` or `diffusion`) as their first
argument; hyperparameters live in `src/train_config.h`.

```bash
./build/train diffusion
./build/generate diffusion "Once upon a time"
```

GPT and DiffusionRecurrentLLM checkpoints aren't interchangeable -- point
`TrainConfig::checkpoint_path` at a different file (or clear it) when
switching `model`, and keep `generate`'s `train_config.h` in sync with
whatever it was trained with (there's no checkpoint metadata recording
architecture hyperparameters like `n_embed` or `sigma_data`).

## Tests

```bash
cmake --build build --target llm_tests
./build/llm_tests
```
