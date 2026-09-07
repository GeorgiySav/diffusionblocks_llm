# LLM

A Llama-style decoder-only language model trained on TinyStories, in C++:
RMSNorm pre-norm blocks, RoPE, SwiGLU MLPs, grouped-query attention, no biases.

Built on [nn library](../nn%20library), a sibling checkout. Override its path
with `-DNN_LIBRARY_DIR=/path/to/nn-library`.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Clang is required -- the nn library does not compile under MSVC.

## 0. Get the corpus

Download [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
to `data/TinyStories/TinyStories-{train,valid}.txt`, then:

```bash
pip install -r scripts/requirements.txt
```

## 1. Train the tokenizer

```bash
py scripts/train_tokenizer.py --vocab-size 4096
```

Set `LlamaTrainConfig::vocab_size` in `src/llama_config.h` to the size it
prints -- it can land below what you asked for.

## 2. Tokenize

```bash
py scripts/tokenize_tiny_stories.py --encoding data/TinyStories/tokenizer-4096.json
```

Writes `data/TinyStories/{train,valid}.bin`.

## 3. Train

```bash
./build/train_llama
```

Runs to `max_steps`, which also sizes the LR schedule. The default 18700 steps
is ~9 hours on one GPU and gives a 62M model.

Checkpoints go to `checkpoints/llama.ckpt` every 400 steps, and to
`.ckpt.best` whenever validation improves. Restarting resumes from the rolling
checkpoint; delete it to start over.

## 4. Generate

```bash
./build/generate_llama "Once upon a time" checkpoints/llama.ckpt.best 0.8 40
#                       prompt            checkpoint                  temp top_k
```

Prefer `.best` -- validation drifts up near the end, so the final checkpoint
is not the best one. `src/llama_config.h` must still match the architecture the
weights were trained with; nothing about it is stored in a checkpoint.

## Tests

```bash
./build/llm_tests
```
