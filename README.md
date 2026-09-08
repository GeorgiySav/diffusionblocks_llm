# LLM

A Llama-style decoder-only language model trained on TinyStories, in C++:
RMSNorm pre-norm blocks, RoPE, SwiGLU MLPs, grouped-query attention, no biases.

Built on [nn library](../nn%20library), a sibling checkout. Override its path
with `-DNN_LIBRARY_DIR=/path/to/nn-library`.

```bash
Once upon a time, there was a little boy named Timmy. Timmy loved to play outside and explore the world around him. One day, he found a big rock in his backyard. It was very heavy and he couldn't lift it. He tried and tried, but it was too hard for him.
Suddenly, Timmy's mom came outside and saw him struggling with the rock. "What's wrong, Timmy?" she asked. Timmy showed her the rock and said, "I can't lift it, Mommy." His mom smiled and said, "Let me help you." She lifted the rock and Timmy was very happy.
But then, Timmy's little sister came outside and wanted to play with the rock too. Timmy didn't want to share the rock, so he said, "No, it's mine!" His sister started to cry and Timmy felt bad. He realized that he was being selfish and decided to share the rock with his sister. They both played together and had fun. The end
```

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
