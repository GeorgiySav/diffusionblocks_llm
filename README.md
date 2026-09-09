# LLM

A Llama-style decoder-only language model trained on TinyStories, in C++:
RMSNorm pre-norm blocks, RoPE, SwiGLU MLPs, grouped-query attention, no biases.

Built on [nn library](../nn%20library), a sibling checkout. Override its path
with `-DNN_LIBRARY_DIR=/path/to/nn-library`.

## The trained model

| Spec | Value |
| --- | --- |
| Parameters | 62.11M |
| Architecture | 12 layers, 640 hidden, 10 heads, 1728 MLP, 256-token context |
| Vocabulary | 4096, byte-level BPE trained on the corpus |
| Best val loss | **1.3409** (perplexity 3.82) |

Sampled at `temperature 0.8`, `top_k 40`, from the prompt `There was a boy`:

```
There was a boy named Jack. He was a good boy. Every day, he went to school with his pencil in hand.
It was a regular pencil, but it made him think about life.
One day, Jack decided to design something special. He wanted to design his favourite toy. He used
blue and green and yellow and green. He worked very hard at designing his toy. 
When he was finished, Jack was very proud of his creation. He ran to show his mum. She was so
impressed with what he had made. She hugged him and said, "Your pencil is a great design! You are a
very talented boy!" 
Jack was so happy. He was so proud and thanked his mum for making him design something so special.
From then on, he was always sure to take good care of his pencil.
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

## 4. Generate

```bash
./build/generate_llama "Once upon a time" checkpoints/llama.ckpt.best 0.8 40
#                       prompt            checkpoint                  temp top_k
```

## Tests

```bash
./build/llm_tests
```
