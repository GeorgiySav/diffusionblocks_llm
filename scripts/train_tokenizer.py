#!/usr/bin/env python3
"""Trains a custom byte-level BPE tokenizer on TinyStories text.

TinyStories' vocabulary is small and repetitive (written for 3-4 year olds),
so a full ~50k-token pretrained encoding like GPT-2's is mostly wasted -- most
of that vocabulary never shows up. Training a small vocabulary on the corpus
itself shrinks the embedding table and LM head (both scale with
vocab_size * n_embed) and the cross-entropy computation, without changing
anything about the model architecture.

Requires: pip install -r scripts/requirements.txt

Usage:
    py scripts/train_tokenizer.py
        Trains an 8192-token tokenizer on data/TinyStories/TinyStories-train.txt
        and writes data/TinyStories/tokenizer-8192.json.

    py scripts/train_tokenizer.py --vocab-size 4096 --input path/to.txt --output path/to/tokenizer.json

After training, point scripts/tokenize_tiny_stories.py --encoding at the
resulting tokenizer.json (instead of a tiktoken encoding name) to produce
.bin files with it, set TrainConfig::vocab_size in src/train_config.h to the
printed actual vocab size, and pass the same tokenizer.json path as
`generate`'s encoding argument at generation time.
"""
import argparse

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DEFAULT_INPUT = "data/TinyStories/TinyStories-train.txt"


def train_tokenizer(input_path: str, output_path: str, vocab_size: int) -> None:
    tokenizer = Tokenizer(models.BPE())
    # Byte-level, GPT-2-style pre-tokenization: every byte sequence is
    # representable, so there's no UNK token and no risk of a character the
    # trainer never saw breaking encoding later.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # <|endoftext|> is registered as a special token *before* training so it
    # gets one dedicated id instead of being split into ordinary BPE pieces
    # -- matches how tokenize_tiny_stories.py treats it via tiktoken's
    # `allowed_special`.
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
    )
    tokenizer.train([input_path], trainer)
    tokenizer.save(output_path)

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Trained a {actual_vocab_size}-token tokenizer on {input_path}")
    print(f"  wrote {output_path}")
    print(f"  set TrainConfig::vocab_size = {actual_vocab_size} in src/train_config.h to match")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the training text file")
    parser.add_argument("--output",
                         help="Path to write the tokenizer.json "
                              "(default: data/TinyStories/tokenizer-<vocab-size>.json)")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Target vocabulary size")
    args = parser.parse_args()

    output = args.output or f"data/TinyStories/tokenizer-{args.vocab_size}.json"
    train_tokenizer(args.input, output, args.vocab_size)


if __name__ == "__main__":
    main()
