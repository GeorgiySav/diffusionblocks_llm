#!/usr/bin/env python3
"""Trains a custom byte-level BPE tokenizer on TinyStories text."""
import argparse

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DEFAULT_INPUT = "data/TinyStories/TinyStories-train.txt"


def train_tokenizer(input_path: str, output_path: str, vocab_size: int) -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

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
    print(f"  set LlamaTrainConfig::vocab_size = {actual_vocab_size} "
          f"in src/llama_config.h to match")


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
