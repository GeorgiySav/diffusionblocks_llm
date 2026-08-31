#!/usr/bin/env python3
"""Tokenizes TinyStories .txt files into flat int32 binary files.

The output format is a raw array of little-endian int32 token ids, matching
what nn::data::load_tokens / nn::data::MappedTokens expect on the C++ side.

Requires: pip install -r scripts/requirements.txt

Usage:
    py scripts/tokenize_tiny_stories.py
        Tokenizes the default train/valid files under data/TinyStories/
        using tiktoken's "gpt2" encoding.

    py scripts/tokenize_tiny_stories.py --input path/to/in.txt --output path/to/out.bin
        Tokenizes a single file.

    py scripts/tokenize_tiny_stories.py --encoding data/TinyStories/tokenizer-8192.json
        Tokenizes with a custom tokenizer trained by train_tokenizer.py
        instead of a tiktoken encoding -- --encoding accepts either a
        tiktoken encoding name or a path to a trained tokenizer.json.
"""
import argparse

import numpy as np

from tokenizer_backend import load_backend

DEFAULT_FILES = [
    ("data/TinyStories/TinyStories-train.txt", "data/TinyStories/train.bin"),
    ("data/TinyStories/TinyStories-valid.txt", "data/TinyStories/valid.bin"),
]


def tokenize_file(input_path: str, output_path: str, encoding_name: str) -> None:
    backend = load_backend(encoding_name)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = backend.encode(text)

    tokens = np.array(ids, dtype=np.int32)
    tokens.tofile(output_path)

    print(f"{input_path}: {len(text):,} chars -> {len(tokens):,} tokens ({encoding_name})")
    print(f"  wrote {output_path} ({tokens.nbytes:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to a single TinyStories .txt file")
    parser.add_argument("--output", help="Path to write the tokenized .bin file (required with --input)")
    parser.add_argument("--encoding", default="gpt2", help="tiktoken encoding name")
    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required when --input is given")
        tokenize_file(args.input, args.output, args.encoding)
    else:
        for input_path, output_path in DEFAULT_FILES:
            tokenize_file(input_path, output_path, args.encoding)


if __name__ == "__main__":
    main()
