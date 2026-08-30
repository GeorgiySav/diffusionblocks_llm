#!/usr/bin/env python3
"""Tokenizes TinyStories .txt files into flat int32 binary files.

The output format is a raw array of little-endian int32 token ids, matching
what nn::data::load_tokens / nn::data::MappedTokens expect on the C++ side.

Requires: pip install -r scripts/requirements.txt

Usage:
    py scripts/tokenize_tiny_stories.py
        Tokenizes the default train/valid files under data/TinyStories/.

    py scripts/tokenize_tiny_stories.py --input path/to/in.txt --output path/to/out.bin
        Tokenizes a single file.
"""
import argparse

import numpy as np
import tiktoken

DEFAULT_FILES = [
    ("data/TinyStories/TinyStories-train.txt", "data/TinyStories/train.bin"),
    ("data/TinyStories/TinyStories-valid.txt", "data/TinyStories/valid.bin"),
]


def tokenize_file(input_path: str, output_path: str, encoding_name: str) -> None:
    enc = tiktoken.get_encoding(encoding_name)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # The dataset already uses "<|endoftext|>" as a story separator; letting
    # tiktoken recognize it as a special token keeps story boundaries in the
    # token stream instead of splitting it into ordinary BPE pieces.
    ids = enc.encode(text, allowed_special={"<|endoftext|>"})

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
