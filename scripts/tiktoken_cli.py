#!/usr/bin/env python3
"""Thin tiktoken CLI used by the C++ tokenizer (src/tokenizer.h).

Modes:
    encode --input <text-file> --encoding <name>
        Reads UTF-8 text from <text-file>, writes one token id per line.

    decode --input <ids-file> --encoding <name>
        Reads one token id per line from <ids-file>, writes the decoded
        UTF-8 text.
"""
import argparse
import sys

import tiktoken


def encode(text_path: str, encoding_name: str) -> None:
    enc = tiktoken.get_encoding(encoding_name)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Matches scripts/tokenize_tiny_stories.py: let a literal "<|endoftext|>"
    # in the prompt come through as its special token rather than erroring.
    ids = enc.encode(text, allowed_special={"<|endoftext|>"})

    out = "\n".join(str(i) for i in ids)
    sys.stdout.buffer.write(out.encode("ascii"))


def decode(ids_path: str, encoding_name: str) -> None:
    enc = tiktoken.get_encoding(encoding_name)
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [int(line) for line in f if line.strip()]

    text = enc.decode(ids)
    sys.stdout.buffer.write(text.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--encoding", default="gpt2", help="tiktoken encoding name")
    args = parser.parse_args()

    if args.mode == "encode":
        encode(args.input, args.encoding)
    else:
        decode(args.input, args.encoding)


if __name__ == "__main__":
    main()
