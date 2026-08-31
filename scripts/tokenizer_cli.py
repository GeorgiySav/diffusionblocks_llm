#!/usr/bin/env python3
"""Thin tokenizer CLI used by the C++ tokenizer (src/tokenizer.h).

--encoding accepts either a tiktoken encoding name (e.g. "gpt2") or a path
to a custom tokenizer.json trained by scripts/train_tokenizer.py.

Modes:
    encode --input <text-file> --encoding <name-or-path>
        Reads UTF-8 text from <text-file>, writes one token id per line.

    decode --input <ids-file> --encoding <name-or-path>
        Reads one token id per line from <ids-file>, writes the decoded
        UTF-8 text.
"""
import argparse
import sys

from tokenizer_backend import load_backend


def encode(text_path: str, encoding: str) -> None:
    backend = load_backend(encoding)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = backend.encode(text)

    out = "\n".join(str(i) for i in ids)
    sys.stdout.buffer.write(out.encode("ascii"))


def decode(ids_path: str, encoding: str) -> None:
    backend = load_backend(encoding)
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [int(line) for line in f if line.strip()]

    text = backend.decode(ids)
    sys.stdout.buffer.write(text.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--encoding", default="gpt2",
                         help="tiktoken encoding name, or a path to a custom tokenizer.json")
    args = parser.parse_args()

    if args.mode == "encode":
        encode(args.input, args.encoding)
    else:
        decode(args.input, args.encoding)


if __name__ == "__main__":
    main()
