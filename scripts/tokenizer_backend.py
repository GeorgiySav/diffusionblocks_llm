"""Shared encode/decode dispatch for tokenize_tiny_stories.py and
tokenizer_cli.py: a tokenizer "spec" string is either a tiktoken encoding
name (e.g. "gpt2") or a path to a custom BPE tokenizer.json trained by
train_tokenizer.py -- distinguished by whether it names an existing file.
"""
import os
from typing import List

_ENDOFTEXT = "<|endoftext|>"


class TokenizerBackend:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError


class TiktokenBackend(TokenizerBackend):
    def __init__(self, encoding_name: str):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        # The dataset uses "<|endoftext|>" as a story separator; letting
        # tiktoken recognize it as a special token keeps story boundaries in
        # the token stream instead of splitting it into ordinary BPE pieces.
        return self._enc.encode(text, allowed_special={_ENDOFTEXT})

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)


class CustomBpeBackend(TokenizerBackend):
    # Tokenizer.encode()/encode_batch() materializes a full Encoding (ids,
    # character offsets, word ids, attention mask, special-tokens mask --
    # several parallel arrays per token) for every input given to one call.
    # Passing TinyStories' ~2GB training file as a single string holds all
    # of that per-token metadata for the *entire* corpus in memory at once:
    # a large multiplier over the ~4 bytes/token actually wanted, easily
    # enough to abort the process with a Rust allocation failure (tiktoken
    # never has this problem -- it just returns a plain list of ints).
    # "<|endoftext|>" is a registered special token, so it never merges
    # into a BPE piece with its neighbors: splitting the text on it and
    # encoding stories in bounded batches gives an identical token stream
    # to one giant encode() call, at a small fraction of the peak memory.
    _BATCH_CHARS = 8_000_000

    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_path)
        self._endoftext_id = self._tok.token_to_id(_ENDOFTEXT)
        if self._endoftext_id is None:
            raise ValueError(
                f"{tokenizer_path}: no \"{_ENDOFTEXT}\" special token -- "
                "was this trained by train_tokenizer.py?")

    def encode(self, text: str) -> List[int]:
        stories = text.split(_ENDOFTEXT)
        n = len(stories)
        ids: List[int] = []

        def flush(start_index: int, items: List[str]) -> None:
            for offset, enc in enumerate(self._tok.encode_batch(items)):
                ids.extend(enc.ids)
                # Every story had a "<|endoftext|>" after it except the
                # very last one in the whole corpus (str.split's semantics).
                if start_index + offset < n - 1:
                    ids.append(self._endoftext_id)

        batch: List[str] = []
        batch_start = 0
        batch_chars = 0
        for i, story in enumerate(stories):
            batch.append(story)
            batch_chars += len(story)
            if batch_chars >= self._BATCH_CHARS or i == n - 1:
                flush(batch_start, batch)
                batch, batch_start, batch_chars = [], i + 1, 0
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def load_backend(spec: str) -> TokenizerBackend:
    if os.path.isfile(spec):
        return CustomBpeBackend(spec)
    return TiktokenBackend(spec)
