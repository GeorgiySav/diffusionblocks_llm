"""Shared encode/decode dispatch for tokenize_tiny_stories.py and
tokenizer_cli.py."""
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
        # The dataset uses "<|endoftext|>" as a story separator
        return self._enc.encode(text, allowed_special={_ENDOFTEXT})

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)


class CustomBpeBackend(TokenizerBackend):
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
