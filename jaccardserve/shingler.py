"""
Token shingling for LLM prompts.

A k-shingle of a token sequence is a contiguous k-gram. We produce
64-bit stable hashes of each shingle so that downstream MinHash
operates on integers rather than tuples.
"""

from __future__ import annotations

import struct
from typing import Iterator, Sequence

from .minhash import stable_hash64


def shingle_hashes(token_ids: Sequence[int], k: int = 5) -> Iterator[int]:
    """
    Yield 64-bit hashes of all k-shingles of the token sequence.

    For a sequence of length n, this yields max(0, n - k + 1) hashes.
    Shingles are encoded as little-endian packed uint32 tuples before
    hashing to keep the encoding canonical across platforms.

    Args:
        token_ids: tokenized prompt, e.g. the output of a HuggingFace
                   tokenizer's input_ids field.
        k: shingle width. Typical values: 3 (high recall), 5 (balanced),
           8 (high precision).
    """
    n = len(token_ids)
    if n < k:
        return
    for i in range(n - k + 1):
        packed = struct.pack(f"<{k}I", *token_ids[i : i + k])
        yield stable_hash64(packed)


def shingle_set(token_ids: Sequence[int], k: int = 5) -> set[int]:
    """Return the deduplicated set of shingle hashes."""
    return set(shingle_hashes(token_ids, k=k))


def shingle_spans(token_ids: Sequence[int], k: int = 5) -> list[tuple[int, int]]:
    """
    Return the (start_token, end_token_exclusive) span of each shingle.

    Useful when mapping a matched shingle back to the originating token
    range, for KV span injection.
    """
    n = len(token_ids)
    return [(i, i + k) for i in range(max(0, n - k + 1))]
