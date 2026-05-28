"""
vLLM Automatic Prefix Cache simulator.

Models vLLM's block-level exact hashing: tokens are grouped into
fixed-size blocks (default 16), each block is strongly hashed
(SHA-256 in production; we use blake2b which is faster and equivalent
for this purpose), and KV reuse is granted when a new request's
leading blocks have been seen before in the same offset position.

This simulator measures the matching-layer hit rate; it does not
model the actual vLLM serving pipeline. The matching-layer hit rate
is the upper bound on what vLLM APC can save - actual TTFT savings
require the full stack.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class APCResult:
    hit_rate: float
    avg_blocks_matched: float
    avg_overhead_ms: float
    requests_total: int


class VLLMAPCSimulator:
    """
    Block-level exact prefix cache. Stores the hash of every block at
    every offset that has been seen. A new request gets a hit on block
    i if the same (block_hash, offset) pair has been registered before.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        # (offset, block_hash) -> count
        self._cache: dict[tuple[int, bytes], int] = {}

    def _block_hash(self, token_ids: Sequence[int]) -> bytes:
        # Same encoding as a real APC: serialize then hash.
        data = b"".join(int(t).to_bytes(4, "little") for t in token_ids)
        return hashlib.blake2b(data, digest_size=16).digest()

    def query(self, token_ids: Sequence[int]) -> int:
        """
        Return the number of leading blocks that hit the cache.
        Stops at the first miss (standard prefix-cache semantics).
        """
        blocks_hit = 0
        for offset in range(0, len(token_ids), self.block_size):
            block = token_ids[offset : offset + self.block_size]
            if len(block) < self.block_size:
                break
            bh = self._block_hash(block)
            if (offset, bh) in self._cache:
                blocks_hit += 1
            else:
                break
        return blocks_hit

    def register(self, token_ids: Sequence[int]) -> None:
        """Admit a request's blocks to the cache."""
        for offset in range(0, len(token_ids), self.block_size):
            block = token_ids[offset : offset + self.block_size]
            if len(block) < self.block_size:
                break
            bh = self._block_hash(block)
            key = (offset, bh)
            self._cache[key] = self._cache.get(key, 0) + 1


def evaluate_apc(
    prompts: list[str],
    tokenizer: Callable[[str], Sequence[int]],
    block_size: int = 16,
) -> APCResult:
    """
    Run vLLM-APC-style block hashing over a stream of prompts and
    report the cache hit characteristics.
    """
    apc = VLLMAPCSimulator(block_size=block_size)
    requests_with_hit = 0
    total_blocks_matched = 0
    total_overhead = 0.0
    for p in prompts:
        t0 = time.perf_counter()
        ids = list(tokenizer(p))
        blocks_hit = apc.query(ids)
        apc.register(ids)
        total_overhead += (time.perf_counter() - t0) * 1000.0
        if blocks_hit > 0:
            requests_with_hit += 1
            total_blocks_matched += blocks_hit

    n = len(prompts)
    return APCResult(
        hit_rate=requests_with_hit / n if n else 0.0,
        avg_blocks_matched=total_blocks_matched / n if n else 0.0,
        avg_overhead_ms=total_overhead / n if n else 0.0,
        requests_total=n,
    )
