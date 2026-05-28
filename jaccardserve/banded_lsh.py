"""
Banded LSH index for online cross-request near-duplicate detection.

This is the online, mutable version of the LSH banding scheme from the
author's prior Java implementation (Panchal, 2018). The differences:

  - Inserts and evictions happen at request lifecycle boundaries, not
    once at startup over a fixed corpus.
  - Entries carry KV-locality metadata (request_id, worker_id, token
    span) so the gateway can route candidates to the worker holding
    the matching KV blocks.
  - The banding curve P(s; b, r) = 1 - (1 - s^r)^b governs the
    candidate threshold and is exposed as a tunable knob.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class LSHEntry:
    """One indexed prompt. Held in the candidate lists of every band it occupies."""
    request_id: str
    worker_id: str | None
    signature: np.ndarray
    shingle_set: set[int]  # kept for exact-Jaccard verification at query time
    num_tokens: int
    token_ids: tuple[int, ...] = ()  # kept for span-resolution LCS
    inserted_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class BandingConfig:
    """
    The (b, r) tuple controls the S-curve:
        P(s) = 1 - (1 - s^r)^b
    Must satisfy num_perms == num_bands * rows_per_band.

    Two recommended defaults:
        High-precision:  b=20, r=8   (m=160), threshold ~0.7
        High-recall:     b=50, r=4   (m=200), threshold ~0.4

    Match the recommendation in Section 4.3 of the paper.
    """
    num_bands: int = 20
    rows_per_band: int = 8

    @property
    def num_perms(self) -> int:
        return self.num_bands * self.rows_per_band

    def collision_prob(self, s: float) -> float:
        """Theoretical candidate-collision probability at true similarity s."""
        return 1.0 - (1.0 - s ** self.rows_per_band) ** self.num_bands


class BandedLSH:
    """
    Thread-safe banded LSH index.

    Storage: one dict per band, keyed by the band-hash, valued by the
    list of request_ids whose signature has that band-hash in that band.

    Lookup is O(num_bands) hash lookups.
    """

    def __init__(self, config: BandingConfig = BandingConfig()):
        self.config = config
        self._bands: list[dict[bytes, list[str]]] = [
            defaultdict(list) for _ in range(config.num_bands)
        ]
        self._entries: dict[str, LSHEntry] = {}
        self._lock = threading.Lock()

    def _band_hashes(self, signature: np.ndarray) -> list[bytes]:
        if signature.size != self.config.num_perms:
            raise ValueError(
                f"Signature length {signature.size} != "
                f"expected {self.config.num_perms}"
            )
        out = []
        r = self.config.rows_per_band
        for band_idx in range(self.config.num_bands):
            band = signature[band_idx * r : (band_idx + 1) * r]
            # blake2b on the band bytes. Cryptographic strength not needed;
            # what matters is uniform distribution and low collision.
            out.append(hashlib.blake2b(band.tobytes(), digest_size=16).digest())
        return out

    def insert(self, entry: LSHEntry) -> None:
        band_hashes = self._band_hashes(entry.signature)
        with self._lock:
            self._entries[entry.request_id] = entry
            for band_idx, bh in enumerate(band_hashes):
                self._bands[band_idx][bh].append(entry.request_id)

    def query_candidates(self, signature: np.ndarray) -> set[str]:
        """
        Return the set of request_ids that share at least one full band
        with the query signature. This is the LSH candidate set; the
        caller is expected to verify exact Jaccard before acting on
        any candidate.
        """
        band_hashes = self._band_hashes(signature)
        candidates: set[str] = set()
        with self._lock:
            for band_idx, bh in enumerate(band_hashes):
                bucket = self._bands[band_idx].get(bh)
                if bucket:
                    candidates.update(bucket)
        return candidates

    def evict(self, request_id: str) -> None:
        """Remove an entry. Called when the worker drops its KV cache."""
        with self._lock:
            entry = self._entries.pop(request_id, None)
            if entry is None:
                return
            band_hashes = self._band_hashes(entry.signature)
            for band_idx, bh in enumerate(band_hashes):
                bucket = self._bands[band_idx].get(bh)
                if bucket and request_id in bucket:
                    bucket.remove(request_id)
                    if not bucket:
                        del self._bands[band_idx][bh]

    def get_entry(self, request_id: str) -> LSHEntry | None:
        with self._lock:
            return self._entries.get(request_id)

    def size(self) -> int:
        with self._lock:
            return len(self._entries)
