"""
MinHash signature computation for token shingles.

Implements the classical Broder MinHash sketch with the same
ax + b mod p hash family used in the author's prior Java
implementation (Panchal, 2018), adapted to operate on token-ID
shingles produced by an LLM tokenizer rather than word shingles
produced by document preprocessing.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# A Mersenne prime larger than any reasonable token-shingle hash space.
# Same role as p in the Java implementation, where p was chosen to be
# >= the number of unique terms. For 64-bit shingle hashes this is safe.
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


@dataclass(frozen=True)
class MinHashConfig:
    num_perms: int = 128
    seed: int = 1729  # fixed for reproducibility


class MinHasher:
    """
    Computes m-dimensional MinHash signatures using a fixed family of
    universal hash functions h_i(x) = (a_i * x + b_i) mod p, with the
    minimum-over-shingles taken per coordinate.

    Concretely matches Broder (1997) and the implementation in
    Panchal (2018). See `shingler.py` for the upstream shingle producer.
    """

    def __init__(self, config: MinHashConfig = MinHashConfig()):
        self.num_perms = config.num_perms
        rng = random.Random(config.seed)
        # Hash family coefficients. Both a and b drawn uniformly from
        # {1, ..., p-1}. Stored as int64 numpy arrays for vectorized
        # signature computation.
        self._a = np.array(
            [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(self.num_perms)],
            dtype=np.uint64,
        )
        self._b = np.array(
            [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(self.num_perms)],
            dtype=np.uint64,
        )

    def signature(self, shingle_hashes: Iterable[int]) -> np.ndarray:
        """
        Compute the MinHash signature for a set of shingle hashes.

        Args:
            shingle_hashes: iterable of 64-bit integer hashes of token shingles.
                            Use shingler.shingle_hashes() to produce these.

        Returns:
            int64 numpy array of length num_perms.
        """
        shingles = np.fromiter(shingle_hashes, dtype=np.uint64)
        if shingles.size == 0:
            # Empty input: return all-max signature so collisions are impossible.
            return np.full(self.num_perms, _MAX_HASH, dtype=np.uint64)

        # Compute (a_i * x + b_i) mod p for every (i, x) pair, then min over x.
        # Shape: (num_perms, num_shingles)
        hashed = np.bitwise_and(
            (np.outer(self._a, shingles) + self._b[:, None]) % _MERSENNE_PRIME,
            _MAX_HASH,
        )
        return hashed.min(axis=1)

    @staticmethod
    def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from two MinHash signatures.

        By the defining property of MinHash:
            E[ 1[h_i(A) = h_i(B)] ] = J(A, B)

        so the mean of equality indicators across the signature is an
        unbiased estimator of J(A, B).
        """
        if sig_a.shape != sig_b.shape:
            raise ValueError("Signature lengths must match")
        return float(np.mean(sig_a == sig_b))

    @staticmethod
    def exact_jaccard(set_a: set, set_b: set) -> float:
        """Reference exact Jaccard for verification and tests."""
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)


def stable_hash64(x: bytes) -> int:
    """64-bit stable hash. Used for token-shingle fingerprinting."""
    return int.from_bytes(hashlib.blake2b(x, digest_size=8).digest(), "big")
