"""
SemShareKV matching-layer simulator.

Important honesty caveat: the real SemShareKV (Zhao & Mastorakis,
IJCNLP-AACL 2025, arXiv:2509.24832) is a full system that:
  1. Computes per-token embeddings on GPU via the actual model
  2. Applies SimHash-style random-projection LSH to those embeddings
  3. Performs pair-wise reference-target matching
  4. Injects the matched KV cache into transformer layers with RoPE
     alignment

This simulator reproduces step 2 (the matching mechanism) on a CPU,
using random-projection LSH applied to synthetic per-token embedding
proxies derived from token IDs. It DOES NOT reproduce steps 1, 3, or
4. What it lets us compare on equal footing is:

  - The matching characteristics: hit rate, candidates examined,
    matching overhead.
  - The kinds of prompts each method surfaces as near-duplicate
    candidates.

It does NOT let us compare end-to-end TTFT, GPU memory savings, or
output quality. For those, the real SemShareKV codebase must be run
on actual GPU hardware. The vllm_integration/ directory contains the
harness for that comparison.

The proxy-embedding construction is deterministic and stable: token
ID -> seeded random 768-d Gaussian, then averaged within a small
context window. This produces embeddings that are similar for similar
token sequences (which is what real embeddings do) without requiring
us to actually run a model.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic token embedding (NOT a real model embedding)
# ---------------------------------------------------------------------------

class SyntheticTokenEmbedding:
    """
    Deterministic per-token-ID embedding derived from a fixed random
    seed. This is a proxy: real token embeddings carry semantic
    structure that random projections do not. SemShareKV exploits
    that semantic structure; this simulator does not. The simulator
    is therefore a LOWER BOUND on SemShareKV's matching power on
    paraphrase workloads, and an upper bound on its matching power
    on workloads where lexical features dominate.

    The fairer comparison is at the matching infrastructure level:
    same input, same hit/miss decisions, same overhead.
    """

    def __init__(self, dim: int = 768, vocab_size: int = 128_000, seed: int = 1729):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.table = rng.standard_normal((vocab_size, dim)).astype(np.float32) / np.sqrt(dim)

    def embed_tokens(self, token_ids: Sequence[int]) -> np.ndarray:
        """Return (n_tokens, dim) embedding matrix."""
        idx = np.asarray(token_ids, dtype=np.int64) % self.table.shape[0]
        return self.table[idx]

    def embed_prompt_mean(self, token_ids: Sequence[int]) -> np.ndarray:
        """Mean-pool token embeddings into a single prompt vector."""
        if len(token_ids) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return self.embed_tokens(token_ids).mean(axis=0)


# ---------------------------------------------------------------------------
# Random-projection LSH (cosine / SimHash style)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimHashConfig:
    num_planes: int = 128
    num_tables: int = 20
    seed: int = 31337


@dataclass
class SemShareEntry:
    request_id: str
    prompt_vector: np.ndarray  # (dim,)
    inserted_at: float = field(default_factory=time.time)


class SemShareKVMatcherSimulator:
    """
    SimHash-style LSH index for prompt-level cosine matching.

    SemShareKV operates at per-token granularity. For a matching-layer
    comparison, prompt-level matching is the analogue of "do we
    declare these two prompts a candidate pair for KV reuse" - which
    is the decision JaccardServe also makes at the prompt level.
    Per-token alignment happens after candidate selection in both
    systems.
    """

    def __init__(self, dim: int = 768, config: SimHashConfig = SimHashConfig()):
        self.dim = dim
        self.cfg = config
        rng = np.random.default_rng(config.seed)
        # num_tables independent random-projection matrices.
        self._planes = [
            rng.standard_normal((config.num_planes, dim)).astype(np.float32)
            for _ in range(config.num_tables)
        ]
        self._tables: list[dict[bytes, list[str]]] = [
            defaultdict(list) for _ in range(config.num_tables)
        ]
        self._entries: dict[str, SemShareEntry] = {}
        self._lock = threading.Lock()

    def _bucket_for(self, vec: np.ndarray, table_idx: int) -> bytes:
        # SimHash: project onto each plane, take sign bits, hash the
        # bit vector.
        bits = (self._planes[table_idx] @ vec) > 0
        return hashlib.blake2b(bits.tobytes(), digest_size=16).digest()

    def insert(self, request_id: str, prompt_vector: np.ndarray) -> None:
        with self._lock:
            self._entries[request_id] = SemShareEntry(request_id, prompt_vector)
            for i in range(self.cfg.num_tables):
                self._tables[i][self._bucket_for(prompt_vector, i)].append(request_id)

    def query(self, prompt_vector: np.ndarray) -> set[str]:
        candidates: set[str] = set()
        with self._lock:
            for i in range(self.cfg.num_tables):
                bucket = self._tables[i].get(self._bucket_for(prompt_vector, i))
                if bucket:
                    candidates.update(bucket)
        return candidates

    def verify_cosine(
        self,
        target_vector: np.ndarray,
        candidate_ids: set[str],
        threshold: float,
    ) -> tuple[str | None, float]:
        """Exact cosine verification on candidates; analogous to JaccardServe's exact-Jaccard step."""
        if not candidate_ids:
            return None, 0.0
        best_id = None
        best_sim = threshold
        target_norm = np.linalg.norm(target_vector) + 1e-9
        with self._lock:
            for cid in candidate_ids:
                entry = self._entries.get(cid)
                if entry is None:
                    continue
                cand = entry.prompt_vector
                cand_norm = np.linalg.norm(cand) + 1e-9
                cos = float(np.dot(target_vector, cand) / (target_norm * cand_norm))
                if cos > best_sim:
                    best_sim = cos
                    best_id = cid
        return best_id, best_sim


@dataclass
class SemShareResult:
    hit_rate: float
    avg_overhead_ms: float
    avg_candidates: float
    requests_total: int


def evaluate_semsharekv_simulated(
    prompts: list[str],
    tokenizer: Callable[[str], Sequence[int]],
    embedder: SyntheticTokenEmbedding,
    threshold: float = 0.85,
    config: SimHashConfig = SimHashConfig(),
) -> SemShareResult:
    """
    Run the SemShareKV matching-layer simulator on a stream of
    prompts and report matching characteristics. To be compared
    against JaccardServe on the same prompt stream.
    """
    matcher = SemShareKVMatcherSimulator(dim=embedder.dim, config=config)
    requests_with_hit = 0
    candidates_total = 0
    overhead_total = 0.0

    for i, p in enumerate(prompts):
        t0 = time.perf_counter()
        ids = list(tokenizer(p))
        vec = embedder.embed_prompt_mean(ids)
        cands = matcher.query(vec)
        candidates_total += len(cands)
        best_id, best_sim = matcher.verify_cosine(vec, cands, threshold)
        matcher.insert(f"req_{i}", vec)
        overhead_total += (time.perf_counter() - t0) * 1000.0
        if best_id is not None:
            requests_with_hit += 1

    n = len(prompts)
    return SemShareResult(
        hit_rate=requests_with_hit / n if n else 0.0,
        avg_overhead_ms=overhead_total / n if n else 0.0,
        avg_candidates=candidates_total / n if n else 0.0,
        requests_total=n,
    )
