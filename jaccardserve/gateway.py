"""
JaccardServe gateway.

Orchestrates the per-request pipeline described in Section 4.1 of the
paper:

    request --> tokenize --> shingle --> MinHash --> LSH lookup
            --> exact-Jaccard verification --> injection plan

The gateway is model-agnostic. It takes a tokenizer function and an
LSH index; integration with a specific serving engine (vLLM, SGLang,
TGI) is delegated to an adapter (see vllm_adapter.py).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .banded_lsh import BandedLSH, BandingConfig, LSHEntry
from .minhash import MinHasher, MinHashConfig
from .shingler import shingle_hashes, shingle_set
from .verifier import longest_matching_span, verify_jaccard

Tokenizer = Callable[[str], Sequence[int]]


@dataclass
class InjectionPlan:
    """
    The output of the matching pipeline. Describes which prior request
    to draw KV state from, and which token range of the current request
    is covered by the reuse.

    If `donor_request_id` is None, no near-duplicate was found above
    threshold and the request should fall through to standard prefill.
    """
    request_id: str
    donor_request_id: str | None = None
    donor_worker_id: str | None = None
    matched_target_span: tuple[int, int] | None = None  # (start, end) in target
    matched_donor_span: tuple[int, int] | None = None  # (start, end) in donor
    measured_jaccard: float = 0.0
    candidates_examined: int = 0
    gateway_overhead_ms: float = 0.0


@dataclass
class GatewayStats:
    requests_total: int = 0
    requests_with_match: int = 0
    candidates_examined_total: int = 0
    overhead_ms_total: float = 0.0

    def record(self, plan: InjectionPlan) -> None:
        self.requests_total += 1
        if plan.donor_request_id is not None:
            self.requests_with_match += 1
        self.candidates_examined_total += plan.candidates_examined
        self.overhead_ms_total += plan.gateway_overhead_ms

    def summary(self) -> dict:
        if self.requests_total == 0:
            return {"requests_total": 0}
        return {
            "requests_total": self.requests_total,
            "hit_rate": self.requests_with_match / self.requests_total,
            "avg_candidates": self.candidates_examined_total / self.requests_total,
            "avg_overhead_ms": self.overhead_ms_total / self.requests_total,
        }


class JaccardServeGateway:
    """
    Main entry point. Holds the LSH index and pipeline configuration.

    Lifecycle:
        plan = gateway.match(prompt)
        # ... submit to serving engine using plan ...
        gateway.register(request_id, prompt, worker_id)
        # ... when the request completes and KV is evicted ...
        gateway.evict(request_id)
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        shingle_width: int = 5,
        banding: BandingConfig = BandingConfig(),
        jaccard_threshold: float = 0.6,
        max_candidates_per_query: int = 16,
    ):
        self.tokenizer = tokenizer
        self.shingle_width = shingle_width
        self.threshold = jaccard_threshold
        self.max_candidates = max_candidates_per_query
        self.minhasher = MinHasher(
            MinHashConfig(num_perms=banding.num_perms)
        )
        self.index = BandedLSH(banding)
        self.stats = GatewayStats()

    # ------------------------------------------------------------------
    # Per-request API
    # ------------------------------------------------------------------

    def match(self, prompt: str) -> tuple[InjectionPlan, Sequence[int]]:
        """
        Run the matching pipeline for an incoming prompt. Returns the
        injection plan plus the tokenized prompt (so the caller does
        not re-tokenize).
        """
        t0 = time.perf_counter()
        request_id = uuid.uuid4().hex
        token_ids = list(self.tokenizer(prompt))

        # Step 2-3: shingle and MinHash.
        target_shingles = shingle_set(token_ids, k=self.shingle_width)
        target_sig = self.minhasher.signature(target_shingles)

        # Step 4: band lookup.
        candidate_ids = self.index.query_candidates(target_sig)

        plan = InjectionPlan(request_id=request_id)
        plan.candidates_examined = len(candidate_ids)

        # Step 5: verification + Step 6: span resolution.
        if candidate_ids:
            best_donor_id: str | None = None
            best_jaccard = self.threshold  # only accept above threshold
            best_donor_entry: LSHEntry | None = None

            # Limit verifications to bound worst-case CPU.
            for cid in list(candidate_ids)[: self.max_candidates]:
                entry = self.index.get_entry(cid)
                if entry is None:
                    continue
                j = verify_jaccard(target_shingles, entry.shingle_set)
                if j > best_jaccard:
                    best_jaccard = j
                    best_donor_id = cid
                    best_donor_entry = entry

            if best_donor_id is not None and best_donor_entry is not None:
                # Step 6: find the contiguous matching token span.
                span = longest_matching_span(
                    token_ids,
                    self._reconstruct_token_ids(best_donor_entry),
                    self.shingle_width,
                )
                if span is not None:
                    target_span, donor_span = span
                    plan.donor_request_id = best_donor_id
                    plan.donor_worker_id = best_donor_entry.worker_id
                    plan.matched_target_span = target_span
                    plan.matched_donor_span = donor_span
                    plan.measured_jaccard = best_jaccard

        plan.gateway_overhead_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.record(plan)
        return plan, token_ids

    def register(
        self,
        request_id: str,
        token_ids: Sequence[int],
        worker_id: str | None = None,
    ) -> None:
        """
        Admit a request to the index once its KV cache is resident on a
        worker. The signature is recomputed here rather than carried
        from match() to keep the API simple; in production you would
        cache it.
        """
        target_shingles = shingle_set(token_ids, k=self.shingle_width)
        sig = self.minhasher.signature(target_shingles)
        entry = LSHEntry(
            request_id=request_id,
            worker_id=worker_id,
            signature=sig,
            shingle_set=target_shingles,
            num_tokens=len(token_ids),
            token_ids=tuple(token_ids),
        )
        self.index.insert(entry)

    def evict(self, request_id: str) -> None:
        self.index.evict(request_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reconstruct_token_ids(self, entry: LSHEntry) -> Sequence[int] | None:
        """Return the donor's token IDs for span alignment."""
        return entry.token_ids if entry.token_ids else None
