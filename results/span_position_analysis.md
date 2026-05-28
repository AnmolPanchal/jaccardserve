# Span-Position Analysis: Why multi_doc_summ Shows No TTFT Improvement

**Date**: 2026-05-27  
**Model tokenizer**: Qwen/Qwen2.5-3B-Instruct  
**Gateway config**: num_bands=20, rows_per_band=4, jaccard_threshold=0.5, shingle_width=5  
**Prompts per workload**: 200  

---

## Background

The Phase 2 TTFT benchmark found a striking asymmetry:

| Workload | Policy | Median TTFT | vs no\_cache |
|----------|--------|-------------|-------------|
| templated\_chat | js\_apc | 10.5 ms | −44% |
| multi\_doc\_summ | js\_apc | 24.0 ms | −0.3% |

The initial hypothesis was that matched spans in multi\_doc\_summ are not at the
prompt prefix (span\_start >> 0), so APC's prefix-only cache cannot exploit them.

**This hypothesis is refuted.** Both workloads have similar mean relative span starts
(~0.23–0.26). The actual cause is structural — described below.

---

## Matching-Layer Statistics

| Metric | templated\_chat | multi\_doc\_summ |
|--------|----------------|-----------------|
| Requests | 200 | 200 |
| Hits | 191 (95.5%) | 117 (58.5%) |
| Unique span\_start positions | 13 | 19 |
| Std dev of span\_start\_abs | **3.05 tokens** | ~34 tokens |
| Mean relative start | 0.2268 | 0.2346 |
| Median relative start | 0.2414 | 0.2554 |
| Mean span length | 43 tokens | 100 tokens |
| Mean prompt length | 56 tokens | 276 tokens |
| Span coverage ratio | **77%** | **36%** |

---

## The True Mechanism: Span-Start Consistency, Not Span-Start Position

### templated\_chat — Near-Constant Offset

For templated\_chat, 145 of 191 hits (75.9%) have `span_start_abs ∈ [11, 17]`.
Standard deviation across all hits is **3.05 tokens** — essentially a constant offset.

This reflects the prompt structure. Qwen's chat template inserts a fixed header
(`<|im_start|>system\n`) before the user-specific fields (`{name}, {role}, {org}`).
The personalized prefix ("You are an assistant for Alice, who works as...") occupies
~12-16 tokens. The shared system prompt body begins at exactly that offset for every
request.

**Effect on Strategy A (prefix reorder):**
After Strategy A, every matched request's reordered prompt begins with the same
~43-token system-prompt span — an **identical token sequence across all requests**.

- Request 2 (first hit): no APC hit (donor was cached with original order), but this
  reordered prefix is now stored in vLLM's APC cache.
- Request 3 onward: APC fires immediately. The 43-token shared prefix is found in
  cache; only the ~13-token user-specific suffix needs fresh prefill.
- With a 95.5% hit rate, the cascade is near-universal within the first 10 requests.

This is why templated\_chat achieves **44% median TTFT reduction**: after the cascade
warms up, nearly every request prefills only ~23% of its tokens from scratch.

### multi\_doc\_summ — Three Fragmented APC Sub-Caches

For multi\_doc\_summ, the span\_start\_abs values form **three distinct document-slot
bands**, not a tight cluster:

| Band | span\_start range | Hits | % of hits | Corresponds to |
|------|-------------------|------|-----------|----------------|
| Band 0 | [0, 15) | 46 | 39.3% | Doc in slot 1 (immediately after ~7-token header) |
| Band 1 | [15, 80) | 39 | 33.3% | Doc in slot 2 (~70 tokens after header + doc 1) |
| Band 2 | [80, 150) | 32 | 27.4% | Doc in slot 3 (~136 tokens after header + docs 1–2) |

The prompt structure is:
```
Summarize the following documents:      <- ~7 tokens (variable query)

Document 1: {doc_A}                     <- ~69 tokens
Document 2: {doc_B}                     <- ~69 tokens
Document 3: {doc_C}                     <- ~69 tokens
Document 4: {doc_D}                     <- ~69 tokens
```

When JaccardServe matches two queries that share `doc_A` but have different queries
and different orderings of the other documents, the match identifies `doc_A`'s span —
but `doc_A` might sit at slot 1 in one query and slot 2 or 3 in another. This puts
the matched span at **different absolute positions** across requests.

After Strategy A reordering:
- A request whose shared doc was in slot 1 produces a reordered prompt starting with
  the slot-1 doc tokens.
- A request whose shared doc was in slot 2 produces a DIFFERENT reordered prompt.
- These two reordered prompts share no common prefix. They create separate APC cache
  entries.

**Three independent APC sub-caches form** (one per band). The cascade within each band
is real but limited: ~40 hits per band out of 200 total requests, and the first hit in
each band sees no APC benefit (novel prefix to vLLM).

Empirical confirmation from per-request TTFT data for `js_apc`:
- Requests with TTFT < 22 ms (likely APC hits): **22 of 200** (11%)
- No-cache baseline: 0 of 200 requests below 22 ms
- Minimum TTFT for js\_apc: 15.5 ms (vs 22.9 ms no\_cache) — confirms real APC hits exist

Even when APC fires in multi\_doc\_summ, the benefit is smaller: the shared doc is
100 tokens out of 276-token prompts (36% coverage vs 77% for templated\_chat), so 176
tokens still require full prefill per hit.

---

## Root Cause Summary

The TTFT asymmetry is not caused by span position (both workloads have spans well
away from absolute position 0). It is caused by **span-start position consistency**:

| Property | templated\_chat | multi\_doc\_summ |
|----------|----------------|-----------------|
| Span-start std dev | **3 tokens** | ~34 tokens |
| Span-start clusters | 1 (all at ~12–16) | 3 (at 7, 72, 136) |
| APC cascade scope | Universal (~95% of requests) | Fragmented (~11% of requests) |
| Span coverage ratio | 77% | 36% |
| Net TTFT reduction | **−44%** | **−0.3%** |

In templated workloads, the user-specific prefix is short and at a **fixed offset**,
so Strategy A's reordering consistently produces the same prefix — the APC cascade
fires almost immediately and covers nearly all requests.

In multi-doc RAG workloads, the shared document can appear in multiple slot positions.
Each slot creates a different reordered prefix and a separate APC sub-cache. The
cascade is fragmented rather than universal.

---

## Implication for Strategy B

This analysis reveals a precise condition under which Strategy A is maximally
effective: **the shared span must occupy a consistent slot offset across requests**.

When this condition fails (shared content at variable offsets, as in RAG), Strategy B
(block-table injection) is the correct mechanism. Strategy B directly installs the
donor's physical KV blocks at the matched offset without reordering tokens, so it:

1. Does not require span-start consistency across requests.
2. Works for shared content at any position (middle, tail, non-contiguous spans).
3. Does not alter the prompt's attention structure (no quality risk from reordering).

The two findings form a complete workload taxonomy:

| Workload type | Span consistency | Best strategy | Achieved TTFT saving |
|---------------|-----------------|--------------|----------------------|
| Templated / agentic (shared system prompt) | High (std dev ≈ 3 tok) | Strategy A | −44% confirmed |
| Multi-doc / RAG (shared doc at variable slot) | Low (3 bands, std dev ≈ 34 tok) | Strategy B | pending |

This also motivates a practical **strategy selector** at the gateway tier: compute the
variance of matched span-start positions seen recently for a given donor cluster. If
variance is low (< threshold), apply Strategy A; otherwise, route to a Strategy B
engine. This adds negligible per-request overhead (one variance check) and would
recover the multi\_doc\_summ TTFT savings without touching Strategy A's proven gains.

---

## Raw Data

Per-hit span positions (workload, absolute start, relative start, span length) are in
`results/span_positions_raw.csv`.
