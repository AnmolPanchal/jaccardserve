# Phase 2 — TTFT Benchmark Results

**Date**: 2026-05-27  
**Model**: Qwen/Qwen2.5-3B-Instruct (fp16)  
**GPU**: RTX 5070 Ti (Blackwell SM_120, 16 GB)  
**vLLM**: 0.21.0, enforce_eager=True, gpu_memory_utilization=0.88  
**Prompts per policy**: 200  

> **IMPORTANT — Phase 3 quality note**: The `js_apc` policy uses Strategy A (prompt-prefix
> reordering). Phase 3 LLM-judge validation found a **60% MAJOR-difference rate** for this
> strategy on chat-template workloads. The TTFT numbers below are real but reflect a
> quality-unsafe mechanism. They bound the latency achievable *if* quality loss were acceptable
> (e.g., raw completion over homogeneous corpora where token order is irrelevant). For
> production chat workloads, Strategy B (block-table injection, no reordering) is required.
> See `results/phase3_summary.md` and the paper's Section 7.9.

## Policies

| Policy | APC | JaccardServe |
|--------|-----|--------------|
| no_cache | off | off |
| apc_only | on  | off |
| js_apc   | on  | on (Strategy A: prefix reorder — quality-unsafe for chat) |

JaccardServe config: num_bands=20, rows_per_band=4, jaccard_threshold=0.5, shingle_width=5.

---

## templated_chat (high-redundancy: same template, varied slots)

| Policy | Median (ms) | p99 (ms) | Mean (ms) | min | max |
|--------|------------|----------|-----------|-----|-----|
| no_cache  | 18.73 | 23.88 | 18.18 | 10.98 | 179.84 |
| apc_only  | 13.98 | 176.87 | 198.32* | 10.70 | 36410.02* |
| js_apc    | **10.47** | 128.72 | 13.19 | 9.71 | 165.56 |

\* apc_only mean/max are inflated by a Triton JIT compilation spike on the policy's first
  request (~36 s). Triton compiles a new kernel variant for the prefix-caching code path.
  Subsequent requests are unaffected. The median (13.98 ms) is the reliable metric.

**Key results:**
- js_apc vs no_cache: **−44% median TTFT** (18.7 ms → 10.5 ms) ⚠️ *quality-unsafe*
- js_apc vs apc_only: **−25% median TTFT** (14.0 ms → 10.5 ms) ⚠️ *quality-unsafe*
- APC alone already helps on this workload (−25% vs no_cache)
- JaccardServe's Strategy A stacks on top, reordering matched token spans to position 0
  so vLLM sees exact prefix hits for near-duplicate requests
- **Phase 3 finding**: This reordering breaks chat-template semantic structure; 60% of
  responses are MAJOR-quality degraded. The latency saving and the quality loss are the
  same mechanism. These numbers must be reported with the quality caveat.

---

## multi_doc_summ (low-redundancy: unique document groups, longer prompts)

| Policy | Median (ms) | p99 (ms) | Mean (ms) | min | max |
|--------|------------|----------|-----------|-----|-----|
| no_cache  | 24.08 | 26.75 | 24.58 | 22.91 | 121.28 |
| apc_only  | 25.89 | 28.66 | 26.10 | 17.72 | 117.95 |
| js_apc    | **24.01** | 28.08 | 24.06 | 15.46 | 95.13 |

**Key results:**
- js_apc vs no_cache: **−0.3% median TTFT** (24.1 ms → 24.0 ms)
- apc_only is 7% *slower* than no_cache — APC lookup overhead with no cache hits
- JaccardServe returns to near no_cache baseline; low Jaccard similarity across doc groups
  means few rewrite opportunities, but the gateway overhead is negligible

---

## Interpretation

JaccardServe's TTFT reduction is workload-dependent, as expected:

- On **high-redundancy prompts** (templated chat, RAG with shared system prompts), the
  near-duplicate detector fires frequently. Strategy A's prefix reordering lets vLLM APC
  reuse KV for the matched span. Median TTFT reduction is substantial (−44%).
- On **low-redundancy prompts** (diverse multi-doc summarisation), few pairs exceed the
  Jaccard threshold, so the gateway mostly passes prompts through unchanged. The overhead
  is <0.1 ms on median and the benchmark shows no regression vs no_cache.

The TTFT asymmetry confirms the matching layer fires where expected. However, the
−44% saving is **inseparable from Strategy A's quality failure**: both arise from
prompt reordering. The publishable claim from Phase 2 is the matching layer's hit rate
and negligible overhead, not the Strategy A TTFT numbers.

**What Phase 2 does establish for the paper:**
- The matching layer correctly identifies high-Jaccard pairs (95.5% hit rate on templated).
- Gateway overhead is negligible on non-matching workloads (−0.3% TTFT on multi_doc).
- The span-start consistency analysis (see `results/span_position_analysis.md`) explains
  the mechanism and motivates Strategy B (block-table injection) as the quality-safe path.

---

## Notes

- `VLLM_USE_FLASHINFER_SAMPLER=0` required to disable FlashInfer's JIT-compiled sampler
  on Blackwell (SM_120); falls back to PyTorch-native top-k/top-p.
- `enforce_eager=True` disables CUDA graphs; necessary for WSL2 compatibility.
- Per-request CSVs at `results/ttft_<workload>_per_request.csv` for full distributions.
