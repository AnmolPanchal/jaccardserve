# JaccardServe

Cross-request prefill acceleration for LLM serving via MinHash-LSH
near-duplicate detection over token shingles.

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh
```

This produces every number and figure in the paper in approximately
two minutes on a single CPU core. No GPU required.

## What's in the repo

```
jaccardserve/
├── paper/jaccardserve_paper.md     # Full paper, with embedded figure refs
├── REPRODUCING_LOCALLY.md          # Detailed local-run guide
├── run_all.sh                      # Master reproduction script
│
├── jaccardserve/                   # Core algorithm package
│   ├── minhash.py                  # ax+b mod p MinHash signatures
│   ├── shingler.py                 # k-shingle token hashing
│   ├── banded_lsh.py               # Online banded LSH index
│   ├── verifier.py                 # Exact Jaccard + LCS span resolution
│   └── gateway.py                  # JaccardServeGateway orchestration
│
├── baselines/                      # Comparison methods
│   ├── vllm_apc_simulator.py       # Block-level exact hashing
│   └── semsharekv_simulator.py     # Matching-layer cosine LSH on synthetic embeddings
│
├── benchmarks/
│   ├── templated_chat.py           # 500 prompts; APC fails here
│   ├── multi_doc_summ.py           # 320 prompts; SemShareKV's workload
│   └── make_figures.py             # All 5 paper figures
│
├── quality/
│   ├── oracle_validation.py        # Ground-truth precision/recall (CPU)
│   └── llm_judge_harness.py        # LLM-judge quality harness (GPU)
│
├── vllm_integration/
│   ├── engine.py                   # JaccardServeEngine wrapper (mock + real)
│   └── ttft_benchmark.py           # Phase 2 TTFT benchmark (GPU)
│
├── analysis/
│   └── span_position_analysis.py   # Span-start consistency analysis
│
├── head_to_head/
│   └── semsharekv_comparison.py    # Head-to-head stub (SemShareKV code pending)
│
├── tests/
│   ├── test_minhash.py             # 5 tests
│   └── test_lsh.py                 # 4 tests including empirical S-curve
│
├── results/                        # Generated CSVs and summaries
└── figures/                        # Generated PNGs (5 paper figures)
```

## Headline numbers

**Matching-layer benchmark — Templated-chat (n=500):**

| Method | Hit rate | Overhead |
|---|---|---|
| vLLM APC | 5.2% | 0.02 ms |
| SemShareKV-sim (medium) | 1.0% | 1.98 ms |
| JaccardServe (balanced) | **97.4%** | **0.28 ms** |

**Matching-layer benchmark — Multi-doc summarization (n=320), oracle F1:**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| vLLM APC | 0.873 | 0.199 | 0.324 |
| SemShareKV-sim (medium) | 0.947 | 0.064 | 0.120 |
| JaccardServe (high-recall) | 0.905 | 0.806 | **0.853** |

**GPU TTFT benchmark — Templated-chat (Qwen2.5-3B, RTX 5070 Ti):**

| Policy | Median TTFT |
|---|---|
| No cache | 18.7 ms |
| vLLM APC only | 14.0 ms |
| JaccardServe + APC (Strategy A) | **10.5 ms** (−44%) |

See `results/phase2_summary.md` for full TTFT results and the quality constraint on this number.

**Quality validation — Strategy A (100 matched pairs, LLM judge):**

Strategy A (prompt-prefix reordering) achieves the TTFT reduction above but produces
a 60% MAJOR-difference rate in LLM-judge evaluation — the reordering breaks
chat-template semantic structure. Strategy A TTFT numbers are reported with this caveat.
Strategy B (block-table injection, no reordering) is the production path; see
`jaccardserve_alt` for that work.

## Reproducing

CPU-only (matching-layer results, figures): `bash run_all.sh`

GPU results (TTFT, quality validation): see `REPRODUCING_LOCALLY.md`.

## Citation

```
Panchal, A. (2026). JaccardServe: Cross-Request Prefill Acceleration
in LLM Serving via MinHash-LSH Token Shingling.
```

## License

MIT.
