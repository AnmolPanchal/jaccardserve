# Reproducing JaccardServe Locally

This document is the step-by-step guide to reproduce every number,
figure, and table in the paper on your own machine.

There are two tiers:

- **CPU-only reproduction.** All matching-layer results, oracle
  precision/recall, S-curve, ablations, and figures. About 2 minutes
  of compute. Required: Python 3.10+.

- **GPU reproduction.** End-to-end TTFT measurements via real vLLM,
  LLM-judge quality validation. Required: NVIDIA GPU with at least
  16 GB VRAM, CUDA 12.1+, Python 3.12 (vLLM constraint).

Start with CPU-only. The GPU tier extends the matching-layer results
with real latency numbers.

---

## Tier 1: CPU-only reproduction (~2 minutes)

### Prerequisites

- Python 3.10 or newer
- About 200 MB of disk

### Setup

```bash
# Clone the repo.
git clone https://github.com/AnmolPanchal/jaccardserve.git
cd jaccardserve

# Create a clean virtual environment.
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# Install the small set of CPU dependencies.
pip install -r requirements.txt
```

### Run everything

```bash
bash run_all.sh
```

This will:

1. Run the templated-chat benchmark (500 prompts, three methods).
2. Run the multi-doc summarization benchmark (320 prompts, three methods).
3. Run the oracle precision/recall validation.
4. Run the unit-test suite including the empirical S-curve test.
5. Generate all five paper figures.

Expected output structure:

```
results/
├── templated_chat_results.csv          # method × config × hit rate
├── multi_doc_summ_results.csv          # same, for multi-doc
├── multi_doc_summ_per_prompt.csv       # per-prompt match decisions
└── quality_oracle_precision_recall.csv # ground-truth P/R for each method

figures/
├── fig1_s_curve.png                    # empirical vs theoretical S-curve
├── fig2_hit_rate_comparison.png        # cross-method hit rate bar chart
├── fig3_precision_recall.png           # oracle precision-recall scatter
├── fig4_match_distributions.png        # Jaccard / span-coverage histograms
└── fig5_br_heatmap.png                 # (b, r) hit-rate heatmap
```

### Verify the numbers

Open `results/multi_doc_summ_results.csv`. You should see:

| method | config | hit_rate |
|---|---|---|
| vLLM_APC | block=16 | ~0.197 |
| SemShareKV_sim | medium | ~0.059 |
| JaccardServe | balanced | ~0.709 |
| JaccardServe | high-recall | ~0.759 |

And `results/quality_oracle_precision_recall.csv`:

| method | precision | recall | F1 |
|---|---|---|---|
| vLLM_APC | ~0.873 | ~0.199 | ~0.324 |
| JaccardServe (high-recall) | ~0.905 | ~0.806 | ~0.853 |

Slight variation is expected if you change seeds. The relative ordering
of methods does not change.

### Run individual pieces

```bash
# Just the multi-doc summ benchmark:
python benchmarks/multi_doc_summ.py --num-groups 20

# Just the oracle validation:
python quality/oracle_validation.py

# Just the unit tests:
python -m unittest discover -s tests -v

# Just the figures (after the above have produced CSVs):
python benchmarks/make_figures.py
```

### Custom workloads and parameters

Every script takes command-line flags. Examples:

```bash
# Larger workload, more groups:
python benchmarks/multi_doc_summ.py --num-groups 50 --queries-per-group 30

# Just the unit tests with a fixed seed:
PYTHONHASHSEED=0 python -m unittest tests.test_minhash
```

### CPU smoke test of the end-to-end gateway

```bash
python vllm_integration/engine.py
```

This runs the JaccardServeEngine wrapping a mock vLLM (no GPU). It
prints the matching decisions and a simulated prefill time per request.
On the included demo prompts, three of the five fire JaccardServe hits
with measured Jaccard ~0.5-0.8 and simulated prefill drops 7-8x.

---

## Tier 2: GPU reproduction

### Prerequisites

- NVIDIA GPU, 16+ GB VRAM.
- CUDA 12.1 or newer.
- Python 3.12 (required by vLLM 0.21+).
- WSL2 on Windows, or native Linux.
- HuggingFace account with access to the generation model.

### Setup

```bash
# In WSL2 Ubuntu or Linux:
python3.12 -m venv ~/venv_jaccardserve
source ~/venv_jaccardserve/bin/activate

pip install vllm==0.21.0 openai bitsandbytes>=0.48.1
# Set before any vLLM import on Blackwell (SM_120):
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### Phase 2: End-to-end TTFT measurement

```bash
# Templated-chat workload (100 requests per policy):
python vllm_integration/ttft_benchmark.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --workload templated_chat \
    --num-requests 200

# Multi-doc summarization workload:
python vllm_integration/ttft_benchmark.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --workload multi_doc_summ \
    --num-requests 200
```

Results are written to `results/ttft_<workload>.csv` and
`results/ttft_<workload>_per_request.csv`. A summary of the measured
results is in `results/phase2_summary.md`.

Measured on RTX 5070 Ti (Blackwell, 16 GB), Qwen2.5-3B-Instruct, vLLM 0.21.0:

| Workload | Policy | Median TTFT |
|---|---|---|
| templated_chat | no_cache | 18.7 ms |
| templated_chat | apc_only | 14.0 ms |
| templated_chat | js_apc (Strategy A) | **10.5 ms** |
| multi_doc_summ | no_cache | 24.1 ms |
| multi_doc_summ | js_apc (Strategy A) | 24.0 ms |

See `results/phase2_summary.md` for the quality constraint on the
templated_chat Strategy A numbers.

### Phase 3: Quality validation (LLM judge)

The quality harness runs in two phases to stay within 16 GB VRAM: first
generate outputs with the generation model, then judge with a quantized
7B model served separately.

```bash
# Step 1: Generate outputs (loads Qwen2.5-3B, exits cleanly)
python quality/llm_judge_harness.py \
    --phase generate \
    --vllm-model Qwen/Qwen2.5-3B-Instruct \
    --workload templated \
    --num-pairs 100

# Step 2: Serve the judge model (in a separate terminal)
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --quantization bitsandbytes --load-format bitsandbytes \
    --port 8001 --api-key local \
    --gpu-memory-utilization 0.95

# Step 3: Run the judge
python quality/llm_judge_harness.py \
    --phase judge \
    --judge local \
    --local-endpoint http://localhost:8001
```

Results are written to `results/llm_judge_results.csv`. A summary
including root-cause analysis is in `results/phase3_summary.md`.

**Measured result**: Strategy A (prompt-prefix reordering) achieves
−44% median TTFT but produces a 60% MAJOR-difference rate in
LLM-judge evaluation. The reordering breaks chat-template semantic
structure by displacing the user-specific preamble to mid-context.
This result is reported in Section 7.9 of the paper.

### Head-to-head with SemShareKV

The stub is in `head_to_head/semsharekv_comparison.py`. A full
head-to-head requires a public SemShareKV reference implementation,
which was not available as of the paper's writing date (see stub for
the checked date).

---

## Troubleshooting

### "HF tokenizer unavailable" warning

Install `transformers` to get the GPT-2 tokenizer used in the paper's
reported numbers:

```bash
pip install transformers
```

### Tests fail with "MinHash variance" error

The MinHash test tolerates ~3 standard deviations of variance at
m=128. If you see a single failure on `test_partial_overlap`, rerun
with a different seed:

```bash
PYTHONHASHSEED=0 python -m unittest tests.test_minhash
```

### vLLM import errors

`vLLM` only runs on CUDA-enabled Linux. On Mac or Windows-CPU, use the
mock engine (`JaccardServeEngine.mock()`); the matching pipeline is
fully exercised, only the model generation is simulated.

### Blackwell (SM_120) FlashInfer crash

Set `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM. This is
already baked into `ttft_benchmark.py` and `llm_judge_harness.py`.
