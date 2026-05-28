"""
Phase 2 TTFT benchmark: end-to-end time-to-first-token measurement.

Measures three cache policies on two workloads:
  no_cache  — APC disabled, cold prefill every request
  apc_only  — APC enabled, exact-prefix cache only
  js_apc    — APC enabled + JaccardServe gateway (Strategy A)

Strategy A: when JaccardServe finds a near-duplicate match, reorder the
prompt tokens so the matched span appears at position 0. vLLM's APC then
sees an exact prefix hit and reuses KV for those tokens; only the residual
suffix needs full prefill. Token IDs are submitted directly to vLLM so no
text decode/re-encode round-trip is needed.

TTFT is measured via AsyncLLMEngine streaming: wall-clock time from
request submission to arrival of the first output token.

Outputs:
  results/ttft_templated_chat.csv
  results/ttft_multi_doc_summ.csv

Usage:
  python vllm_integration/ttft_benchmark.py \\
      --workload templated_chat \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --num-prompts 200
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from typing import Optional

# Disable flashinfer's JIT-compiled sampler — it requires nvcc which may not
# be installed. vLLM falls back to PyTorch-native top-k/top-p which is
# sufficient for TTFT measurement and has identical numerical results.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaccardserve import BandingConfig, JaccardServeGateway

POLICIES = ["no_cache", "apc_only", "js_apc"]
CHECKPOINT_EVERY = 50  # write intermediate CSV after this many prompts


# ---------------------------------------------------------------------------
# TTFT measurement via AsyncLLMEngine streaming
# ---------------------------------------------------------------------------

async def measure_ttft_async(engine, prompt_input, request_id: str) -> float:
    """Return TTFT in ms. max_tokens=1 stops after the first generated token."""
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=1, temperature=0.0)
    t0 = asyncio.get_event_loop().time()
    async for output in engine.generate(prompt_input, params, request_id):
        if output.outputs and output.outputs[0].token_ids:
            return (asyncio.get_event_loop().time() - t0) * 1000.0
    return (asyncio.get_event_loop().time() - t0) * 1000.0


async def run_policy_async(
    model_name: str,
    prompts: list,
    policy: str,
    model_tokenizer,
    results_dir: str,
    workload_name: str,
) -> list[float]:
    """
    Load a fresh engine for `policy`, run all prompts serially, return TTFTs.
    Writes a checkpoint CSV every CHECKPOINT_EVERY prompts so progress survives
    an interruption.
    """
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    enable_apc = policy != "no_cache"
    engine_args = AsyncEngineArgs(
        model=model_name,
        enable_prefix_caching=enable_apc,
        max_model_len=4096,
        dtype="float16",
        gpu_memory_utilization=0.88,
        enable_log_requests=False,
        disable_log_stats=True,
        enforce_eager=True,  # skip CUDA graph capture; avoids flashinfer JIT on Blackwell
    )
    print(f"  Loading engine (apc={enable_apc})...")
    t_load = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"  Engine ready in {time.perf_counter()-t_load:.1f}s")

    # Fresh gateway per policy run.
    gateway: Optional[JaccardServeGateway] = None
    if policy == "js_apc":
        def tok_fn(s: str) -> list[int]:
            return model_tokenizer(s, add_special_tokens=False)["input_ids"]
        gateway = JaccardServeGateway(
            tokenizer=tok_fn,
            banding=BandingConfig(num_bands=20, rows_per_band=4),
            jaccard_threshold=0.5,
            shingle_width=5,
        )

    # Single warmup request to trigger Triton JIT before timed loop.
    # Without this, the first policy run's first request is an outlier (30s+),
    # dominating the mean and making policy comparison unfair.
    print("  Warming up Triton JIT...")
    async for _ in engine.generate("warmup", SamplingParams(max_tokens=1, temperature=0.0), "warmup_0"):
        pass
    print("  Warmup done.")

    ttfts: list[float] = []
    ckpt_path = os.path.join(
        results_dir, f"ttft_{workload_name}_{policy}_ckpt.csv"
    )

    for i, item in enumerate(prompts):
        prompt_text = item["prompt"] if isinstance(item, dict) else item
        prompt_input = prompt_text

        if policy == "js_apc" and gateway is not None:
            def tok_fn_inner(s: str) -> list[int]:
                return model_tokenizer(s, add_special_tokens=False)["input_ids"]
            token_ids = tok_fn_inner(prompt_text)
            plan, token_ids = gateway.match(prompt_text)

            if plan.matched_target_span:
                start, end = plan.matched_target_span
                # Strategy A: reorder tokens so matched span is the prefix.
                # vLLM APC will then find an exact prefix hit for those tokens.
                matched = token_ids[start:end]
                before = token_ids[:start]
                after = token_ids[end:]
                rewritten = matched + before + after
                prompt_input = {"prompt_token_ids": rewritten}

            gateway.register(plan.request_id, token_ids, worker_id="w0")

        ttft = await measure_ttft_async(engine, prompt_input, f"{policy}_{i}")
        ttfts.append(ttft)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            _write_checkpoint(ckpt_path, ttfts, policy)
            s = sorted(ttfts)
            n = len(s)
            print(f"  [{policy}] {i+1}/{len(prompts)} "
                  f"median={s[n//2]:.1f}ms  p99={s[min(int(n*.99),n-1)]:.1f}ms")

    # Remove checkpoint on clean completion.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    engine.shutdown()
    import torch
    torch.cuda.empty_cache()
    return ttfts


def _write_checkpoint(path: str, ttfts: list[float], policy: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "policy", "ttft_ms"])
        for i, v in enumerate(ttfts):
            w.writerow([i, policy, f"{v:.3f}"])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(ttfts: list[float]) -> dict:
    s = sorted(ttfts)
    n = len(s)
    return {
        "n": n,
        "median_ms": round(s[n // 2], 2),
        "p99_ms": round(s[min(int(n * 0.99), n - 1)], 2),
        "mean_ms": round(sum(s) / n, 2),
        "min_ms": round(s[0], 2),
        "max_ms": round(s[-1], 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 2 TTFT benchmark")
    ap.add_argument("--workload", choices=["templated_chat", "multi_doc_summ"],
                    required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--policies", nargs="+", default=POLICIES,
                    choices=POLICIES,
                    help="Subset of policies to run (default: all three)")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Build workload.
    if args.workload == "templated_chat":
        from benchmarks.templated_chat import generate_prompts
        prompts = generate_prompts(args.num_prompts, seed=0)
    else:
        from benchmarks.multi_doc_summ import generate_multi_doc_workload
        raw = generate_multi_doc_workload(
            num_groups=20, docs_per_group=8,
            queries_per_group=15, docs_per_query=4, seed=42,
        )
        prompts = raw[: args.num_prompts]

    print(f"Workload : {args.workload}")
    print(f"Model    : {args.model}")
    print(f"Prompts  : {len(prompts)}")
    print(f"Policies : {args.policies}")

    # Load HuggingFace tokenizer once (shared across policies).
    from transformers import AutoTokenizer
    print(f"\nLoading tokenizer for {args.model}...")
    hf_tok = AutoTokenizer.from_pretrained(args.model)

    summary_rows = []
    per_request_rows: list[dict] = []

    for policy in args.policies:
        print(f"\n{'='*60}")
        print(f"Policy: {policy}")
        print(f"{'='*60}")
        t0 = time.perf_counter()

        ttfts = asyncio.run(run_policy_async(
            model_name=args.model,
            prompts=prompts,
            policy=policy,
            model_tokenizer=hf_tok,
            results_dir=args.results_dir,
            workload_name=args.workload,
        ))

        elapsed = time.perf_counter() - t0
        s = stats(ttfts)
        print(f"\n  median={s['median_ms']}ms  p99={s['p99_ms']}ms  "
              f"mean={s['mean_ms']}ms  wall={elapsed:.0f}s")

        summary_rows.append({
            "workload": args.workload,
            "policy": policy,
            "model": args.model,
            **s,
        })
        for i, v in enumerate(ttfts):
            per_request_rows.append({
                "workload": args.workload,
                "policy": policy,
                "prompt_idx": i,
                "ttft_ms": round(v, 3),
            })

    # Write summary CSV.
    summary_path = os.path.join(args.results_dir, f"ttft_{args.workload}.csv")
    with open(summary_path, "w", newline="") as f:
        fields = ["workload", "policy", "model", "n",
                  "median_ms", "p99_ms", "mean_ms", "min_ms", "max_ms"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {summary_path}")

    # Write per-request CSV for diagnostics.
    detail_path = os.path.join(
        args.results_dir, f"ttft_{args.workload}_per_request.csv"
    )
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["workload", "policy", "prompt_idx", "ttft_ms"]
        )
        w.writeheader()
        w.writerows(per_request_rows)
    print(f"Wrote {detail_path}")

    # Print final table.
    print(f"\n{'='*60}")
    print(f"TTFT SUMMARY — {args.workload}")
    print(f"{'='*60}")
    print(f"{'Policy':<14} {'Median':>10} {'p99':>10} {'Mean':>10}")
    print(f"{'-'*44}")
    for r in summary_rows:
        print(f"{r['policy']:<14} {r['median_ms']:>9.1f}ms "
              f"{r['p99_ms']:>9.1f}ms {r['mean_ms']:>9.1f}ms")


if __name__ == "__main__":
    main()
