"""
LLM-judge quality validation harness.

This script REQUIRES a GPU. It cannot be run in a CPU-only
environment. We do not run it in the reference distribution; it is
provided for the user to run on their own hardware.

What it does:

  1. Generate a workload (templated chat or multi-doc summarization).
  2. For each pair (prompt P, donor D) that JaccardServe declares a
     match, generate two model outputs:
       - O_baseline: model output on P with NO KV reuse
       - O_jaccardserve: model output on P with the donor span's KV
                         injected (Strategy A: prompt-prefix
                         reordering relying on vLLM APC)
  3. Score each pair with a strong judge model (default: GPT-4 via
     OpenAI API). Three-class judgment: equivalent, minor difference,
     major difference.
  4. Report:
       - Equivalence rate (% pairs judged "equivalent")
       - Mean ROUGE-L score O_jaccardserve vs O_baseline
       - Distribution of judge labels
  5. Write per-pair results to results/llm_judge_results.csv.

Acceptance criterion: <= 1% major differences. Empirically this
is what SemShareKV reports and what the JaccardServe paper claims.

Run (local judge — no API key required):
    # Step 1: serve judge model on port 8001
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --api-key local

    # Step 2: run harness against it
    python quality/llm_judge_harness.py \\
        --vllm-model Qwen/Qwen2.5-3B-Instruct \\
        --num-pairs 100 \\
        --judge local \\
        --judge-model Qwen/Qwen2.5-7B-Instruct \\
        --local-endpoint http://localhost:8001

Run (OpenAI):
    export OPENAI_API_KEY=sk-...
    python quality/llm_judge_harness.py \\
        --vllm-model meta-llama/Meta-Llama-3-8B-Instruct \\
        --num-pairs 100 \\
        --judge-model gpt-4o-mini

Notes:
  - Local judge uses vLLM's OpenAI-compatible server. Generate outputs
    first with the generation model, then swap to the judge model.
  - The judge model can also be Claude 3.5 Sonnet via the Anthropic API.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# ROUGE-L (simple implementation - sufficient for QA)
# ---------------------------------------------------------------------------

def rouge_l(a: str, b: str) -> float:
    """ROUGE-L F-measure between two strings, token level."""
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    n, m = len(ta), len(tb)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ta[i - 1] == tb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Judge: GPT-4 via OpenAI API
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an impartial judge. You will see a user prompt and two model responses to it. \
Your job is to determine whether the two responses convey the same information and would be considered \
interchangeable by an end user.

Respond with exactly one of:
  EQUIVALENT - the two responses say substantively the same thing
  MINOR      - small differences in wording, ordering, or formatting; no information loss
  MAJOR      - one response contains information or a conclusion the other does not, or they disagree

Output format: a single JSON object with two keys:
  label: one of EQUIVALENT, MINOR, MAJOR
  reason: one sentence explaining the judgment

USER PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

JSON:"""


def judge_pair_openai(
    prompt: str,
    response_a: str,
    response_b: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Call OpenAI's API to judge a pair. Requires OPENAI_API_KEY env var."""
    from openai import OpenAI  # pip install openai
    client = OpenAI()
    msg = JUDGE_PROMPT.format(
        prompt=prompt, response_a=response_a, response_b=response_b,
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": msg}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            if attempt == 2:
                return {"label": "ERROR", "reason": str(e)}
            time.sleep(2 ** attempt)
    return {"label": "ERROR", "reason": "max retries"}


def judge_pair_local(
    prompt: str,
    response_a: str,
    response_b: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    endpoint: str = "http://localhost:8001",
) -> dict:
    """
    Call a local vLLM OpenAI-compatible server as judge.
    Start the server with:
        vllm serve <model> --port 8001 --api-key local
    """
    from openai import OpenAI
    client = OpenAI(base_url=f"{endpoint}/v1", api_key="local")
    msg = JUDGE_PROMPT.format(
        prompt=prompt, response_a=response_a, response_b=response_b,
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": msg}],
                temperature=0.0,
                max_tokens=200,
            )
            content = resp.choices[0].message.content
            # Extract JSON robustly — vLLM may emit surrounding text.
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON in response: {content[:200]}")
            return json.loads(content[start : end + 1])
        except Exception as e:
            if attempt == 2:
                return {"label": "ERROR", "reason": str(e)}
            time.sleep(2 ** attempt)
    return {"label": "ERROR", "reason": "max retries"}


def judge_pair_anthropic(
    prompt: str,
    response_a: str,
    response_b: str,
    model: str = "claude-sonnet-4-5",
) -> dict:
    """Anthropic alternative."""
    import anthropic
    client = anthropic.Anthropic()
    msg = JUDGE_PROMPT.format(
        prompt=prompt, response_a=response_a, response_b=response_b,
    )
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": msg}],
            )
            content = resp.content[0].text
            # Extract JSON. Robust against minor formatting.
            start = content.find("{")
            end = content.rfind("}")
            return json.loads(content[start : end + 1])
        except Exception as e:
            if attempt == 2:
                return {"label": "ERROR", "reason": str(e)}
            time.sleep(2 ** attempt)
    return {"label": "ERROR", "reason": "max retries"}


# ---------------------------------------------------------------------------
# Workload + paired generation
# ---------------------------------------------------------------------------

def collect_match_pairs(workload, gateway, tokenizer, max_pairs):
    """
    Run the gateway over the workload and collect (prompt, donor_prompt)
    pairs where JaccardServe declares a match. Returns up to max_pairs.

    Each entry includes reordered_token_ids: the Strategy-A rewrite of
    the prompt (matched span first) used to generate the JS candidate.
    """
    prompts_by_rid: dict[str, str] = {}
    pairs = []
    for item in workload:
        if len(pairs) >= max_pairs:
            break
        prompt = item["prompt"] if isinstance(item, dict) else item
        plan, token_ids = gateway.match(prompt)
        gateway.register(plan.request_id, token_ids, worker_id="w0")
        prompts_by_rid[plan.request_id] = prompt
        if plan.donor_request_id is not None:
            donor_prompt = prompts_by_rid.get(plan.donor_request_id)
            if donor_prompt is not None:
                # Build Strategy A reordering: matched span → position 0.
                if plan.matched_target_span:
                    start, end = plan.matched_target_span
                    reordered = token_ids[start:end] + token_ids[:start] + token_ids[end:]
                else:
                    reordered = list(token_ids)
                pairs.append({
                    "prompt": prompt,
                    "donor_prompt": donor_prompt,
                    "jaccard": plan.measured_jaccard,
                    "span_target_start": plan.matched_target_span[0] if plan.matched_target_span else 0,
                    "span_target_end": plan.matched_target_span[1] if plan.matched_target_span else 0,
                    "reordered_token_ids": reordered,
                })
    return pairs


def _build_workload(workload_name: str, num_pairs: int):
    if workload_name == "multidoc":
        from benchmarks.multi_doc_summ import generate_multi_doc_workload
        workload = generate_multi_doc_workload(
            num_groups=20, docs_per_group=8,
            queries_per_group=15, docs_per_query=4, seed=42,
        )
    else:
        from benchmarks.templated_chat import generate_prompts
        workload = [{"prompt": p} for p in generate_prompts(500, seed=0)]
    return workload


def phase_generate(args):
    """
    Phase 1: Load the generation model, collect matched pairs, generate
    baseline and Strategy-A candidate outputs, save to an intermediate CSV.
    Does NOT load or contact the judge model — the GPU is freed after exit.
    """
    workload = _build_workload(args.workload, args.num_pairs)

    print("Loading generation model (vLLM)...")
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm_integration.engine import RealVLLMEngineAdapter
    engine = RealVLLMEngineAdapter(
        model_name=args.vllm_model,
        enforce_eager=True,
        gpu_memory_utilization=0.88,
        max_model_len=4096,
    )

    # Wrap HuggingFace tokenizer into the (str -> list[int]) form the gateway expects.
    hf_tok = engine.get_tokenizer()
    def tok_fn(s: str) -> list:
        return hf_tok(s, add_special_tokens=False)["input_ids"]

    from jaccardserve import BandingConfig, JaccardServeGateway
    gateway = JaccardServeGateway(
        tokenizer=tok_fn,
        banding=BandingConfig(num_bands=20, rows_per_band=4),
        jaccard_threshold=0.5,
        shingle_width=5,
    )

    print("Collecting match pairs...")
    pairs = collect_match_pairs(workload, gateway, tok_fn, args.num_pairs)
    print(f"Collected {len(pairs)} pairs.")
    if not pairs:
        print("No pairs found — try a different workload or lower threshold.")
        return

    print(f"Generating {len(pairs)} pair outputs (baseline + Strategy-A reordered)...")
    gen_rows = []
    for i, pair in enumerate(pairs):
        baseline = engine.generate(pair["prompt"], max_tokens=args.max_tokens,
                                   temperature=0.0)
        jcandidate = engine.generate(
            {"prompt_token_ids": pair["reordered_token_ids"]},
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        rl = rouge_l(baseline["text"], jcandidate["text"])
        gen_rows.append({
            "pair_idx": i,
            "jaccard": round(pair["jaccard"], 4),
            "span_start": pair["span_target_start"],
            "span_end": pair["span_target_end"],
            "prompt": pair["prompt"][:1000],
            "baseline_text": baseline["text"][:500],
            "jcandidate_text": jcandidate["text"][:500],
            "rouge_l": round(rl, 4),
            "baseline_ttft_ms": round(baseline.get("ttft_ms", 0), 2),
            "jcandidate_ttft_ms": round(jcandidate.get("ttft_ms", 0), 2),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)} generated")

    gen_path = os.path.join(args.results_dir, f"llm_judge_generations_{args.workload}.csv")
    with open(gen_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gen_rows[0].keys()))
        w.writeheader()
        w.writerows(gen_rows)
    print(f"\nWrote {gen_path}")
    print("Generation phase complete. Now:")
    print(f"  1. Start judge server: vllm serve {args.judge_model} --port 8001 --api-key local")
    print(f"  2. Run: python quality/llm_judge_harness.py --phase judge --workload {args.workload}")


def phase_judge(args):
    """
    Phase 2: Read generations CSV, call local judge server, write final results.
    Requires vllm serve <judge_model> --port 8001 --api-key local to be running.
    """
    import os
    gen_path = os.path.join(args.results_dir, f"llm_judge_generations_{args.workload}.csv")
    if not os.path.exists(gen_path):
        print(f"Generations file not found: {gen_path}")
        print("Run --phase generate first.")
        return

    with open(gen_path, newline="", encoding="utf-8") as f:
        gen_rows = list(csv.DictReader(f))
    print(f"Loaded {len(gen_rows)} pairs from {gen_path}")

    if args.judge == "local":
        def judge_fn(prompt, response_a, response_b, model):
            return judge_pair_local(prompt, response_a, response_b, model=model,
                                    endpoint=args.local_endpoint)
    elif args.judge == "anthropic":
        judge_fn = judge_pair_anthropic
    else:
        judge_fn = judge_pair_openai

    rows = []
    label_counts: dict = {"EQUIVALENT": 0, "MINOR": 0, "MAJOR": 0, "ERROR": 0}
    rouge_scores = []

    for i, gr in enumerate(gen_rows):
        rouge_scores.append(float(gr["rouge_l"]))
        verdict = judge_fn(
            prompt=gr["prompt"],
            response_a=gr["baseline_text"],
            response_b=gr["jcandidate_text"],
            model=args.judge_model,
        )
        label = verdict.get("label", "ERROR").upper()
        label_counts[label] = label_counts.get(label, 0) + 1
        rows.append({
            "pair_idx": gr["pair_idx"],
            "jaccard": gr["jaccard"],
            "span_start": gr["span_start"],
            "rouge_l": gr["rouge_l"],
            "label": label,
            "reason": verdict.get("reason", "")[:200],
            "baseline_ttft_ms": gr["baseline_ttft_ms"],
            "jcandidate_ttft_ms": gr["jcandidate_ttft_ms"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(gen_rows)} judged | {label_counts}")

    n = len(rows)
    eq_minor = label_counts.get("EQUIVALENT", 0) + label_counts.get("MINOR", 0)
    major_rate = 100 * label_counts.get("MAJOR", 0) / n

    print()
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    for lab, c in label_counts.items():
        print(f"  {lab:12s}: {c:3d}  ({100*c/n:.1f}%)")
    print(f"  Mean ROUGE-L    : {sum(rouge_scores)/n:.4f}")
    print(f"  Non-major rate  : {100*eq_minor/n:.1f}%")
    print(f"  MAJOR-diff rate : {major_rate:.1f}%  (acceptance bar: ≤1.0%)")
    if major_rate <= 1.0:
        print("  RESULT: PASS — Strategy A is quality-preserving.")
    else:
        print("  RESULT: FAIL — MAJOR rate exceeds 1% threshold.")

    out_path = os.path.join(args.results_dir, "llm_judge_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")
    return major_rate, sum(rouge_scores) / n, label_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["generate", "judge", "all"],
                    default="all",
                    help="'generate': collect pairs and generate outputs (3B model); "
                         "'judge': score saved outputs via local judge server (7B model); "
                         "'all': both in one process (requires both models to fit in VRAM)")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--num-pairs", type=int, default=100)
    ap.add_argument("--judge", choices=["openai", "anthropic", "local"],
                    default="local")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--local-endpoint", default="http://localhost:8001",
                    help="Base URL for local vLLM judge server (--judge local only)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workload", choices=["templated", "multidoc"], default="templated",
                    help="Use 'templated' for the high-redundancy workload where "
                         "Strategy A is effective.")
    args = ap.parse_args()

    import os
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("LLM-JUDGE QUALITY VALIDATION")
    print("=" * 60)
    print(f"Phase:       {args.phase}")
    print(f"vLLM model:  {args.vllm_model}")
    print(f"Judge:       {args.judge} / {args.judge_model}")
    print(f"Workload:    {args.workload}")
    print(f"Num pairs:   {args.num_pairs}")
    print()

    if args.phase == "generate":
        phase_generate(args)
    elif args.phase == "judge":
        phase_judge(args)
    else:  # all
        phase_generate(args)
        phase_judge(args)


if __name__ == "__main__":
    main()
