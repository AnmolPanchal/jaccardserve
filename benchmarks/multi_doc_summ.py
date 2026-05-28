"""
Multi-document summarization benchmark.

Generates a workload that simulates the Multi-News pattern: groups of
queries where each query asks the model to summarize a subset of
overlapping documents in different orderings or with different
emphasis. This is the workload SemShareKV used in their paper, so
running JaccardServe on the same workload structure is the
direct comparison.

Each benchmark group consists of:
  - A base set of M source "documents" (synthetic, ~60 tokens each)
  - K queries that summarize K-of-M documents in different orderings
  - One "outlier" query that summarizes different documents

The benchmark reports matching layer characteristics for all three
methods on the same prompt stream:
  - vLLM APC (block-level exact)
  - SemShareKV simulator (cosine LSH on synthetic embeddings)
  - JaccardServe (MinHash-LSH on token shingles)

Outputs:
  - results/multi_doc_summ_results.csv: per-(method, config) row.
  - results/multi_doc_summ_per_prompt.csv: per-prompt match decisions
    by each method, used downstream for quality validation.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import (
    SimHashConfig,
    SyntheticTokenEmbedding,
    evaluate_apc,
    evaluate_semsharekv_simulated,
)
from jaccardserve import BandingConfig, JaccardServeGateway


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

DOC_VOCAB = [
    "earnings", "revenue", "growth", "decline", "merger", "acquisition",
    "regulator", "compliance", "court", "ruling", "appeal", "injunction",
    "market", "share", "competitor", "valuation", "billion", "million",
    "quarter", "fiscal", "investor", "analyst", "outlook", "guidance",
    "technology", "platform", "infrastructure", "deployment", "scalability",
    "patient", "treatment", "trial", "efficacy", "regulatory", "approval",
    "policy", "legislation", "framework", "proposal", "committee", "hearing",
    "researchers", "study", "data", "findings", "methodology", "conclusion",
    "supply", "chain", "logistics", "shortage", "demand", "production",
    "agreement", "partnership", "joint", "venture", "stake", "shareholder",
]

TEMPLATES = [
    "Summarize the following documents in three sentences:\n\n{docs}\n\nFocus on key findings.",
    "Given these source documents:\n\n{docs}\n\nProvide a concise summary highlighting the main themes.",
    "Read the documents below and produce a structured summary:\n\n{docs}",
    "The following articles cover related topics:\n\n{docs}\n\nWrite a unified summary.",
]


def generate_document(seed: int, length: int = 60) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(DOC_VOCAB) for _ in range(length))


def generate_multi_doc_workload(
    num_groups: int,
    docs_per_group: int,
    queries_per_group: int,
    docs_per_query: int,
    seed: int = 0,
) -> list[dict]:
    """Generate the workload. Returns list of dicts with prompt + metadata."""
    rng = random.Random(seed)
    out = []
    docs_by_group: dict[int, list[str]] = {}
    for g in range(num_groups):
        docs_by_group[g] = [
            generate_document(seed=g * 100 + i) for i in range(docs_per_group)
        ]

    for g in range(num_groups):
        docs = docs_by_group[g]
        for q in range(queries_per_group):
            template = rng.choice(TEMPLATES)
            chosen_ids = sorted(rng.sample(range(docs_per_group), docs_per_query))
            order = chosen_ids[:]
            rng.shuffle(order)
            if rng.random() < 0.3 and q > 0:
                swap_in_candidates = [
                    i for i in range(docs_per_group) if i not in chosen_ids
                ]
                if swap_in_candidates:
                    order[0] = rng.choice(swap_in_candidates)
            doc_text = "\n\n".join(f"[doc {i}] {docs[i]}" for i in order)
            prompt = template.format(docs=doc_text)
            out.append({
                "prompt": prompt,
                "group_id": g,
                "is_outlier": False,
                "doc_ids": tuple(order),
            })

    for g in range(num_groups):
        other_g = (g + num_groups // 2) % num_groups
        docs = docs_by_group[other_g]
        chosen_ids = sorted(rng.sample(range(docs_per_group), docs_per_query))
        template = rng.choice(TEMPLATES)
        doc_text = "\n\n".join(f"[doc {i}] {docs[i]}" for i in chosen_ids)
        prompt = template.format(docs=doc_text)
        out.append({
            "prompt": prompt,
            "group_id": g,
            "is_outlier": True,
            "doc_ids": tuple(chosen_ids),
        })

    rng.shuffle(out)
    return out


def make_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        print("[info] Using gpt2 tokenizer.")
        return lambda s: tok(s, add_special_tokens=False)["input_ids"]
    except Exception as e:
        print(f"[warn] HF tokenizer unavailable ({e}); using whitespace fallback.")
        vocab: dict[str, int] = {}

        def fallback(s: str) -> list[int]:
            ids = []
            for word in s.split():
                if word not in vocab:
                    vocab[word] = len(vocab) + 1
                ids.append(vocab[word])
            return ids

        return fallback


@dataclass
class JSResult:
    hit_rate: float
    avg_overhead_ms: float
    avg_candidates: float
    requests_total: int
    avg_jaccard_when_hit: float
    avg_span_coverage_when_hit: float


def evaluate_jaccardserve(prompts, tokenizer, banding, threshold, shingle_width):
    gw = JaccardServeGateway(
        tokenizer=tokenizer,
        shingle_width=shingle_width,
        banding=banding,
        jaccard_threshold=threshold,
    )
    per_prompt = []
    jaccards = []
    span_coverages = []
    for i, p in enumerate(prompts):
        plan, token_ids = gw.match(p)
        gw.register(plan.request_id, token_ids, worker_id="w0")
        hit = plan.donor_request_id is not None
        span_cov = 0.0
        if hit and plan.matched_target_span:
            start, end = plan.matched_target_span
            span_cov = (end - start) / max(1, len(token_ids))
            jaccards.append(plan.measured_jaccard)
            span_coverages.append(span_cov)
        per_prompt.append({
            "prompt_idx": i,
            "hit": hit,
            "jaccard": plan.measured_jaccard if hit else 0.0,
            "span_coverage": span_cov,
            "candidates_examined": plan.candidates_examined,
            "overhead_ms": plan.gateway_overhead_ms,
            "donor_id": plan.donor_request_id or "",
        })

    summary = gw.stats.summary()
    return JSResult(
        hit_rate=summary["hit_rate"],
        avg_overhead_ms=summary["avg_overhead_ms"],
        avg_candidates=summary["avg_candidates"],
        requests_total=len(prompts),
        avg_jaccard_when_hit=sum(jaccards) / len(jaccards) if jaccards else 0.0,
        avg_span_coverage_when_hit=sum(span_coverages) / len(span_coverages) if span_coverages else 0.0,
    ), per_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-groups", type=int, default=20)
    ap.add_argument("--docs-per-group", type=int, default=8)
    ap.add_argument("--queries-per-group", type=int, default=15)
    ap.add_argument("--docs-per-query", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    workload = generate_multi_doc_workload(
        args.num_groups, args.docs_per_group, args.queries_per_group,
        args.docs_per_query, seed=args.seed,
    )
    prompts = [item["prompt"] for item in workload]
    print(f"Generated {len(prompts)} prompts ({args.num_groups} groups, "
          f"{args.queries_per_group}+1 queries each).")

    tokenizer = make_tokenizer()

    print("\n[1/3] vLLM APC simulator...")
    apc = evaluate_apc(prompts, tokenizer, block_size=16)
    print(f"  hit_rate={apc.hit_rate:.4f}  blocks={apc.avg_blocks_matched:.2f}  "
          f"overhead={apc.avg_overhead_ms:.3f}ms")

    print("\n[2/3] SemShareKV matching-layer simulator...")
    embedder = SyntheticTokenEmbedding(dim=768)
    semshare_configs = [
        ("low", SimHashConfig(num_planes=64, num_tables=10), 0.80),
        ("medium", SimHashConfig(num_planes=128, num_tables=20), 0.85),
        ("high", SimHashConfig(num_planes=128, num_tables=40), 0.90),
    ]
    semshare_results = []
    for name, cfg, thr in semshare_configs:
        res = evaluate_semsharekv_simulated(prompts, tokenizer, embedder, threshold=thr, config=cfg)
        semshare_results.append((name, cfg, thr, res))
        print(f"  [{name}] planes={cfg.num_planes} tables={cfg.num_tables} thr={thr}: "
              f"hit_rate={res.hit_rate:.4f}  overhead={res.avg_overhead_ms:.3f}ms")

    print("\n[3/3] JaccardServe sweep...")
    js_configs = [
        ("balanced", BandingConfig(num_bands=20, rows_per_band=4), 0.5, 5),
        ("high-prec", BandingConfig(num_bands=20, rows_per_band=8), 0.7, 5),
        ("high-recall", BandingConfig(num_bands=50, rows_per_band=4), 0.4, 5),
    ]
    js_results = []
    per_prompt_balanced = None
    for name, banding, thr, k in js_configs:
        res, per_prompt = evaluate_jaccardserve(prompts, tokenizer, banding, thr, k)
        js_results.append((name, banding, thr, k, res))
        if name == "balanced":
            per_prompt_balanced = per_prompt
        print(f"  [{name}] b={banding.num_bands} r={banding.rows_per_band} "
              f"thr={thr} k={k}: hit_rate={res.hit_rate:.4f}  "
              f"overhead={res.avg_overhead_ms:.3f}ms  "
              f"J={res.avg_jaccard_when_hit:.3f}  span_cov={res.avg_span_coverage_when_hit:.3f}")

    summary_path = os.path.join(args.results_dir, "multi_doc_summ_results.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "config", "hit_rate", "avg_overhead_ms",
                    "avg_candidates", "avg_jaccard", "avg_span_coverage", "n"])
        w.writerow(["vLLM_APC", "block=16", f"{apc.hit_rate:.6f}",
                    f"{apc.avg_overhead_ms:.4f}", "-", "-", "-", apc.requests_total])
        for name, cfg, thr, res in semshare_results:
            w.writerow([
                "SemShareKV_sim", f"{name}_planes={cfg.num_planes}_tables={cfg.num_tables}_thr={thr}",
                f"{res.hit_rate:.6f}", f"{res.avg_overhead_ms:.4f}",
                f"{res.avg_candidates:.2f}", "-", "-", res.requests_total,
            ])
        for name, b, thr, k, res in js_results:
            w.writerow([
                "JaccardServe", f"{name}_b={b.num_bands}_r={b.rows_per_band}_thr={thr}_k={k}",
                f"{res.hit_rate:.6f}", f"{res.avg_overhead_ms:.4f}",
                f"{res.avg_candidates:.2f}", f"{res.avg_jaccard_when_hit:.4f}",
                f"{res.avg_span_coverage_when_hit:.4f}", res.requests_total,
            ])
    print(f"\nWrote {summary_path}")

    if per_prompt_balanced is not None:
        per_prompt_path = os.path.join(args.results_dir, "multi_doc_summ_per_prompt.csv")
        with open(per_prompt_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "prompt_idx", "group_id", "is_outlier",
                "hit", "jaccard", "span_coverage",
                "candidates_examined", "overhead_ms", "donor_id",
            ])
            w.writeheader()
            for item, row in zip(workload, per_prompt_balanced):
                w.writerow({
                    "prompt_idx": row["prompt_idx"],
                    "group_id": item["group_id"],
                    "is_outlier": int(item["is_outlier"]),
                    "hit": int(row["hit"]),
                    "jaccard": f"{row['jaccard']:.4f}",
                    "span_coverage": f"{row['span_coverage']:.4f}",
                    "candidates_examined": row["candidates_examined"],
                    "overhead_ms": f"{row['overhead_ms']:.4f}",
                    "donor_id": row["donor_id"],
                })
        print(f"Wrote {per_prompt_path}")


if __name__ == "__main__":
    main()
