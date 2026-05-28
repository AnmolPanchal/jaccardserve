"""
Templated-chat benchmark.

Comparable structure to multi_doc_summ.py but with the templated-prompt
workload from the paper's introductory example. This is the canonical
failure case for vLLM APC: prompts share 90%+ of their tokens but the
per-user substitution lands in the first block, so block-level hashing
misses the reuse opportunity.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import (
    SimHashConfig,
    SyntheticTokenEmbedding,
    evaluate_apc,
    evaluate_semsharekv_simulated,
)
from jaccardserve import BandingConfig, JaccardServeGateway


NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Kevin", "Linda", "Mallory", "Niaj", "Olivia", "Peggy",
    "Quentin", "Rupert", "Sybil", "Trent", "Uma", "Victor", "Wendy", "Xavier",
    "Yvonne", "Zara",
]
ROLES = [
    "software engineer", "data scientist", "product manager",
    "research analyst", "trader", "physician", "lawyer", "designer",
]
ORGS = [
    "Meta", "Google", "Anthropic", "OpenAI", "Microsoft", "Apple",
    "Stripe", "Citadel", "JPMorgan", "Goldman Sachs",
]
TEMPLATES = [
    (
        "You are an assistant for {name}, who works as a {role} at {org}. "
        "Help them with their request. Be concise and accurate. Always cite "
        "sources when you reference factual claims. If you do not know the "
        "answer, say so rather than guessing."
    ),
    (
        "Acting on behalf of {name}, a {role} based at {org}, respond to the "
        "following user query. Maintain a professional tone. Decompose the "
        "problem before answering. If clarification is needed, ask."
    ),
]
USER_TURNS = [
    "Summarize the attached document.",
    "What is the capital of France?",
    "Write a SQL query to find the top ten customers by revenue.",
    "Explain locality sensitive hashing in two paragraphs.",
    "Draft a polite email declining a meeting.",
]


def generate_prompts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        template = rng.choice(TEMPLATES)
        system = template.format(
            name=rng.choice(NAMES),
            role=rng.choice(ROLES),
            org=rng.choice(ORGS),
        )
        user = rng.choice(USER_TURNS)
        out.append(system + "\n\nUser: " + user)
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


def evaluate_jaccardserve(prompts, tokenizer, banding, threshold, k):
    gw = JaccardServeGateway(
        tokenizer=tokenizer, shingle_width=k,
        banding=banding, jaccard_threshold=threshold,
    )
    for p in prompts:
        plan, ids = gw.match(p)
        gw.register(plan.request_id, ids, worker_id="w0")
    s = gw.stats.summary()
    return JSResult(
        hit_rate=s["hit_rate"], avg_overhead_ms=s["avg_overhead_ms"],
        avg_candidates=s["avg_candidates"], requests_total=s["requests_total"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-prompts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    prompts = generate_prompts(args.num_prompts, seed=args.seed)
    print(f"Generated {len(prompts)} templated-chat prompts.")
    tokenizer = make_tokenizer()

    print("\n[1/3] vLLM APC simulator...")
    apc = evaluate_apc(prompts, tokenizer, block_size=16)
    print(f"  hit_rate={apc.hit_rate:.4f}  overhead={apc.avg_overhead_ms:.3f}ms")

    print("\n[2/3] SemShareKV simulator...")
    embedder = SyntheticTokenEmbedding(dim=768)
    ssr_configs = [
        ("low",    SimHashConfig(num_planes=64,  num_tables=10), 0.80),
        ("medium", SimHashConfig(num_planes=128, num_tables=20), 0.85),
        ("high",   SimHashConfig(num_planes=128, num_tables=40), 0.90),
    ]
    ssr_results = []
    for name, cfg, thr in ssr_configs:
        r = evaluate_semsharekv_simulated(prompts, tokenizer, embedder, threshold=thr, config=cfg)
        ssr_results.append((name, cfg, thr, r))
        print(f"  [{name}] hit_rate={r.hit_rate:.4f}  overhead={r.avg_overhead_ms:.3f}ms")

    print("\n[3/3] JaccardServe sweep...")
    js_configs = [
        ("balanced",    BandingConfig(num_bands=20, rows_per_band=4), 0.5, 5),
        ("high-prec",   BandingConfig(num_bands=20, rows_per_band=8), 0.7, 5),
        ("high-recall", BandingConfig(num_bands=50, rows_per_band=4), 0.4, 5),
    ]
    js_results = []
    for name, b, thr, k in js_configs:
        r = evaluate_jaccardserve(prompts, tokenizer, b, thr, k)
        js_results.append((name, b, thr, k, r))
        print(f"  [{name}] b={b.num_bands} r={b.rows_per_band} thr={thr}: "
              f"hit_rate={r.hit_rate:.4f}  overhead={r.avg_overhead_ms:.3f}ms")

    out = os.path.join(args.results_dir, "templated_chat_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "config", "hit_rate", "avg_overhead_ms", "avg_candidates", "n"])
        w.writerow(["vLLM_APC", "block=16", f"{apc.hit_rate:.6f}",
                    f"{apc.avg_overhead_ms:.4f}", "-", apc.requests_total])
        for name, cfg, thr, r in ssr_results:
            w.writerow([
                "SemShareKV_sim",
                f"{name}_planes={cfg.num_planes}_tables={cfg.num_tables}_thr={thr}",
                f"{r.hit_rate:.6f}", f"{r.avg_overhead_ms:.4f}",
                f"{r.avg_candidates:.2f}", r.requests_total,
            ])
        for name, b, thr, k, r in js_results:
            w.writerow([
                "JaccardServe",
                f"{name}_b={b.num_bands}_r={b.rows_per_band}_thr={thr}_k={k}",
                f"{r.hit_rate:.6f}", f"{r.avg_overhead_ms:.4f}",
                f"{r.avg_candidates:.2f}", r.requests_total,
            ])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
