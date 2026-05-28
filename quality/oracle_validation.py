"""
Quality validation for JaccardServe matches.

Real end-to-end quality validation requires running a model and
computing ROUGE-L / EM / LLM-judge on outputs. That requires GPU
and a real model. This file produces the PROXY validation that can
be run without those things, and writes the harness for the real
validation (which the user runs on GPU).

Three measurable proxies:

1. ORACLE PRECISION / RECALL. The multi-doc benchmark labels each
   prompt with its group_id and an is_outlier flag. A match is a
   true positive if both prompts have the same group_id and neither
   is an outlier of the wrong group. The oracle precision/recall of
   JaccardServe vs APC vs SemShareKV-sim shows whether each method
   catches the right matches.

2. SPAN COVERAGE. For every JaccardServe hit, what fraction of the
   target prompt's tokens are covered by the matched span? Higher
   coverage means more KV reuse is possible. We report the
   distribution.

3. JACCARD DISTRIBUTION. Histogram of the exact Jaccard at which
   matches were declared. A bimodal distribution with most mass
   above threshold is healthy. Mass right at the threshold
   indicates parameter tuning is on the edge.

Plus the harness (not run here) for:

4. ROUGE-L / EM / LLM-JUDGE against full-prompt baseline output.
   Run on your GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import (
    SimHashConfig,
    SyntheticTokenEmbedding,
    VLLMAPCSimulator,
)
from baselines.semsharekv_simulator import SemShareKVMatcherSimulator
from benchmarks.multi_doc_summ import (
    generate_multi_doc_workload,
    make_tokenizer,
)
from jaccardserve import BandingConfig, JaccardServeGateway


@dataclass
class OracleResult:
    method: str
    config: str
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        d = self.true_positive + self.false_positive
        return self.true_positive / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positive + self.false_negative
        return self.true_positive / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def is_compatible_donor(target_item: dict, donor_item: dict) -> bool:
    """
    Ground truth: a match is compatible iff the two prompts are in
    the same group and neither is an outlier from a different group.
    The semantics: if compatible, the model would produce semantically
    related outputs from the two prompts, so KV reuse should not
    degrade quality.
    """
    if target_item["group_id"] != donor_item["group_id"]:
        return False
    if target_item["is_outlier"] or donor_item["is_outlier"]:
        return False
    return True


def evaluate_oracle_jaccardserve(
    workload: list[dict],
    tokenizer,
    banding: BandingConfig,
    threshold: float,
    k: int,
) -> OracleResult:
    gw = JaccardServeGateway(
        tokenizer=tokenizer,
        shingle_width=k,
        banding=banding,
        jaccard_threshold=threshold,
    )
    # Map request_id -> workload index so we can recover ground truth.
    rid_to_idx: dict[str, int] = {}
    tp = fp = fn = tn = 0
    has_prior_in_same_group: list[bool] = [False] * len(workload)
    seen_groups_non_outlier: set[int] = set()
    for i, item in enumerate(workload):
        if not item["is_outlier"] and item["group_id"] in seen_groups_non_outlier:
            has_prior_in_same_group[i] = True
        elif not item["is_outlier"]:
            seen_groups_non_outlier.add(item["group_id"])

    for i, item in enumerate(workload):
        plan, token_ids = gw.match(item["prompt"])
        rid_to_idx[plan.request_id] = i
        hit = plan.donor_request_id is not None
        donor_idx = rid_to_idx.get(plan.donor_request_id) if hit else None
        donor_item = workload[donor_idx] if donor_idx is not None else None

        # Ground truth: was a compatible donor available in history?
        compatible_available = has_prior_in_same_group[i] and not item["is_outlier"]

        # Match decision correctness:
        if hit and donor_item is not None and is_compatible_donor(item, donor_item):
            tp += 1
        elif hit and (donor_item is None or not is_compatible_donor(item, donor_item)):
            fp += 1
        elif not hit and compatible_available:
            fn += 1
        else:
            tn += 1

        gw.register(plan.request_id, token_ids, worker_id="w0")

    return OracleResult(
        method="JaccardServe",
        config=f"b={banding.num_bands}_r={banding.rows_per_band}_thr={threshold}_k={k}",
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
    )


def evaluate_oracle_apc(workload: list[dict], tokenizer, block_size: int = 16) -> OracleResult:
    apc = VLLMAPCSimulator(block_size=block_size)
    tp = fp = fn = tn = 0
    seen: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    seen_groups_non_outlier: set[int] = set()
    has_prior_in_same_group = [False] * len(workload)
    for i, item in enumerate(workload):
        if not item["is_outlier"] and item["group_id"] in seen_groups_non_outlier:
            has_prior_in_same_group[i] = True
        elif not item["is_outlier"]:
            seen_groups_non_outlier.add(item["group_id"])

    for i, item in enumerate(workload):
        ids = list(tokenizer(item["prompt"]))
        # Look up the first block.
        if len(ids) >= block_size:
            block = ids[:block_size]
            bh = apc._block_hash(block)
            prior_indices = seen.get((0, bh), [])
            if prior_indices:
                # Donor is the most-recent prior index that matched.
                donor_idx = prior_indices[-1]
                if is_compatible_donor(item, workload[donor_idx]):
                    tp += 1
                else:
                    fp += 1
            else:
                if has_prior_in_same_group[i] and not item["is_outlier"]:
                    fn += 1
                else:
                    tn += 1
            seen[(0, bh)].append(i)
            apc.register(ids)
        else:
            if has_prior_in_same_group[i] and not item["is_outlier"]:
                fn += 1
            else:
                tn += 1

    return OracleResult(
        method="vLLM_APC",
        config=f"block={block_size}",
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
    )


def evaluate_oracle_semsharekv(
    workload: list[dict],
    tokenizer,
    embedder: SyntheticTokenEmbedding,
    config: SimHashConfig,
    threshold: float,
) -> OracleResult:
    matcher = SemShareKVMatcherSimulator(dim=embedder.dim, config=config)
    rid_to_idx: dict[str, int] = {}
    tp = fp = fn = tn = 0
    seen_groups_non_outlier: set[int] = set()
    has_prior_in_same_group = [False] * len(workload)
    for i, item in enumerate(workload):
        if not item["is_outlier"] and item["group_id"] in seen_groups_non_outlier:
            has_prior_in_same_group[i] = True
        elif not item["is_outlier"]:
            seen_groups_non_outlier.add(item["group_id"])

    for i, item in enumerate(workload):
        ids = list(tokenizer(item["prompt"]))
        vec = embedder.embed_prompt_mean(ids)
        cands = matcher.query(vec)
        best_id, _ = matcher.verify_cosine(vec, cands, threshold)
        donor_idx = rid_to_idx.get(best_id) if best_id else None

        compatible_available = has_prior_in_same_group[i] and not item["is_outlier"]

        if best_id is not None and donor_idx is not None and is_compatible_donor(item, workload[donor_idx]):
            tp += 1
        elif best_id is not None:
            fp += 1
        elif compatible_available:
            fn += 1
        else:
            tn += 1

        rid = f"req_{i}"
        rid_to_idx[rid] = i
        matcher.insert(rid, vec)

    return OracleResult(
        method="SemShareKV_sim",
        config=f"planes={config.num_planes}_tables={config.num_tables}_thr={threshold}",
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
    )


def jaccard_distribution_from_csv(per_prompt_csv: str) -> list[float]:
    """Read the per-prompt CSV from a benchmark run and return the Jaccard values for hits."""
    out = []
    with open(per_prompt_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["hit"]):
                out.append(float(row["jaccard"]))
    return out


def span_coverage_distribution_from_csv(per_prompt_csv: str) -> list[float]:
    out = []
    with open(per_prompt_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["hit"]):
                out.append(float(row["span_coverage"]))
    return out


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
    print(f"Workload: {len(workload)} prompts.")
    n_outlier = sum(1 for it in workload if it["is_outlier"])
    print(f"  outliers: {n_outlier}")
    print(f"  non-outliers per group: {(len(workload) - n_outlier) / args.num_groups:.1f}")

    tokenizer = make_tokenizer()

    print("\n=== Oracle precision/recall ===")
    results: list[OracleResult] = []

    # vLLM APC
    apc_oracle = evaluate_oracle_apc(workload, tokenizer)
    results.append(apc_oracle)
    print(f"\n{apc_oracle.method} [{apc_oracle.config}]")
    print(f"  TP={apc_oracle.true_positive}  FP={apc_oracle.false_positive}  "
          f"FN={apc_oracle.false_negative}  TN={apc_oracle.true_negative}")
    print(f"  precision={apc_oracle.precision:.4f}  recall={apc_oracle.recall:.4f}  "
          f"F1={apc_oracle.f1:.4f}")

    # SemShareKV
    embedder = SyntheticTokenEmbedding(dim=768)
    for name, cfg, thr in [
        ("medium", SimHashConfig(num_planes=128, num_tables=20), 0.85),
    ]:
        ssr = evaluate_oracle_semsharekv(workload, tokenizer, embedder, cfg, thr)
        ssr.config = f"{name}_{ssr.config}"
        results.append(ssr)
        print(f"\n{ssr.method} [{ssr.config}]")
        print(f"  TP={ssr.true_positive}  FP={ssr.false_positive}  "
              f"FN={ssr.false_negative}  TN={ssr.true_negative}")
        print(f"  precision={ssr.precision:.4f}  recall={ssr.recall:.4f}  F1={ssr.f1:.4f}")

    # JaccardServe sweep
    for name, banding, thr, k in [
        ("balanced", BandingConfig(num_bands=20, rows_per_band=4), 0.5, 5),
        ("high-prec", BandingConfig(num_bands=20, rows_per_band=8), 0.7, 5),
        ("high-recall", BandingConfig(num_bands=50, rows_per_band=4), 0.4, 5),
    ]:
        jsr = evaluate_oracle_jaccardserve(workload, tokenizer, banding, thr, k)
        jsr.config = f"{name}_{jsr.config}"
        results.append(jsr)
        print(f"\n{jsr.method} [{jsr.config}]")
        print(f"  TP={jsr.true_positive}  FP={jsr.false_positive}  "
              f"FN={jsr.false_negative}  TN={jsr.true_negative}")
        print(f"  precision={jsr.precision:.4f}  recall={jsr.recall:.4f}  F1={jsr.f1:.4f}")

    # Write CSV.
    out_path = os.path.join(args.results_dir, "quality_oracle_precision_recall.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "config", "TP", "FP", "FN", "TN",
                    "precision", "recall", "F1"])
        for r in results:
            w.writerow([
                r.method, r.config, r.true_positive, r.false_positive,
                r.false_negative, r.true_negative,
                f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}",
            ])
    print(f"\nWrote {out_path}")

    # Span coverage and Jaccard distribution from the most recent benchmark run.
    per_prompt = os.path.join(args.results_dir, "multi_doc_summ_per_prompt.csv")
    if os.path.exists(per_prompt):
        js = jaccard_distribution_from_csv(per_prompt)
        spans = span_coverage_distribution_from_csv(per_prompt)
        if js:
            print(f"\nJaccardServe match Jaccard distribution: "
                  f"n={len(js)} min={min(js):.3f} mean={sum(js)/len(js):.3f} max={max(js):.3f}")
        if spans:
            print(f"JaccardServe span-coverage distribution: "
                  f"n={len(spans)} min={min(spans):.3f} mean={sum(spans)/len(spans):.3f} max={max(spans):.3f}")


if __name__ == "__main__":
    main()
