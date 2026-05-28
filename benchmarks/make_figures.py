"""
Generate all figures used in the paper.

Reads the results/*.csv files produced by the benchmarks and the
quality validation. Saves PNG figures to figures/.

Run order:
  1. benchmarks/templated_chat.py
  2. benchmarks/multi_doc_summ.py
  3. quality/oracle_validation.py
  4. benchmarks/make_figures.py   <- this file
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend for servers / CI
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaccardserve import (
    BandedLSH,
    BandingConfig,
    LSHEntry,
    MinHasher,
    MinHashConfig,
)


# Consistent palette across figures.
COLORS = {
    "vLLM_APC": "#888888",
    "SemShareKV_sim": "#d8773e",
    "JaccardServe": "#3a6bb5",
    "theoretical": "#222222",
    "empirical": "#d8773e",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------------------
# Figure 1: Empirical vs theoretical S-curve
# ---------------------------------------------------------------------------

def figure_s_curve(out_path: str, trials_per_point: int = 100):
    """
    Sweep true Jaccard from 0.1 to 0.95, measure empirical collision
    rate of banded LSH, compare to P(s; b, r) = 1 - (1 - s^r)^b.
    """
    cfg = BandingConfig(num_bands=20, rows_per_band=4)
    mh = MinHasher(MinHashConfig(num_perms=cfg.num_perms, seed=11))
    rng = random.Random(0)

    donor = set(rng.randint(0, 1_000_000) for _ in range(2000))
    idx = BandedLSH(cfg)
    idx.insert(LSHEntry(
        request_id="donor", worker_id=None,
        signature=mh.signature(donor), shingle_set=donor, num_tokens=2000,
    ))

    s_grid = np.linspace(0.1, 0.95, 18)
    empirical = []
    std_errs = []
    for s in s_grid:
        n = 2000
        k = int(2 * n * s / (1 + s))
        hits = 0
        for _ in range(trials_per_point):
            overlap = rng.sample(sorted(donor), k=k)
            extras = set(
                rng.randint(2_000_000, 3_000_000) for _ in range(n - k)
            )
            target = set(overlap) | extras
            if "donor" in idx.query_candidates(mh.signature(target)):
                hits += 1
        p = hits / trials_per_point
        empirical.append(p)
        std_errs.append(np.sqrt(p * (1 - p) / trials_per_point))

    theoretical = 1.0 - (1.0 - s_grid ** cfg.rows_per_band) ** cfg.num_bands

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(s_grid, theoretical, "-", color=COLORS["theoretical"], linewidth=2,
            label=r"Theoretical: $1-(1-s^r)^b$")
    ax.errorbar(s_grid, empirical, yerr=std_errs, fmt="o",
                color=COLORS["empirical"], capsize=3, markersize=5,
                label=f"Empirical ({trials_per_point} trials/point)")
    ax.set_xlabel("True Jaccard similarity $s$")
    ax.set_ylabel("Candidate-collision probability $P(s; b, r)$")
    ax.set_title(f"Banded-LSH S-curve at b={cfg.num_bands}, r={cfg.rows_per_band}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return s_grid, empirical, theoretical


# ---------------------------------------------------------------------------
# Figure 2: Hit-rate comparison across methods
# ---------------------------------------------------------------------------

def figure_hit_rate_comparison(csv_path: str, out_path: str):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    labels = []
    rates = []
    colors = []
    for r in rows:
        labels.append(f"{r['method']}\n{r['config']}")
        rates.append(float(r["hit_rate"]))
        colors.append(COLORS.get(r["method"], "#bbbbbb"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(labels)), rates, color=colors, edgecolor="white")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, max(rates) * 1.15 + 0.05)
    ax.set_title("Cross-request match rate on multi-doc summarization (n=320)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Precision-Recall scatter
# ---------------------------------------------------------------------------

def figure_precision_recall(csv_path: str, out_path: str):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # F1 contours first (background).
    rec_grid = np.linspace(0.01, 1.0, 200)
    for f1 in (0.2, 0.4, 0.6, 0.8):
        p_curve = f1 * rec_grid / np.where(2 * rec_grid - f1 > 0,
                                            2 * rec_grid - f1, np.nan)
        mask = (p_curve > 0) & (p_curve <= 1.0)
        ax.plot(rec_grid[mask], p_curve[mask], "--", color="#cccccc",
                linewidth=0.8, alpha=0.7, zorder=1)
        idx_label = np.argmin(np.abs(p_curve - 0.5))
        if mask[idx_label]:
            ax.text(rec_grid[idx_label] + 0.01, 0.51, f"F1={f1}",
                    fontsize=7, color="#888888", ha="left", rotation=-30)

    # Hand-tuned annotation offsets to avoid overlap.
    offsets = {
        ("vLLM_APC", "block=16"): (12, -8),
        ("SemShareKV_sim", "medium"): (12, 8),
        ("JaccardServe", "balanced"): (-95, -20),
        ("JaccardServe", "high-prec"): (-95, 12),
        ("JaccardServe", "high-recall"): (12, -4),
    }
    for r in rows:
        method = r["method"]
        color = COLORS.get(method, "#bbbbbb")
        prec = float(r["precision"])
        rec = float(r["recall"])
        config = r["config"]
        if method == "JaccardServe":
            short = "balanced" if "balanced" in config else (
                "high-prec" if "high-prec" in config else (
                "high-recall" if "high-recall" in config else config[:15]
            ))
            label = f"JaccardServe ({short})"
        elif method == "SemShareKV_sim":
            label = "SemShareKV-sim (medium)"
            short = "medium"
        else:
            label = f"vLLM APC (block=16)"
            short = "block=16"
        ax.scatter(rec, prec, s=140, color=color, edgecolor="black",
                   linewidth=0.5, alpha=0.9, zorder=3)
        off = offsets.get((method, short), (8, 8))
        ax.annotate(label, (rec, prec),
                    xytext=off, textcoords="offset points",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#aaaaaa", alpha=0.9),
                    zorder=4)

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall (catching reusable matches when they exist)")
    ax.set_ylabel("Precision (catching only correct matches)")
    ax.set_title("Oracle precision/recall on multi-doc summarization")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Jaccard and span-coverage distributions for matches
# ---------------------------------------------------------------------------

def figure_match_distributions(per_prompt_csv: str, out_path: str):
    jaccards = []
    spans = []
    with open(per_prompt_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["hit"]):
                jaccards.append(float(row["jaccard"]))
                spans.append(float(row["span_coverage"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.hist(jaccards, bins=20, color=COLORS["JaccardServe"], edgecolor="white")
    ax1.axvline(np.mean(jaccards), color="black", linestyle="--",
                label=f"mean = {np.mean(jaccards):.3f}")
    ax1.set_xlabel("Exact Jaccard of declared match")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Match-Jaccard distribution (n={len(jaccards)} hits)")
    ax1.legend()

    ax2.hist(spans, bins=20, color=COLORS["JaccardServe"], edgecolor="white")
    ax2.axvline(np.mean(spans), color="black", linestyle="--",
                label=f"mean = {np.mean(spans):.3f}")
    ax2.set_xlabel("Fraction of target tokens covered by matched span")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Span coverage distribution (n={len(spans)} hits)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: (b, r) heatmap for hit rate at fixed threshold
# ---------------------------------------------------------------------------

def figure_br_heatmap(out_path: str, tokenizer_fn=None):
    from benchmarks.multi_doc_summ import generate_multi_doc_workload, make_tokenizer
    from jaccardserve import JaccardServeGateway
    tokenizer = tokenizer_fn if tokenizer_fn is not None else make_tokenizer()

    workload = generate_multi_doc_workload(
        num_groups=10, docs_per_group=8,
        queries_per_group=10, docs_per_query=4, seed=42,
    )
    prompts = [item["prompt"] for item in workload]

    bs = [10, 20, 40, 80]
    rs = [2, 4, 6, 8]
    hit_rates = np.zeros((len(bs), len(rs)))
    for i, b in enumerate(bs):
        for j, r in enumerate(rs):
            gw = JaccardServeGateway(
                tokenizer=tokenizer,
                shingle_width=5,
                banding=BandingConfig(num_bands=b, rows_per_band=r),
                jaccard_threshold=0.5,
            )
            for p in prompts:
                plan, ids = gw.match(p)
                gw.register(plan.request_id, ids)
            hit_rates[i, j] = gw.stats.summary().get("hit_rate", 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(hit_rates, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(rs)))
    ax.set_yticks(range(len(bs)))
    ax.set_xticklabels(rs)
    ax.set_yticklabels(bs)
    ax.set_xlabel("Rows per band (r)")
    ax.set_ylabel("Number of bands (b)")
    ax.set_title("Hit rate vs (b, r), threshold = 0.5, multi-doc workload")
    for i in range(len(bs)):
        for j in range(len(rs)):
            ax.text(j, i, f"{hit_rates[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if hit_rates[i, j] < 0.5 else "white",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="Hit rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--s-curve-trials", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    print("Figure 1: S-curve")
    figure_s_curve(
        os.path.join(args.figures_dir, "fig1_s_curve.png"),
        trials_per_point=args.s_curve_trials,
    )

    multi_doc_results = os.path.join(args.results_dir, "multi_doc_summ_results.csv")
    if os.path.exists(multi_doc_results):
        print("Figure 2: Hit-rate comparison")
        figure_hit_rate_comparison(
            multi_doc_results,
            os.path.join(args.figures_dir, "fig2_hit_rate_comparison.png"),
        )

    quality_csv = os.path.join(args.results_dir, "quality_oracle_precision_recall.csv")
    if os.path.exists(quality_csv):
        print("Figure 3: Precision-recall")
        figure_precision_recall(
            quality_csv,
            os.path.join(args.figures_dir, "fig3_precision_recall.png"),
        )

    per_prompt_csv = os.path.join(args.results_dir, "multi_doc_summ_per_prompt.csv")
    if os.path.exists(per_prompt_csv):
        print("Figure 4: Match distributions")
        figure_match_distributions(
            per_prompt_csv,
            os.path.join(args.figures_dir, "fig4_match_distributions.png"),
        )

    print("Figure 5: (b, r) heatmap")
    figure_br_heatmap(
        os.path.join(args.figures_dir, "fig5_br_heatmap.png"),
    )

    print(f"\nAll figures written to {args.figures_dir}/")


if __name__ == "__main__":
    main()
