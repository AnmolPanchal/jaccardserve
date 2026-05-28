#!/usr/bin/env bash
# Reproduce all results in the paper. CPU-only; no GPU required.
# Run from the repository root.

set -e

mkdir -p results figures

echo "==============================================================="
echo "JaccardServe: Reproducing all CPU-only results"
echo "==============================================================="
echo

echo "[1/5] Templated-chat benchmark..."
python benchmarks/templated_chat.py -n 500 --seed 0

echo
echo "[2/5] Multi-doc summarization benchmark..."
python benchmarks/multi_doc_summ.py \
    --num-groups 20 --docs-per-group 8 \
    --queries-per-group 15 --docs-per-query 4 --seed 42

echo
echo "[3/5] Oracle precision/recall validation..."
python quality/oracle_validation.py --seed 42

echo
echo "[4/5] Unit tests (incl. empirical S-curve)..."
python -m unittest discover -s tests -v 2>&1 | tail -20

echo
echo "[5/5] Generating figures..."
python benchmarks/make_figures.py --s-curve-trials 100

echo
echo "==============================================================="
echo "Reproduction complete."
echo
echo "Results CSVs in:   results/"
echo "Figures (PNG) in:  figures/"
echo "==============================================================="
ls -la results/ figures/
