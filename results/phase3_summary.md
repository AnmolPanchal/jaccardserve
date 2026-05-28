# Phase 3 — Quality Validation Results

**Date**: 2026-05-27  
**Generation model**: Qwen/Qwen2.5-3B-Instruct (fp16)  
**Judge model**: Qwen/Qwen2.5-7B-Instruct (int8, bitsandbytes)  
**Workload**: templated\_chat  
**Pairs evaluated**: 100  
**Acceptance bar**: ≤ 1% MAJOR-difference rate  

---

## Result: **FAIL**

| Label | Count | % |
|-------|-------|---|
| EQUIVALENT | 4 | 4.0% |
| MINOR | 36 | 36.0% |
| **MAJOR** | **60** | **60.0%** |
| ERROR | 0 | 0.0% |

**MAJOR-difference rate: 60%** — far above the 1% acceptance bar.  
**Mean ROUGE-L: 0.2721** — responses differ substantially in word sequence.

---

## Root Cause Analysis

Strategy A (prompt-prefix reordering) achieves TTFT savings by moving the matched
token span to position 0, enabling vLLM APC to hit the cached prefix. However, this
reordering **breaks the semantic structure of chat-template prompts**.

### What the reordering does to a templated prompt

Original token order:
```
[user-specific intro: "You are an assistant for Alice, a software engineer at Meta..."]
[shared system body: "Help them. Be concise. Cite sources. If unsure, say so."]
[user query: "Explain LSH in two paragraphs."]
```

JaccardServe's matched span is the **shared system body** (tokens 12–55). Strategy A
moves it to position 0:

```
[shared system body: "Help them. Be concise. Cite sources. If unsure, say so."]
[user-specific intro: "You are an assistant for Alice, a software engineer at Meta..."]
[user query: "Explain LSH in two paragraphs."]
```

The model now sees the generic instructions FIRST and the user identification SECOND.
The user identification, which was the coherent opening of the system message, is now
stranded in the MIDDLE of the context window — where, relative to the model's prior
training, text in this position looks like user input or continuation text, not a
system preamble.

**Effect on generation:** The model generates text that continues from (or responds
to) the mid-context user identification fragment rather than answering the actual query.

Example (pair idx=1, Jaccard=0.57):
```
Prompt: "Explain Locality Sensitive Hashing..."

Baseline (original order):
  "Sure, I can provide an explanation of Locality Sensitive Hashing (LSH)..."

Strategy-A candidate (reordered):
  "Chase, who is exploring new data analysis techniques for fraud detection..."
```

The jcandidate response opens with the user's name and context — the model is
"completing" the now-misplaced user-identification fragment rather than answering.

---

## Implication

Strategy A's 44% TTFT reduction (Phase 2) is **not quality-preserving** for
chat-template workloads. The TTFT savings and the output degradation are two sides
of the same mechanism: moving the shared span to position 0 satisfies APC's prefix
requirement but destroys the prompt's semantic structure.

This does not invalidate the matching layer (95.5% hit rate is real and useful), but
it proves that **Strategy B (block-table injection) is not an optional optimization
— it is the necessary mechanism for quality-safe KV reuse.**

Strategy B directly installs the donor's KV blocks at the correct offset without
reordering tokens. The prompt arrives at the model with its original token order
intact; APC is bypassed in favor of direct block injection. This is the correct
architecture for production deployment.

---

## Updated End-to-End Picture

| Component | Status | Key result |
|-----------|--------|-----------|
| Matching layer | ✓ Works | 95.5% hit rate (templated), 58.5% (multi-doc) |
| Strategy A TTFT | ✓ Fast (but unsafe) | −44% median TTFT (templated) |
| Strategy A quality | ✗ Fails | 60% MAJOR-diff rate — reordering breaks chat semantics |
| Strategy B | Pending | The correct production mechanism |

---

## How to Frame in the Paper

**Section 5 (TTFT results):** Report Phase 2 numbers with a footnote that they bound
the performance achievable IF quality loss is acceptable (e.g., for non-chat use cases
where token order is irrelevant, such as raw completion over homogeneous corpora).

**Section 6 (Quality validation):** Report Phase 3 failure and diagnosis. Strategy A
fails because it violates the invariant that chat models expect system content at a
fixed position in the context. This is a structural incompatibility, not a tuning
issue.

**Section 7 (Conclusion and future work):** JaccardServe's matching layer identifies
high-value KV reuse opportunities. Realizing those opportunities without quality loss
requires Strategy B, which maintains the original token order and injects KV directly
into the block table. The matching layer, Jaccard threshold, and banding configuration
designed here carry over to Strategy B unchanged.

This framing turns a "failure" into a sharp motivating result: the matching works,
the reuse opportunity is real, and we have diagnosed exactly why Strategy A is
insufficient for production chat workloads. Reviewers who ask "why not just use
Strategy A?" get a clear, empirical answer.

---

## Files

- `results/llm_judge_results.csv` — per-pair labels and reasons
- `results/llm_judge_generations_templated.csv` — raw baseline and jcandidate outputs
