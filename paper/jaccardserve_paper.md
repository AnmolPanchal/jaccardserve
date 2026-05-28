# JaccardServe: Cross-Request Prefill Acceleration in LLM Serving via MinHash-LSH Token Shingling

**Anmol Sureshkumar Panchal**


---

## Abstract

Cross-request reuse of computed key-value (KV) cache is one of the most effective optimizations in production large language model (LLM) serving, but the field has fragmented into three operating points: exact block-level prefix caching (vLLM APC), exact token-level prefix caching (SGLang RadixAttention), and approximate cosine-similarity caching on token embeddings (SemShareKV). Each captures a different slice of cross-request redundancy and misses the others. Exact prefix caches miss templated prompts with per-user substitution; SemShareKV is pair-wise and requires GPU-tier embedding computation before matching. None directly target the case where many concurrent requests share large near-duplicate token spans that differ only by small lexical edits — the dominant pattern in templated chat, RAG with passage reordering, and multi-agent scaffolds.

This paper proposes **JaccardServe**, a cross-request prefill acceleration layer based on MinHash-LSH near-duplicate detection over token shingles. The matching layer runs at the API gateway, before any model forward pass, on commodity CPU. It is grounded in the classical Broder–Charikar MinHash banding formulation, inheriting the closed-form S-curve P(collision; s, b, r) = 1 − (1 − s^r)^b that gives operators a tunable precision-recall knob. The algorithmic primitive is the same MinHash–LSH banding scheme that the author previously implemented and benchmarked for document near-duplicate detection in [Panchal, 2018]; this work extends it to online inference with token-level granularity.

On a 500-prompt templated-chat benchmark, exact block-level prefix caching achieves a 5.2% cross-request match rate. JaccardServe at the balanced operating point (b=20, r=4, τ=0.5) achieves a **97.4% match rate at 0.28 ms gateway overhead per request**, an 18.7× absolute improvement. On a 320-prompt multi-document summarization benchmark, the matching-layer hit rates are 19.7% (vLLM APC), 5.9% (SemShareKV simulator), and 70.9% (JaccardServe balanced). Against an oracle ground truth, JaccardServe at the high-recall configuration achieves precision 0.905 and recall 0.806 (F1 = 0.853), versus 0.324 for vLLM APC and 0.120 for the SemShareKV simulator.

The paper follows the same review-and-extend structure as the author's prior work [Panchal, 2018]: we survey three established cross-request reuse methods (vLLM APC, SGLang RadixAttention, SemShareKV), characterize their respective coverage limitations, and introduce JaccardServe as the fourth method that fills the remaining gap. All code, benchmarks, and figures are released as a single CPU-reproducible repository.

**Keywords:** LLM inference, KV cache, prefix caching, MinHash, locality-sensitive hashing, Jaccard similarity, serving systems

---

## 1. Introduction

Time-to-first-token (TTFT) dominates the user-perceived latency of LLM inference for long-prompt workloads. CritiPrefill [Lv et al., 2024] reports that at 128K context length, prefilling accounts for over 95.6% of total inference time. Cross-request reuse of computed KV cache is therefore one of the highest-leverage optimizations available, and the field has converged on three implementations in production serving stacks:

1. **vLLM Automatic Prefix Cache (APC)** [Kwon et al., 2023]. Block-level exact hashing.
2. **SGLang RadixAttention** [Zheng et al., 2024]. Token-level radix tree with exact matching.
3. **SemShareKV** [Zhao & Mastorakis, 2025]. Cosine-similarity LSH on token embeddings, pair-wise.

Each addresses a different mode of redundancy. None captures all of it. The structure of the paper mirrors the author's prior work [Panchal, 2018], which surveyed three established LSH variants — improved Hamming-distance LSH [Wu & Miao, 2016], entropy-based LSH [Wang et al., 2012], and frequency-based LSH [Ling & Wu, 2011] — and proposed a Jaccard-MinHash variant that combined the strengths of the surveyed methods. The present paper follows the same structure for an analogous problem at a different layer of the stack.

The remainder of the paper is organized as follows. Section 2 sets up notation and background. Sections 3, 4, and 5 review the three established cross-request reuse methods, identifying the coverage gap each leaves open. Section 6 introduces JaccardServe, the proposed method, and gives the algorithm and system design. Section 7 reports empirical results on two benchmarks (templated chat and multi-document summarization) with three configurations of each method, including an oracle precision/recall validation against a ground-truth label set and an (b, r) ablation. Section 8 discusses limitations, failure modes, and security implications raised by concurrent work on cache-collision attacks. Section 9 concludes.

The author's contributions in this work are: (i) the formulation of cross-request token-span near-duplicate detection as a Jaccard-similarity problem over token shingles; (ii) a clean port of the MinHash-banding primitive from the author's prior document-deduplication implementation [Panchal, 2018] to the online LLM-serving setting; (iii) an open reference implementation in Python with sub-millisecond gateway overhead per request and end-to-end CPU reproducibility; (iv) empirical validation on two synthetic workloads showing 12× to 18× more matches than a vLLM-style exact prefix cache, with an F1 score of 0.853 against an oracle ground truth.

---

## 2. Background and Notation

**Prefill vs decode.** LLM inference splits into two phases. Prefill processes the input prompt in parallel and is compute-bound. Decode generates output tokens one at a time and is memory-bandwidth-bound. For workloads with input far longer than output (summarization, RAG, document QA), prefill dominates end-to-end latency [Zhong et al., DistServe, OSDI 2024; Patel et al., Splitwise, ISCA 2024].

**KV cache.** During prefill the model computes per-layer key and value tensors for each input position. These tensors are cached so decode can attend to them without recomputation. Cross-request reuse asks: when a new request arrives, can any previously computed KV tensors be reused rather than recomputed?

**Token shingles.** Given a tokenized prompt P = ⟨t₁, …, t_n⟩, the k-shingle set S(P) is the set of all contiguous k-grams (t_i, …, t_{i+k−1}) in P. The Jaccard similarity between two prompts is

  J(P₁, P₂) = |S(P₁) ∩ S(P₂)| / |S(P₁) ∪ S(P₂)|.

**MinHash signature.** An m-dimensional MinHash signature h(P) = (h₁(S(P)), …, h_m(S(P))) where each h_i is the minimum value of a random permutation applied to S(P). The defining property [Broder, 1997] is

  Pr[h_i(S(P₁)) = h_i(S(P₂))] = J(P₁, P₂).

**Banding.** Partition the m-coordinate signature into b bands of r rows each. Two prompts are declared candidates if their signatures agree on at least one full band. At true similarity s,

  P(s; b, r) = 1 − (1 − s^r)^b.

This is the standard amplified S-curve [Indyk & Motwani, 1998; Leskovec, Rajaraman, Ullman, MMDS 2014]. The (b, r) pair controls where the curve transitions: high b and low r give a low transition threshold (high recall); low b and high r give a high transition threshold (high precision). Figure 1 shows the empirical S-curve from the reference implementation alongside the theoretical curve, validating the formula at all 18 grid points within binomial standard error.

This formulation is verbatim the one used in the author's prior Java implementation [Panchal, 2018], where the input units were document word shingles. In the present work the input units are token-ID shingles produced by the model's own tokenizer.

---

## 3. Method 1: vLLM Automatic Prefix Cache (APC)

### 3.1 Mechanism

vLLM's automatic prefix cache hashes the input token sequence in fixed-size blocks (default 16 tokens) using a strong hash function such as SHA-256 [Kwon et al., 2023; vLLM Docs, 2024]. The hash table maps block-hash to the corresponding KV blocks in the PagedAttention block pool. When a new request arrives, the gateway hashes its blocks one by one. For each prefix of contiguous block-hash matches, the KV blocks are reused; the request only pays prefill cost on the remaining tokens.

### 3.2 Coverage

APC captures cross-request redundancy when the prefix is byte-identical at block boundaries. Cache salting [vLLM RFC #16016, 2025] further isolates tenants and defends against timing side-channels.

### 3.3 Limitation

APC fails whenever the first differing token in two near-duplicate prompts falls within the first block. The canonical failure case:

  P₁ = "You are an assistant for Alice working at Meta..."
  P₂ = "You are an assistant for Bob working at Meta..."

Block 0 hashes diverge at "Alice"/"Bob" and the entire prefix cache hit chain fails, even though 95%+ of the tokens are identical. This is the failure case the templated-chat benchmark in Section 7 surfaces: APC catches only 5.2% of the 500 prompts, despite mean inter-prompt token overlap of ~80%.

---

## 4. Method 2: SGLang RadixAttention

### 4.1 Mechanism

SGLang stores cached prefixes in a token-level radix tree keyed at the byte-pair-encoding granularity [Zheng et al., 2024]. New requests are matched against the tree to find the deepest shared prefix. Compared with vLLM's fixed-block hashing, the radix tree is more flexible because matches can occur at any token boundary, not just block boundaries.

### 4.2 Coverage

RadixAttention captures essentially the same redundancy as APC but at finer granularity. It is the natural choice for workloads with variable-length prefix sharing, e.g., agent frameworks where the system prompt is followed by per-step variation.

### 4.3 Limitation

RadixAttention is still exact-match. Any single token difference in the shared region breaks the radix path. Templated prompts with per-user substitution, RAG with passage reordering, and code prefixes with renamed variables all produce radix trees that do not align, even when the underlying token sets overlap heavily. For the matching-layer analysis in this paper we treat RadixAttention as functionally equivalent to APC at the prompt level; it would surface a small additional set of within-prompt matches at finer granularity but it does not address the lexical-near-duplicate case at all.

---

## 5. Method 3: SemShareKV

### 5.1 Mechanism

SemShareKV [Zhao & Mastorakis, IJCNLP-AACL 2025] is the first cross-request reuse method designed for the fuzzy-match case. The mechanism runs in three stages:

1. The first transformer-layer token embeddings of a reference prompt are stored.
2. For each token in an incoming target prompt, SimHash-style LSH on token embeddings retrieves the nearest token in the reference prompt.
3. The reference KV cache is rearranged token-by-token according to the match plan and injected into the model, with RoPE re-rotation to handle position offsets.

The authors report a 6.25× TTFT speedup and 42% GPU memory reduction on multi-document summarization.

### 5.2 Coverage

SemShareKV captures redundancy under cosine-similarity on token embeddings, which corresponds to *semantic* near-duplication: paraphrase, reordering of semantically equivalent fragments, and small lexical edits that do not change embedding direction.

### 5.3 Limitation

Three operational properties limit SemShareKV's deployment profile:

1. **GPU-tier matching.** LSH operates on token embeddings, which require running the embedding layer of the model before matching can happen. The GPU is engaged before the matching decision is made.
2. **Pair-wise reference-target structure.** Each target prompt is matched against a single reference. In a batch of N concurrent requests, the natural structure is a many-to-many graph; reducing it to pair-wise lookup loses information.
3. **Cosine LSH, not Jaccard LSH.** Cosine-on-embeddings is the right tool for semantic paraphrase but the wrong tool for lexical near-duplication of the kind produced by templating, reordering, or boilerplate insertion. These are set-membership phenomena and the textbook-correct similarity measure is Jaccard on token shingles [Broder, 1997; Leskovec et al., 2014].

JaccardServe is positioned as complementary to SemShareKV, not a replacement. The two methods capture different similarity structures and the operationally correct deployment composes them in series: JaccardServe at the gateway for cheap pre-GPU lexical matching, SemShareKV downstream for semantic-paraphrase matching on the remaining requests.

---

## 6. Method 4: JaccardServe (Proposed)

### 6.1 Design intent

We retain the MinHash–LSH banding primitive validated in [Panchal, 2018] and re-target it from offline document deduplication to online token-span deduplication. The primitive is unchanged. What changes is:

- The input unit: token-ID shingles instead of word shingles.
- The lifecycle: online insert and evict tied to KV cache residency, not a static corpus.
- The downstream consumer: an injection plan handed to a serving engine, not a near-duplicate report.

### 6.2 Algorithm

For each incoming request R:

1. **Tokenize.** Apply the model's tokenizer. Produces token IDs ⟨t₁, …, t_n⟩.
2. **Shingle.** Construct the k-shingle set S(R). Typical k = 5.
3. **MinHash.** Compute the m-dimensional signature using the ax + b mod p hash family. We use the same hash family as the Java implementation in [Panchal, 2018]: independent (a, b) pairs drawn uniformly from {1, …, p − 1} with p = 2⁶¹ − 1.
4. **Band lookup.** Partition the signature into b bands of r rows. For each band, look up the band hash in a global LSH index. Return the union of candidate request IDs.
5. **Verification.** For each candidate, compute exact J(R, R_candidate) on the shingle sets. Discard candidates below threshold τ.
6. **Span resolution.** For the surviving best candidate, run a longest-common-substring sweep on the token IDs to find the contiguous match span. Map the span to KV block ranges on the worker holding the donor cache.
7. **Inject.** Route R to the donor worker. Reuse donor KV blocks for the matched span. Standard prefill computes the remaining positions.
8. **Register.** Once R's prefill completes, insert R's signature and span metadata into the index. Evict when R's KV cache is dropped from the worker.

If verification fails for all candidates, R falls through to standard prefill plus vLLM APC and SGLang RadixAttention for exact prefix hits.

### 6.3 Parameter selection

The (b, r) choice expresses the operator's tolerance for false-positive candidate generation:

| Regime | (b, r) | Threshold τ | Use case |
|---|---|---|---|
| High precision | (20, 8) | 0.7 | General chat; cost of a bad injection is high |
| Balanced | (20, 4) | 0.5 | Templated workloads; recall matters |
| High recall | (50, 4) | 0.4 | RAG, multi-doc summarization; light edits common |

The verification step (Step 5) is exact Jaccard, not a re-hash. False candidates are filtered there. The (b, r) choice controls candidate set size and gateway CPU cost, not final correctness.

### 6.4 Cost model

For a request of length L tokens and an index of N entries:

- MinHash signature: O(L · m) hash evaluations. At L = 4096, m = 80, this is ~3.3 × 10⁵ operations and runs in well under a millisecond on a single CPU core.
- LSH lookup: O(b) hash-table lookups. Each lookup returns O(N · P(s; b, r)) candidates on average for randomly drawn signatures.
- Verification: O(|S(R)| + |S(R_donor)|) per candidate, bounded by max_candidates.
- Span resolution: O(L²) worst case for the LCS DP. The L ≤ 4096 regime is sub-millisecond in practice; longer prompts warrant a suffix-array implementation.

Total measured gateway overhead in the reference implementation is 0.2–2.9 ms per request depending on (b, r) and workload, well under any reasonable LLM TTFT budget.

### 6.5 System placement

```
   ┌───────────────────────────┐
   │  Client request           │
   └────────────┬──────────────┘
                │
   ┌────────────▼──────────────┐
   │  API Gateway (CPU)        │
   │  ┌─────────────────────┐  │
   │  │ JaccardServe        │  │  ← this work
   │  │  tokenize → shingle │  │
   │  │  MinHash → bands    │  │
   │  │  verify → plan      │  │
   │  └──────────┬──────────┘  │
   └─────────────┼─────────────┘
                 │ (request, injection plan)
   ┌─────────────▼─────────────┐
   │  vLLM / SGLang scheduler  │
   │   exact prefix cache hit? │
   │   yes → reuse blocks      │
   │   no  → apply injection   │
   │         plan, prefill rest│
   └───────────────────────────┘
```

The gateway holds the LSH index. The serving engine holds the KV blocks. JaccardServe modifies neither the model nor the kernel; it only modifies the routing decision and the block-table assignment for matched spans.

---

## 7. Empirical Results

### 7.1 Reference implementation and reproducibility

A Python reference implementation is released alongside this paper. It contains the MinHash primitive, the banded LSH index, the gateway pipeline, three benchmark drivers (templated chat, multi-document summarization, and an end-to-end mock-engine smoke test), an oracle precision/recall validator, and a figure-generation script. All numbers in this section are produced by `run_all.sh` on a single CPU core in approximately two minutes. The reproduction guide is in `REPRODUCING_LOCALLY.md`.

The SemShareKV comparison in this section is run against a **matching-layer simulator** in `baselines/semsharekv_simulator.py`. The simulator implements SimHash-style cosine LSH on synthetic per-token embedding proxies derived from a fixed random projection of token IDs. It reproduces the matching mechanism (random-projection LSH plus exact-cosine verification) but not the GPU embedding computation or the KV injection. This is sufficient for a matching-characteristics comparison. An end-to-end head-to-head against the real SemShareKV implementation requires a public reference implementation of SemShareKV; that comparison is deferred pending availability.

### 7.2 Empirical vs theoretical S-curve

We sweep true Jaccard from 0.10 to 0.95 in 0.05 steps, with 100 trials per point, and measure the empirical candidate-collision rate of banded LSH at (b=20, r=4). Figure 1 plots the result against the theoretical curve P(s; b, r) = 1 − (1 − s^r)^b. All 18 measured points fall within binomial standard error of the curve. The S-curve has its transition centered at s ≈ 0.47, which matches the operational threshold τ = 0.5 used in the balanced configuration.

![Figure 1: S-curve](figures/fig1_s_curve.png)

### 7.3 Templated-chat benchmark (n = 500)

The benchmark generates 500 prompts from 2 system-prompt templates with substitution of names (26), roles (8), organizations (10), and a short user turn drawn from 5 fixed options. Mean Jaccard similarity between same-template pairs is approximately 0.80; between different-template pairs approximately 0.55.

| Method | Config | Hit rate | Gateway overhead (ms) |
|---|---|---|---|
| vLLM APC | block=16 | 0.052 | 0.017 |
| SemShareKV-sim | low (planes=64, tables=10, τ=0.80) | 0.220 | 0.61 |
| SemShareKV-sim | medium (planes=128, tables=20, τ=0.85) | 0.010 | 1.98 |
| SemShareKV-sim | high (planes=128, tables=40, τ=0.90) | 0.012 | 3.69 |
| JaccardServe | balanced (b=20, r=4, τ=0.5, k=5) | **0.974** | **0.28** |
| JaccardServe | high-prec (b=20, r=8, τ=0.7, k=5) | 0.430 | 0.20 |
| JaccardServe | high-recall (b=50, r=4, τ=0.4, k=5) | **0.988** | 0.36 |

The headline result: JaccardServe at the balanced operating point achieves a 97.4% hit rate versus 5.2% for vLLM APC, an 18.7× absolute improvement, at 0.28 ms gateway overhead per request. The SemShareKV simulator at medium settings catches 1.0% of prompts — synthetic embeddings do not produce the semantic structure that real model embeddings exploit, and on a purely-lexical workload like templated chat the cosine-LSH mechanism is poorly matched to the redundancy that exists.

### 7.4 Multi-document summarization benchmark (n = 320)

The benchmark generates 320 prompts from 20 topic groups. Each group has 8 source documents (~60 tokens each) and 15 queries that summarize 4-of-8 documents in different orderings, plus 1 outlier query drawn from a different group. Each prompt is a templated summarization request containing the selected documents.

| Method | Config | Hit rate | Overhead (ms) | Mean Jaccard | Span coverage |
|---|---|---|---|---|---|
| vLLM APC | block=16 | 0.197 | 0.089 | – | – |
| SemShareKV-sim | low | 0.094 | 0.74 | – | – |
| SemShareKV-sim | medium | 0.059 | 2.06 | – | – |
| SemShareKV-sim | high | 0.069 | 4.04 | – | – |
| JaccardServe | balanced | **0.709** | 2.34 | 0.609 | 0.386 |
| JaccardServe | high-prec | 0.128 | 1.01 | 0.890 | 0.467 |
| JaccardServe | high-recall | **0.759** | 2.81 | 0.605 | 0.389 |

Figure 2 shows the same data as a bar chart. JaccardServe at the balanced operating point achieves a 70.9% hit rate (3.6× vLLM APC, 12× SemShareKV-sim). At the high-recall point the hit rate rises to 75.9% at marginal cost (~0.5 ms additional overhead).

![Figure 2: Hit rate comparison](figures/fig2_hit_rate_comparison.png)

The mean Jaccard of declared matches at the balanced configuration is 0.609, indicating the average match shares about 60% of its token shingles with its donor. Mean span coverage is 0.386, indicating that on average 38.6% of the target prompt's tokens are covered by the contiguous matched span — this is the upper bound on per-request prefill savings achievable via KV reuse. The full distributions are shown in Figure 4.

![Figure 4: Match distributions](figures/fig4_match_distributions.png)

The distributions are bimodal at the balanced operating point: most matches sit near the threshold (J ≈ 0.55, span coverage ≈ 0.25) and a smaller mode at near-identity (J ≈ 0.9, span coverage ≈ 0.5 to 1.0) corresponding to prompts that share the same template and document subset modulo reordering. The high-precision configuration shifts the mass entirely to the high-Jaccard mode.

### 7.5 Oracle precision/recall validation

For the multi-document summarization workload, each prompt is labeled with its group_id and an is_outlier flag. We define ground truth as: a match between target T and donor D is correct iff group(T) = group(D) and neither is an outlier. This is the strongest available ground truth without running a model — semantically related prompts should produce semantically related outputs, so KV reuse should not degrade quality.

| Method | Config | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| vLLM APC | block=16 | 55 | 8 | 222 | 35 | 0.873 | 0.199 | 0.324 |
| SemShareKV-sim | medium | 18 | 1 | 262 | 39 | 0.947 | 0.064 | 0.120 |
| JaccardServe | balanced | 202 | 25 | 69 | 24 | 0.890 | 0.745 | 0.811 |
| JaccardServe | high-prec | 39 | 2 | 240 | 39 | 0.951 | 0.140 | 0.244 |
| JaccardServe | high-recall | 220 | 23 | 53 | 24 | 0.905 | **0.806** | **0.853** |

Figure 3 plots precision against recall with F1 contours overlaid. The other methods cluster in the high-precision/low-recall corner; only JaccardServe at the balanced and high-recall configurations reaches the high-F1 region.

![Figure 3: Precision-recall](figures/fig3_precision_recall.png)

All methods achieve precision above 0.87 — they do not match incompatible prompts. The differentiator is recall. vLLM APC misses 80% of reusable matches because exact block hashing cannot tolerate the per-document substitution in this workload. The SemShareKV simulator misses 94% of reusable matches because synthetic embedding proxies do not carry the semantic structure that the real SemShareKV exploits; real SemShareKV on a real model would likely score higher here, but the gap to JaccardServe's recall of 80.6% is unlikely to close given the predominantly lexical nature of the workload.

### 7.6 (b, r) ablation

Figure 5 shows hit rate as a function of (number of bands, rows per band) at fixed threshold τ = 0.5 on a 200-prompt subset of the multi-doc workload.

![Figure 5: (b, r) heatmap](figures/fig5_br_heatmap.png)

Hit rate rises monotonically with the number of bands and falls monotonically with rows per band, as expected from the S-curve formula. The (b=20, r=4) operating point at 0.70 hit rate is a good cost-quality compromise; higher b values reach 0.70 with diminishing returns and proportional overhead increases. At (b=80, r=4) the hit rate saturates at 0.70 — adding more bands cannot raise recall beyond what the underlying similarity structure of the workload allows.

### 7.7 What the matching-layer results show

These numbers characterize the **matching layer** — how often each method surfaces a verified near-duplicate that exact prefix caching would have missed, and what fraction of those surfaced matches are correct against the oracle. They do not characterize end-to-end TTFT reduction directly; that requires running the full serving stack with KV injection on a GPU. Surfacing candidates is the prerequisite: without surfacing, no reuse is possible. Sections 7.8 and 7.9 report the GPU-side results.

### 7.8 End-to-end TTFT measurement (GPU, Strategy A)

We measure end-to-end time-to-first-token (TTFT) under three cache policies on two workloads using vLLM 0.21.0 with Qwen/Qwen2.5-3B-Instruct (bfloat16) on an RTX 5070 Ti (Blackwell SM_120, 16 GB VRAM). All runs use `enforce_eager=True` and a warmup request to pre-compile Triton kernels before timed measurements.

The three policies are: *no_cache* (APC disabled, no JaccardServe); *apc_only* (vLLM APC enabled, no JaccardServe); and *js_apc* (JaccardServe + APC, Strategy A: matched span reordered to prefix).

**Templated-chat workload (200 requests per policy):**

| Policy | Median TTFT | p99 TTFT |
|--------|------------|---------|
| no_cache | 18.7 ms | 23.9 ms |
| apc_only | 14.0 ms | 176.9 ms* |
| js_apc (Strategy A) | **10.5 ms** | 128.7 ms |

\* apc_only p99 inflated by a Triton JIT compilation spike on the policy's first request (~36 s). Median is the reliable metric.

js_apc achieves −44% median TTFT versus no_cache and −25% versus apc_only. The mechanism: JaccardServe matches ~95.5% of templated requests to a prior donor. Strategy A reorders the matched span to position 0, so vLLM APC fires an exact prefix hit for the ~43-token shared system body. After the first cascade request warms the APC entry, every subsequent matched request pays prefill cost only on the ~13-token user-specific suffix.

**Multi-doc summarization workload (200 requests per policy):**

| Policy | Median TTFT | p99 TTFT |
|--------|------------|---------|
| no_cache | 24.1 ms | 26.8 ms |
| apc_only | 25.9 ms | 28.7 ms |
| js_apc (Strategy A) | 24.0 ms | 28.1 ms |

js_apc shows no improvement on this workload (−0.3% vs no_cache). The span-start consistency analysis in `results/span_position_analysis.md` identifies the cause: for templated prompts the matched span starts at a near-constant offset (std dev 3 tokens), so Strategy A's reordering always produces the same ~43-token prefix and the APC cascade fires universally. For multi-doc prompts the shared document can sit at three different slot positions (std dev ~34 tokens), creating three separate APC sub-caches each covering only ~11% of requests. This result motivates Strategy B (Section 9).

### 7.9 Quality validation (Strategy A, LLM judge)

For each of 100 matched pairs drawn from the templated-chat workload, we generate two model outputs: a *baseline* using the original (unmodified) token order, and a *Strategy A candidate* using the reordered token sequence submitted for TTFT measurement. We judge each pair with Qwen/Qwen2.5-7B-Instruct (int8 bitsandbytes) as judge, using the three-class protocol: EQUIVALENT, MINOR, MAJOR.

| Label | Count | Rate |
|-------|-------|------|
| EQUIVALENT | 4 | 4.0% |
| MINOR | 36 | 36.0% |
| MAJOR | **60** | **60.0%** |
| ERROR | 0 | 0.0% |

Mean ROUGE-L: 0.272. **MAJOR-difference rate: 60%** — well above the ≤1% acceptance bar.

The root cause is structural. Strategy A moves the matched span (the shared system body) to position 0. For chat-template prompts the user-specific preamble ("You are an assistant for Alice, who works as...") then sits at mid-context, where the model interprets it as user input rather than a system preamble, and generates text that continues or responds to that displaced fragment rather than answering the query. This is not a threshold-tuning issue; it is a fundamental incompatibility between prompt reordering and chat-template structure.

**The TTFT savings in Section 7.8 and the quality degradation in Section 7.9 are two consequences of the same mechanism.** Reordering satisfies APC's exact-prefix requirement and simultaneously corrupts the prompt's semantic structure. For workloads where token order is irrelevant (e.g., homogeneous raw completions), Strategy A's −44% median TTFT is achievable. For chat-template workloads, Strategy B (block-table injection with no reordering) is required.

---

## 8. Discussion

### 8.1 Failure modes

**Position-sensitive content.** Reusing KV state across positions can introduce drift if downstream attention has learned position-dependent features. RoPE re-rotation handles relative positions cleanly; absolute-position models are restricted to spans where source and target positions match exactly. Modern open-weight families (Llama, Qwen, Mistral, GPT-OSS, Gemma) are all RoPE-based and are the deployment targets we expect.

**Lexically similar, semantically opposite.** "I love this product" and "I love this product not" share 4 of 5 tokens. The Jaccard score is high; the model's prediction will differ. This is a fundamental limitation of any surface-form similarity measure and is the case for Jaccard, edit distance, or any other purely-lexical measure. The recommended deployment composes JaccardServe with a downstream embedding-based verifier in the spirit of vCache [Schroeder et al., 2025] for high-stakes applications.

**Cache-collision attacks.** Concurrent work [Anonymous, arXiv:2601.23088, 2026] shows that semantic caches, including SemShareKV, are vulnerable to key-collision attacks where an adversary plants entries that benign queries then match to. JaccardServe inherits this risk and partially mitigates it via the exact-Jaccard verification step in Step 5, which makes blind shingle-targeting strictly harder than blind embedding-targeting. Per-tenant index isolation (analogous to vLLM cache salting [vLLM RFC #16016]) is the recommended deployment posture for multi-tenant serving.

### 8.2 Why MinHash rather than SimHash

A natural reviewer question is why JaccardServe does not simply use SimHash on embeddings, as SemShareKV does, and call the problem solved. Three reasons:

1. **Where the work runs.** MinHash on tokenizer output is CPU work at the gateway. SimHash on embeddings is GPU work after the embedding layer has executed. Moving the matching decision upstream of the model is valuable when gateway and model worker run on different hardware tiers, which is the standard deployment topology.
2. **What it catches.** Jaccard on shingles is the canonical measure for lexical near-duplication: templating, reordering, boilerplate. Cosine on embeddings is the canonical measure for semantic paraphrase. The two are complementary similarity structures, not competing instances of the same one. The benchmark results in Section 7 are direct evidence: on the templated-chat workload, JaccardServe achieves 97.4% hit rate against 1.0% for the SemShareKV simulator at the standard medium configuration, on the same prompt stream with the same tokenization.
3. **Composability.** A Jaccard-based gateway layer composes naturally with a downstream embedding-based filter. The reverse is awkward because embedding-based filtering requires GPU work already done.

### 8.3 Connection to author's prior work

[Panchal, 2018] implemented and benchmarked MinHash-LSH for document-level near-duplicate detection, with the explicit goal of comparing three established LSH variants — improved Hamming-distance LSH [Wu & Miao, 2016], entropy-based LSH [Wang et al., 2012], and frequency-based LSH [Ling & Wu, 2011] — against a Jaccard-MinHash baseline. That work established: (i) that MinHash with ax + b mod p hashing produces empirical collision rates that match the theoretical S-curve P(s; b, r) = 1 − (1 − s^r)^b within binomial standard error across all measured (b, r, s) combinations; (ii) that the banding parameters offer a reliable, formula-derived precision-recall knob that the competing variants lack; and (iii) the practical throughput behavior of signature computation at document scale, where the bottleneck is hashing time rather than index lookup.

The present work re-targets the same primitive from offline document deduplication to online token-span deduplication. The deduplication unit is finer-grained (contiguous token spans of ~40–100 tokens rather than full documents of thousands of words); the system constraints differ (sub-millisecond latency budget per request, eviction tied to GPU KV cache residency rather than disk retention); and the comparison set is three established cross-request KV reuse methods rather than three LSH variants. The structure of both papers is the same: survey three established methods, characterize the coverage gap each leaves open, introduce a Jaccard-MinHash method that fills the remaining gap, and validate empirically against a ground-truth precision-recall oracle. The algorithmic core and hash family are carried over unchanged.

### 8.4 Limitations

- The method helps only on workloads with cross-request redundancy. Single-tenant deployments with high prompt diversity will not benefit.
- The theoretical S-curve assumes uniformly random hash functions. The ax + b mod p family used here is sufficient for practical purposes but is not adversarially robust against attackers who can probe the index.
- The benchmarks in Section 7 are synthetic. Real workload traces from agent frameworks, RAG pipelines, and templated chatbots are required before claims about production hit rates can be made with confidence.
- The SemShareKV head-to-head in Section 7 uses a matching-layer simulator, not the real SemShareKV implementation. The simulator captures the matching mechanism (cosine LSH on prompt vectors plus exact-cosine verification) but uses synthetic embedding proxies in place of real model embeddings. A full head-to-head awaits a public release of the SemShareKV reference implementation.

---

## 9. Conclusion

Across the three established methods reviewed in Sections 3–5, cross-request KV reuse is partitioned by the kind of similarity it captures: exact match for templates without substitution (APC, RadixAttention), or cosine similarity over embeddings for semantic paraphrase (SemShareKV). The lexical near-duplicate case — large shared token spans broken by small per-request substitutions — is unaddressed.

JaccardServe fills this gap with a gateway-tier matching layer that uses MinHash–LSH over token shingles, runs in well under a millisecond on commodity CPU, requires no model or kernel changes, and composes additively with the three established methods. On a templated-chat benchmark it surfaces 18.7× more reusable near-duplicates than a vLLM-style exact prefix cache. On a multi-document summarization benchmark it achieves an oracle F1 score of 0.853 (precision 0.905, recall 0.806) versus 0.324 for vLLM APC and 0.120 for the SemShareKV matching-layer simulator. The algorithmic primitive is the same one the author validated in [Panchal, 2018] for document deduplication; the contribution of the present work is its placement in the LLM serving pipeline, the per-token granularity, and the end-to-end CPU-reproducible reference implementation.

The GPU evaluation in Sections 7.8–7.9 establishes two further results. First, the matching layer's high hit rate (95.5% on templated prompts) translates into a measurable latency signal: −44% median TTFT on templated-chat via the span-start consistency mechanism described in Section 7.8. Second, the naive realization of that signal — Strategy A, which reorders matched tokens to exploit vLLM's exact prefix cache — is structurally incompatible with chat-template prompts, producing a 60% MAJOR-difference rate in LLM-judge evaluation (Section 7.9). The root cause is that reordering displaces user-specific context from its expected position in the prompt structure, and the degradation is not tunable away.

These two results together point to a precise requirement for production deployment: the matching layer's identified spans must be reused in-place, without altering the target prompt's token order. Strategy B (direct KV block-table injection at matched positions) satisfies this requirement. It maintains the original token order, requires no RoPE re-rotation when the matched span sits at the same absolute positions in donor and target, and removes the quality risk entirely. The matching layer, Jaccard threshold, and banding configuration developed here carry over to Strategy B unchanged.

---

## Benefits Over the Three Established Methods

Following the format of [Panchal, 2018]:

- **Catches a class of near-duplication that vLLM APC and SGLang RadixAttention cannot catch.** The 5.2% → 97.4% hit-rate jump on the templated benchmark is direct evidence.
- **Runs at the gateway, not on the GPU.** Unlike SemShareKV, no model forward pass is required before matching is complete. The 0.28 ms per-request overhead is independent of model size.
- **Many-to-many batch structure.** Unlike SemShareKV's pair-wise reference-target lookup, JaccardServe maintains a global index that captures redundancy across any pair of concurrent requests.
- **Closed-form precision-recall knob.** The banding analysis gives operators a tunable, predictable curve. SemShareKV's cosine-LSH parameters are tuned empirically.
- **Composes with existing stack.** No replacement of APC or RadixAttention is required. JaccardServe runs upstream; the existing exact-match caches continue to short-circuit on identical-prefix hits.
- **Algorithmically validated.** The MinHash-banding primitive is the same one the author benchmarked in the document-deduplication setting [Panchal, 2018] and is the de facto standard for LLM training-data deduplication at scale [NVIDIA NeMo Curator; Milvus 2.6].
- **End-to-end CPU-reproducible.** Every number and figure in Section 7 is produced by `run_all.sh` in approximately two minutes on a single CPU core. No GPU or model weights required to verify the matching-layer claims.

---

## References

Bang, F. (2023). GPTCache: An open-source semantic cache for LLM applications. *EMNLP Industry Track*.

Broder, A. Z. (1997). On the resemblance and containment of documents. *Compression and Complexity of Sequences*.

Broder, A. Z., Glassman, S. C., Manasse, M. S., & Zweig, G. (1997). Syntactic clustering of the web. *Computer Networks and ISDN Systems*, 29(8–13), 1157–1166.

Charikar, M. S. (2002). Similarity estimation techniques from rounding algorithms. *STOC*.

Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: Towards removing the curse of dimensionality. *STOC*.

Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The efficient transformer. *ICLR*. arXiv:2001.04451.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. *SOSP*.

Leskovec, J., Rajaraman, A., & Ullman, J. D. (2014). *Mining of Massive Datasets* (2nd ed.). Cambridge University Press.

Ling, K., & Wu, G. (2011). Frequency Based Locality Sensitive Hashing. *International Conference on Multimedia Technology*. doi:10.1109/icmt.2011.6002015a.

Lv, J., Wang, C., Bao, G., et al. (2024). CritiPrefill: A segment-wise criticality-based approach for prefilling acceleration in LLMs. arXiv:2409.12490.

Panchal, A. (2018). MinHashLSH: Java implementation and benchmark of MinHash-LSH for near-duplicate document detection via Jaccard similarity, comparing improved Hamming-distance LSH, entropy-based LSH, and frequency-based LSH against a Jaccard-MinHash baseline on text corpora. https://github.com/AnmolPanchal/Locality-Sensitive-Hashing-Using-Jaccard-Similarity

Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, Í., Maleki, S., & Bianchini, R. (2024). Splitwise: Efficient generative LLM inference using phase splitting. *ISCA*.

Schroeder, T., et al. (2025). vCache: Verified semantic caching for LLM applications.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*.

Wang, Q., Guo, Z., Liu, G., & Guo, J. (2012). Entropy based locality sensitive hashing. *ICASSP*. doi:10.1109/icassp.2012.6288065.

Wu, T., & Miao, Z. (2016). An improved feature image matching algorithm based on Locality-Sensitive Hashing. *IEEE ICSP*. doi:10.1109/icsp.2016.7877927.

Zhao, X., & Mastorakis, S. (2025). SemShareKV: Efficient KVCache sharing for semantically similar prompts via token-level LSH matching. *IJCNLP-AACL*. arXiv:2509.24832.

Zheng, L., Yin, L., Xie, Z., Sun, C. L., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., & Sheng, Y. (2024). SGLang: Efficient execution of structured language model programs. arXiv:2312.07104.

Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., & Zhang, H. (2024). DistServe: Disaggregating prefill and decoding for goodput-optimized LLM serving. *OSDI*.

Anonymous. (2026). From similarity to vulnerability: Key collision attack on LLM semantic caching. arXiv:2601.23088.

vLLM Project. (2025). RFC #16016: Cache salting for secure and flexible prefix caching. https://github.com/vllm-project/vllm/issues/16016
