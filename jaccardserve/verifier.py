"""
Verification and span-resolution helpers.

LSH gives candidate matches; verifier.py computes exact Jaccard on
those candidates and identifies the longest contiguous matching token
span between two prompts.
"""

from __future__ import annotations

from typing import Sequence

from .shingler import shingle_hashes


def verify_jaccard(target_shingles: set[int], donor_shingles: set[int]) -> float:
    """Exact Jaccard similarity between two shingle sets."""
    if not target_shingles and not donor_shingles:
        return 1.0
    union = target_shingles | donor_shingles
    if not union:
        return 0.0
    return len(target_shingles & donor_shingles) / len(union)


def longest_matching_span(
    target_token_ids: Sequence[int],
    donor_token_ids: Sequence[int] | None,
    shingle_width: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Find the longest contiguous matching token span between target and
    donor. Returns ((target_start, target_end), (donor_start, donor_end))
    in target/donor token coordinates respectively, or None if no match
    of length >= shingle_width is found.

    Uses a standard longest-common-substring DP on token IDs. O(|t| *
    |d|) time and space. For long prompts this is the largest cost in
    the verification step; an FM-index or suffix-array implementation
    is recommended for production.

    If donor_token_ids is None (gateway holds shingle sets only, not
    raw tokens), we cannot localize the span and return None.
    """
    if donor_token_ids is None:
        return None
    n = len(target_token_ids)
    m = len(donor_token_ids)
    if n == 0 or m == 0:
        return None

    # DP table; use two rows to save memory.
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    best_len = 0
    best_target_end = 0
    best_donor_end = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target_token_ids[i - 1] == donor_token_ids[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_target_end = i
                    best_donor_end = j
            else:
                curr[j] = 0
        prev, curr = curr, [0] * (m + 1)

    if best_len < shingle_width:
        return None
    target_span = (best_target_end - best_len, best_target_end)
    donor_span = (best_donor_end - best_len, best_donor_end)
    return target_span, donor_span
