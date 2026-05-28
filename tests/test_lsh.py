"""
Tests for the banded LSH index and an empirical check of the S-curve
collision-probability formula. The empirical curve plot will go into
the paper's ablation section.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from jaccardserve.banded_lsh import BandedLSH, BandingConfig, LSHEntry
from jaccardserve.minhash import MinHasher, MinHashConfig


class TestBandedLSH(unittest.TestCase):

    def setUp(self):
        self.cfg = BandingConfig(num_bands=20, rows_per_band=8)
        self.mh = MinHasher(MinHashConfig(num_perms=self.cfg.num_perms, seed=7))
        self.index = BandedLSH(self.cfg)

    def _make_entry(self, rid: str, shingles: set[int]) -> LSHEntry:
        return LSHEntry(
            request_id=rid,
            worker_id="w0",
            signature=self.mh.signature(shingles),
            shingle_set=shingles,
            num_tokens=len(shingles) + 4,
        )

    def test_identical_sets_become_candidates(self):
        a = set(range(200))
        self.index.insert(self._make_entry("r1", a))
        sig = self.mh.signature(a)
        cands = self.index.query_candidates(sig)
        self.assertIn("r1", cands)

    def test_disjoint_sets_are_not_candidates(self):
        self.index.insert(self._make_entry("r1", set(range(200))))
        far = set(range(10000, 10200))
        cands = self.index.query_candidates(self.mh.signature(far))
        # With b=20, r=8 and J=0, P(collision) is essentially zero.
        self.assertEqual(cands, set())

    def test_evict_removes_entry(self):
        a = set(range(200))
        self.index.insert(self._make_entry("r1", a))
        self.index.evict("r1")
        self.assertEqual(self.index.size(), 0)
        self.assertEqual(self.index.query_candidates(self.mh.signature(a)), set())

    def test_empirical_s_curve(self):
        """
        Sweep true Jaccard, measure empirical collision rate, compare to
        P(s; b, r) = 1 - (1 - s^r)^b. This is the data behind Fig X of
        the paper's ablation section.
        """
        cfg = BandingConfig(num_bands=10, rows_per_band=5)
        mh = MinHasher(MinHashConfig(num_perms=cfg.num_perms, seed=11))
        idx = BandedLSH(cfg)
        rng = random.Random(0)

        # Build a fixed "donor" set.
        donor = set(rng.randint(0, 1_000_000) for _ in range(2000))
        idx.insert(LSHEntry(
            request_id="donor",
            worker_id=None,
            signature=mh.signature(donor),
            shingle_set=donor,
            num_tokens=2000,
        ))

        target_jaccards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        trials_per_point = 50
        results = []
        for target_j in target_jaccards:
            # Construct sets with the desired Jaccard via inclusion math:
            # for |A| = |B| = n and overlap k, J = k / (2n - k).
            n = 2000
            k = int(2 * n * target_j / (1 + target_j))
            hits = 0
            for _ in range(trials_per_point):
                overlap = rng.sample(sorted(donor), k=k)
                extras = set(rng.randint(2_000_000, 3_000_000) for _ in range(n - k))
                target = set(overlap) | extras
                cands = idx.query_candidates(mh.signature(target))
                if "donor" in cands:
                    hits += 1
            empirical = hits / trials_per_point
            theoretical = cfg.collision_prob(target_j)
            results.append((target_j, empirical, theoretical))

        # Sanity: empirical and theoretical should track each other within
        # binomial standard error. We assert a generous bound here; the
        # tighter version is for the paper plot, not for CI.
        for s, emp, theo in results:
            self.assertAlmostEqual(emp, theo, delta=0.25,
                                   msg=f"s={s}: emp={emp}, theo={theo}")


if __name__ == "__main__":
    unittest.main()
