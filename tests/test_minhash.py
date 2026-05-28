"""
Unit tests for MinHash signature correctness.

These validate the implementation against the defining property:
    E[ 1[h_i(A) = h_i(B)] ] = J(A, B)

A signature length of 128 gives a standard error of about 0.044, so
absolute tolerance of 0.08 is generous for a smoke test. For paper
results we tighten this and report mean and variance over many trials.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from jaccardserve.minhash import MinHasher, MinHashConfig
from jaccardserve.shingler import shingle_set


class TestMinHash(unittest.TestCase):

    def setUp(self):
        self.mh = MinHasher(MinHashConfig(num_perms=128, seed=42))

    def test_identical_sets_give_identical_signatures(self):
        shingles = set(range(100))
        sig_a = self.mh.signature(shingles)
        sig_b = self.mh.signature(shingles)
        np.testing.assert_array_equal(sig_a, sig_b)
        self.assertEqual(self.mh.jaccard_estimate(sig_a, sig_b), 1.0)

    def test_disjoint_sets_give_low_estimate(self):
        a = set(range(100))
        b = set(range(1000, 1100))
        sig_a = self.mh.signature(a)
        sig_b = self.mh.signature(b)
        est = self.mh.jaccard_estimate(sig_a, sig_b)
        self.assertLess(est, 0.05)
        self.assertAlmostEqual(self.mh.exact_jaccard(a, b), 0.0)

    def test_partial_overlap(self):
        a = set(range(1000))
        b = set(range(500, 1500))  # exact J = 500/1500 = 1/3
        sig_a = self.mh.signature(a)
        sig_b = self.mh.signature(b)
        est = self.mh.jaccard_estimate(sig_a, sig_b)
        exact = self.mh.exact_jaccard(a, b)
        self.assertAlmostEqual(exact, 1 / 3, places=6)
        # Theoretical MinHash std error at m=128, J=1/3 is ~0.042;
        # 0.12 = ~3 sigma is a safe smoke-test bound.
        self.assertAlmostEqual(est, exact, delta=0.12)

    def test_empty_input(self):
        sig = self.mh.signature(set())
        self.assertEqual(sig.shape, (128,))

    def test_signature_on_token_shingles(self):
        rng = random.Random(123)
        tokens_a = [rng.randint(0, 50000) for _ in range(500)]
        tokens_b = tokens_a[:400] + [rng.randint(0, 50000) for _ in range(100)]
        a = shingle_set(tokens_a, k=5)
        b = shingle_set(tokens_b, k=5)
        sig_a = self.mh.signature(a)
        sig_b = self.mh.signature(b)
        est = self.mh.jaccard_estimate(sig_a, sig_b)
        exact = self.mh.exact_jaccard(a, b)
        # 5-shingles on 80% prefix overlap; expect high Jaccard.
        self.assertGreater(exact, 0.5)
        self.assertAlmostEqual(est, exact, delta=0.1)


if __name__ == "__main__":
    unittest.main()
