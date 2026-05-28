"""JaccardServe: cross-request prefill acceleration via MinHash-LSH on token shingles."""

from .banded_lsh import BandedLSH, BandingConfig, LSHEntry
from .gateway import GatewayStats, InjectionPlan, JaccardServeGateway
from .minhash import MinHashConfig, MinHasher, stable_hash64
from .shingler import shingle_hashes, shingle_set, shingle_spans
from .verifier import longest_matching_span, verify_jaccard

__version__ = "0.1.0"

__all__ = [
    "BandedLSH",
    "BandingConfig",
    "LSHEntry",
    "MinHasher",
    "MinHashConfig",
    "JaccardServeGateway",
    "InjectionPlan",
    "GatewayStats",
    "shingle_hashes",
    "shingle_set",
    "shingle_spans",
    "verify_jaccard",
    "longest_matching_span",
    "stable_hash64",
]
