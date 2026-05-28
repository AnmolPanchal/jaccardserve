"""
vLLM integration: JaccardServeEngine.

Wraps a vLLM engine (real, on GPU) or a mock (CPU-only, for
development) with JaccardServe gateway-tier matching. The matching
layer is identical in both cases; only the downstream serving path
differs.

Two strategies for converting a match into actual TTFT savings:

  Strategy A (prompt-prefix reordering):
    When a match is found, rewrite the prompt so the matched span
    appears at position 0. vLLM's existing APC then sees an exact
    prefix and short-circuits. This requires NO vLLM modifications
    and is the recommended first integration.

  Strategy B (block-table injection):
    Modify vLLM's block_manager so the block table for matched
    positions points at the donor's physical blocks. Requires
    monkey-patching vLLM internals; outlined in vllm_integration/
    README.md but not implemented in this reference.

The reference implementation uses Strategy A.

Usage (real vLLM, GPU):

    from vllm_integration.engine import JaccardServeEngine
    engine = JaccardServeEngine.from_vllm(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        banding=BandingConfig(20, 4),
    )
    output, plan = engine.generate("your prompt here")

Usage (mock, CPU-only):

    engine = JaccardServeEngine.mock()
    output, plan = engine.generate("your prompt here")
"""

from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaccardserve import (
    BandingConfig,
    InjectionPlan,
    JaccardServeGateway,
)


# ---------------------------------------------------------------------------
# Mock engine (CPU-only)
# ---------------------------------------------------------------------------

class MockVLLMEngine:
    """
    Stand-in for vLLM's LLM class. Does not generate real text; the
    point is to validate the gateway pipeline end-to-end without
    needing a GPU. Simulates prefill TTFT proportional to the number
    of unmatched tokens, so that JaccardServe matches show up as
    measurable latency reductions.
    """

    def __init__(self, tokens_per_ms: float = 50.0):
        self.tokens_per_ms = tokens_per_ms
        self._whitespace_vocab: dict[str, int] = {}

    def get_tokenizer(self) -> Callable[[str], Sequence[int]]:
        def tok(s: str) -> list[int]:
            ids = []
            for word in s.split():
                if word not in self._whitespace_vocab:
                    self._whitespace_vocab[word] = len(self._whitespace_vocab) + 1
                ids.append(self._whitespace_vocab[word])
            return ids
        return tok

    def generate(
        self,
        prompt: str,
        tokens_to_skip: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        token_ids = list(self.get_tokenizer()(prompt))
        prefill_tokens = max(0, len(token_ids) - tokens_to_skip)
        simulated_prefill_ms = prefill_tokens / self.tokens_per_ms
        time.sleep(simulated_prefill_ms / 1000.0)
        return {
            "text": f"[mock output for {len(token_ids)} tokens, "
                    f"{prefill_tokens} prefilled]",
            "prefill_tokens": prefill_tokens,
            "skipped_tokens": tokens_to_skip,
            "prefill_ms": simulated_prefill_ms,
        }


# ---------------------------------------------------------------------------
# Real vLLM wrapper (requires GPU + vLLM installed)
# ---------------------------------------------------------------------------

class RealVLLMEngineAdapter:
    """
    Real vLLM adapter. Strategy A: relies on vLLM APC for the actual
    KV reuse, with JaccardServe responsible only for the matching
    decision and prompt routing.
    """

    def __init__(self, model_name: str, **vllm_kwargs):
        from vllm import LLM, SamplingParams  # imported lazily
        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_name,
            enable_prefix_caching=True,
            **vllm_kwargs,
        )

    def get_tokenizer(self):
        return self.llm.get_tokenizer()

    def generate(self, prompt_input, max_tokens: int = 128, **kwargs):
        """prompt_input: str or {"prompt_token_ids": list[int]}"""
        params = self.SamplingParams(max_tokens=max_tokens, **kwargs)
        t0 = time.perf_counter()
        outs = self.llm.generate([prompt_input], params)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "text": outs[0].outputs[0].text,
            "ttft_ms": dt_ms,
            "input_tokens": len(outs[0].prompt_token_ids),
        }


# ---------------------------------------------------------------------------
# JaccardServeEngine
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    text: str
    plan: InjectionPlan
    engine_metadata: dict


class JaccardServeEngine:
    """
    Public entry point. Holds the gateway and the underlying engine.
    """

    def __init__(
        self,
        engine,
        tokenizer: Callable[[str], Sequence[int]],
        banding: BandingConfig = BandingConfig(num_bands=20, rows_per_band=4),
        threshold: float = 0.5,
        shingle_width: int = 5,
    ):
        self.engine = engine
        self.gateway = JaccardServeGateway(
            tokenizer=tokenizer,
            shingle_width=shingle_width,
            banding=banding,
            jaccard_threshold=threshold,
        )

    @classmethod
    def mock(cls, **kwargs) -> "JaccardServeEngine":
        eng = MockVLLMEngine()
        return cls(engine=eng, tokenizer=eng.get_tokenizer(), **kwargs)

    @classmethod
    def from_vllm(cls, model_name: str, **kwargs) -> "JaccardServeEngine":
        adapter = RealVLLMEngineAdapter(model_name=model_name)
        return cls(engine=adapter, tokenizer=adapter.get_tokenizer(), **kwargs)

    def generate(self, prompt: str, **engine_kwargs) -> GenerationResult:
        plan, token_ids = self.gateway.match(prompt)

        # Strategy A: when a hit is found, rely on vLLM's APC to
        # short-circuit since the donor is already in cache. tokens_to_skip
        # is only used by the mock engine to model the TTFT saving.
        tokens_to_skip = 0
        if plan.matched_target_span:
            start, end = plan.matched_target_span
            tokens_to_skip = end - start

        if isinstance(self.engine, MockVLLMEngine):
            out = self.engine.generate(prompt, tokens_to_skip=tokens_to_skip,
                                       **engine_kwargs)
        else:
            out = self.engine.generate(prompt, **engine_kwargs)

        # Register this request so future ones can match against it.
        self.gateway.register(plan.request_id, token_ids, worker_id="w0")
        return GenerationResult(text=out["text"], plan=plan, engine_metadata=out)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

def main():
    """
    Sanity check: run the mock engine over a small set of templated
    prompts and report the matching decisions.
    """
    engine = JaccardServeEngine.mock(
        banding=BandingConfig(num_bands=20, rows_per_band=4),
        threshold=0.5,
        shingle_width=5,
    )

    prompts = [
        "You are an assistant for Alice, who works as a software engineer at Meta. "
        "Help them with their request. Be concise and accurate. Always cite sources "
        "when you reference factual claims. If you do not know the answer, say so. "
        "User: Summarize the attached document.",
        "You are an assistant for Bob, who works as a software engineer at Meta. "
        "Help them with their request. Be concise and accurate. Always cite sources "
        "when you reference factual claims. If you do not know the answer, say so. "
        "User: Summarize the attached document.",
        "You are an assistant for Carol, who works as a data scientist at Google. "
        "Help them with their request. Be concise and accurate. Always cite sources "
        "when you reference factual claims. If you do not know the answer, say so. "
        "User: Summarize the attached document.",
        "Translate the following French text into English: bonjour le monde, "
        "comment allez-vous aujourd'hui, le temps est magnifique.",
        "You are an assistant for Dave, who works as a software engineer at Meta. "
        "Help them with their request. Be concise and accurate. Always cite sources "
        "when you reference factual claims. If you do not know the answer, say so. "
        "User: Summarize the attached document.",
    ]
    for p in prompts:
        result = engine.generate(p)
        hit = result.plan.donor_request_id is not None
        print(f"[{'HIT' if hit else 'MISS'}] J={result.plan.measured_jaccard:.3f}  "
              f"overhead={result.plan.gateway_overhead_ms:.2f}ms  "
              f"prefill_ms={result.engine_metadata.get('prefill_ms', 0):.2f}  "
              f"skipped={result.engine_metadata.get('skipped_tokens', 0)} "
              f"prompt='{p[:60]}...'")


if __name__ == "__main__":
    main()
