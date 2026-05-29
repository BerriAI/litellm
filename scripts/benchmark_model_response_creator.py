#!/usr/bin/env python3
"""Tight microbenchmark for CustomStreamWrapper.model_response_creator.

Calls model_response_creator() in a tight loop on a pre-built wrapper to
isolate per-call cost. Driving the full wrapper adds threadpool logging,
gc, and other noise that swamps microsecond-scale changes here.

Example:
    uv run python scripts/benchmark_model_response_creator.py --label baseline
    uv run python scripts/benchmark_model_response_creator.py --label optimized
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import List
from unittest.mock import MagicMock

os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import litellm  # noqa: E402

litellm.suppress_debug_info = True

from litellm.litellm_core_utils.streaming_handler import (
    CustomStreamWrapper,
)  # noqa: E402


def _make_logging_obj(provider: str) -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {
        "custom_llm_provider": provider,
        "litellm_params": {},
    }
    logging_obj.call_type = "completion"
    logging_obj.stream_options = None
    logging_obj.messages = [{"role": "user", "content": "hi"}]
    logging_obj.completion_start_time = None
    logging_obj._llm_caching_handler = None
    return logging_obj


def _make_wrapper(provider: str, model: str) -> CustomStreamWrapper:
    return CustomStreamWrapper(
        completion_stream=iter([]),
        model=model,
        logging_obj=_make_logging_obj(provider),
        custom_llm_provider=provider,
    )


@dataclass
class Result:
    label: str
    scenario: str
    iterations: int
    elapsed_min_s: float
    elapsed_median_s: float
    per_call_us: float
    calls_per_sec: float


SCENARIOS = {
    "no_chunk": {
        "description": "model_response_creator() — no chunk arg (most common path)",
        "chunk_factory": lambda i: None,
    },
    "text_chunk": {
        "description": "model_response_creator(chunk={'text': '...'}) — text delta path",
        "chunk_factory": lambda i: {"text": f"token{i}"},
    },
    "rich_chunk": {
        "description": "model_response_creator(chunk={...}) — full chunk dict path",
        "chunk_factory": lambda i: {
            "id": f"id-{i}",
            "object": "chat.completion.chunk",
            "created": 1234567890,
        },
    },
}


def bench_no_chunk(wrapper: CustomStreamWrapper, iterations: int) -> float:
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(iterations):
            wrapper.model_response_creator()
        elapsed = time.perf_counter() - start
    finally:
        gc.enable()
    return elapsed


def bench_with_chunk(wrapper: CustomStreamWrapper, factory, iterations: int) -> float:
    # Pre-build chunks so we don't measure their construction cost.
    chunks = [factory(i) for i in range(iterations)]
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        for chunk in chunks:
            wrapper.model_response_creator(chunk=dict(chunk))  # copy because mutated
        elapsed = time.perf_counter() - start
    finally:
        gc.enable()
    return elapsed


def run_scenario(
    label: str,
    scenario_key: str,
    iterations: int,
    repeats: int,
    warmup: int,
) -> Result:
    spec = SCENARIOS[scenario_key]
    wrapper = _make_wrapper(provider="anthropic", model="claude-3-5-sonnet")

    if scenario_key == "no_chunk":
        runner = lambda: bench_no_chunk(wrapper, iterations)  # noqa: E731
    else:
        runner = lambda: bench_with_chunk(
            wrapper, spec["chunk_factory"], iterations
        )  # noqa: E731

    for _ in range(warmup):
        runner()
    samples = [runner() for _ in range(repeats)]

    elapsed_min = min(samples)
    elapsed_median = statistics.median(samples)
    per_call_us = (elapsed_min * 1_000_000) / iterations
    calls_per_sec = iterations / elapsed_min if elapsed_min > 0 else 0.0

    return Result(
        label=label,
        scenario=scenario_key,
        iterations=iterations,
        elapsed_min_s=elapsed_min,
        elapsed_median_s=elapsed_median,
        per_call_us=per_call_us,
        calls_per_sec=calls_per_sec,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--iterations", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    print(
        f"\n=== label={args.label}  iterations={args.iterations:,}  "
        f"warmup={args.warmup}  repeats={args.repeats} (min reported) ==="
    )
    results: List[Result] = []
    for scenario in SCENARIOS:
        r = run_scenario(
            args.label, scenario, args.iterations, args.repeats, args.warmup
        )
        results.append(r)
        print(
            f"  {r.scenario:12s}: "
            f"min={r.elapsed_min_s*1000:8.2f} ms  "
            f"median={r.elapsed_median_s*1000:8.2f} ms  "
            f"per-call={r.per_call_us:7.3f} μs  "
            f"calls/s={r.calls_per_sec:>12,.0f}"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nWrote {len(results)} results to {args.json_out}")


if __name__ == "__main__":
    main()
