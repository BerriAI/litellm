"""Benchmark: Rust token counter vs Python token counter.

Compares wall-clock time across realistic workloads. Run with:
    python scripts/bench_rust_token_counter.py

Requires LITELLM_RUST_TOKEN_COUNTER=1 and the Rust bridge to be built.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

os.environ["LITELLM_RUST_TOKEN_COUNTER"] = "1"

from litellm.litellm_core_utils.token_counter import token_counter as python_token_counter
from litellm.rust_bridge.token_counter import try_rust_token_counter


@dataclass
class Workload:
    name: str
    kwargs: dict


def _build_workloads() -> list[Workload]:
    short_text = "Hello world"
    medium_text = "A" * 10_000
    long_text = "The quick brown fox jumps over the lazy dog. " * 2000

    few_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    many_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message number {i} with some content to make it realistic."}
        for i in range(100)
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "num_results": {"type": "integer", "description": "Number of results"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The math expression"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
    ]

    tool_choice_named = {"type": "function", "function": {"name": "search_web"}}

    return [
        Workload("short_text (10 chars)", {"model": "gpt-4o", "text": short_text}),
        Workload("medium_text (10K chars)", {"model": "gpt-4o", "text": medium_text}),
        Workload("long_text (100K chars)", {"model": "gpt-4o", "text": long_text}),
        Workload("few_messages (3)", {"model": "gpt-4o", "messages": few_messages}),
        Workload("many_messages (100)", {"model": "gpt-4o", "messages": many_messages}),
        Workload(
            "messages + tools (3)",
            {"model": "gpt-4o", "messages": few_messages, "tools": tools},
        ),
        Workload(
            "messages + tools + tool_choice",
            {"model": "gpt-4o", "messages": few_messages, "tools": tools, "tool_choice": tool_choice_named},
        ),
    ]


def _benchmark(func, kwargs: dict, iterations: int = 100) -> list[float]:
    """Run func(kwargs) `iterations` times, return list of elapsed microseconds."""
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(**kwargs)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        timings.append(elapsed_us)
    return timings


def _percentile(data: list[float], p: float) -> float:
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[f]
    return data[f] + (k - f) * (data[c] - data[f])


def main():
    workloads = _build_workloads()
    iterations = 100

    print(f"Token Counter Benchmark: Rust vs Python ({iterations} iterations each)")
    print("=" * 90)

    rust_available = try_rust_token_counter(model="gpt-4o", text="test") is not None
    if not rust_available:
        print("WARNING: Rust bridge unavailable. All Rust results will be skipped.")
        print()

    header = f"{'Workload':<35} {'Python p50':>10} {'Python p95':>10} {'Rust p50':>10} {'Rust p95':>10} {'Speedup':>8}"
    print(header)
    print("-" * 90)

    for workload in workloads:
        python_timings = _benchmark(python_token_counter, workload.kwargs, iterations)
        python_timings.sort()

        rust_timings = None
        if rust_available:
            rust_result = try_rust_token_counter(**workload.kwargs)
            if rust_result is not None:
                rust_timings = _benchmark(
                    lambda **kw: try_rust_token_counter(**kw),
                    workload.kwargs,
                    iterations,
                )
                rust_timings.sort()

        py_p50 = _percentile(python_timings, 50)
        py_p95 = _percentile(python_timings, 95)

        if rust_timings:
            rust_p50 = _percentile(rust_timings, 50)
            rust_p95 = _percentile(rust_timings, 95)
            speedup = py_p50 / rust_p50 if rust_p50 > 0 else float("inf")
            print(
                f"{workload.name:<35} "
                f"{py_p50:>8.0f}us "
                f"{py_p95:>8.0f}us "
                f"{rust_p50:>8.0f}us "
                f"{rust_p95:>8.0f}us "
                f"{speedup:>7.1f}x"
            )
        else:
            print(
                f"{workload.name:<35} "
                f"{py_p50:>8.0f}us "
                f"{py_p95:>8.0f}us "
                f"{'N/A':>10} "
                f"{'N/A':>10} "
                f"{'N/A':>8}"
            )

    print()
    print("Notes:")
    print("  - Python path uses tiktoken via Python wrapper with chunked encoding")
    print("  - Rust path uses tiktoken crate directly with O(n) encoding")
    print("  - Speedup = Python p50 / Rust p50")
    print("  - All times in microseconds (us)")


if __name__ == "__main__":
    main()
