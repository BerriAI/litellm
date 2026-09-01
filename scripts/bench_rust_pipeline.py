"""Comprehensive benchmark: Rust vs Python across all pipeline operations.

Compares wall-clock time for token counting, cost calculation, auth hashing,
and the full pipeline. Run with:
    python scripts/bench_rust_pipeline.py

Requires all LITELLM_RUST_* env vars set to 1 and the Rust bridge to be built.
"""

from __future__ import annotations

import os
import statistics
import time

os.environ["LITELLM_RUST_TOKEN_COUNTER"] = "1"
os.environ["LITELLM_RUST_COST_CALCULATOR"] = "1"
os.environ["LITELLM_RUST_AUTH"] = "1"
os.environ["LITELLM_RUST_PIPELINE"] = "1"

import litellm
from litellm.cost_calculator import cost_per_token as python_cost_per_token
from litellm.proxy.utils import hash_token as python_hash_token
from litellm.litellm_core_utils.token_counter import token_counter as python_token_counter
from litellm.rust_bridge.auth import try_rust_hash_token
from litellm.rust_bridge.cost_calculator import try_rust_completion_cost
from litellm.rust_bridge.pipeline import process_request as rust_process_request
from litellm.rust_bridge.token_counter import try_rust_token_counter


def benchmark(func, args: tuple, kwargs: dict, iterations: int = 200) -> list[float]:
    """Run func(*args, **kwargs) `iterations` times, return list of elapsed microseconds."""
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        timings.append(elapsed_us)
    return timings


def percentile(data: list[float], p: float) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def bench_token_counter():
    print("\n=== Token Counter ===")
    workloads = [
        ("short text", {"model": "gpt-4o", "text": "Hello world"}),
        ("medium text (10K)", {"model": "gpt-4o", "text": "A" * 10_000}),
        ("long text (100K)", {"model": "gpt-4o", "text": "A" * 100_000}),
        ("3 messages", {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ],
        }),
        ("100 messages", {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
            ] + [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}."}
                for i in range(100)
            ],
        }),
    ]

    for name, kwargs in workloads:
        py_timings = benchmark(python_token_counter, (), kwargs)
        rust_result = try_rust_token_counter(**kwargs)
        if rust_result is None:
            print(f"  {name:<25} Python p50={percentile(py_timings, 50):>8.0f}us  Rust: UNAVAILABLE")
            continue
        rust_timings = benchmark(try_rust_token_counter, (), kwargs)
        py_p50 = percentile(py_timings, 50)
        rust_p50 = percentile(rust_timings, 50)
        speedup = py_p50 / rust_p50 if rust_p50 > 0 else float("inf")
        print(f"  {name:<25} Python p50={py_p50:>8.0f}us  Rust p50={rust_p50:>8.0f}us  {speedup:.1f}x")


def bench_cost_calculator():
    print("\n=== Cost Calculator ===")
    workloads = [
        ("basic (100/50 tokens)", "gpt-4o", 100, 50),
        ("large (10K/1K tokens)", "gpt-4o", 10_000, 1_000),
        ("with cache hit", "gpt-4o", 1_000, 100),
        ("claude with cache", "claude-3-opus-20240229", 1_000, 100),
    ]

    for name, model, prompt_tokens, completion_tokens in workloads:
        py_kwargs = {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        py_timings = benchmark(python_cost_per_token, (), py_kwargs)
        rust_kwargs = {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        rust_result = try_rust_completion_cost(**rust_kwargs)
        if rust_result is None:
            print(f"  {name:<25} Python p50={percentile(py_timings, 50):>8.0f}us  Rust: UNAVAILABLE")
            continue
        rust_timings = benchmark(try_rust_completion_cost, (), rust_kwargs)
        py_p50 = percentile(py_timings, 50)
        rust_p50 = percentile(rust_timings, 50)
        speedup = py_p50 / rust_p50 if rust_p50 > 0 else float("inf")
        print(f"  {name:<25} Python p50={py_p50:>8.0f}us  Rust p50={rust_p50:>8.0f}us  {speedup:.1f}x")


def bench_auth_hash():
    print("\n=== Auth Hash (SHA-256) ===")
    tokens = [
        ("short key", "sk-1234567890abcdef"),
        ("medium key", "sk-" + "a" * 100),
        ("long key", "sk-" + "x" * 1000),
    ]

    for name, token in tokens:
        py_timings = benchmark(python_hash_token, (token,), {})
        rust_result = try_rust_hash_token(token)
        if rust_result is None:
            print(f"  {name:<25} Python p50={percentile(py_timings, 50):>8.0f}us  Rust: UNAVAILABLE")
            continue
        rust_timings = benchmark(try_rust_hash_token, (token,), {})
        py_p50 = percentile(py_timings, 50)
        rust_p50 = percentile(rust_timings, 50)
        speedup = py_p50 / rust_p50 if rust_p50 > 0 else float("inf")
        print(f"  {name:<25} Python p50={py_p50:>8.0f}us  Rust p50={rust_p50:>8.0f}us  {speedup:.1f}x")


def bench_full_pipeline():
    print("\n=== Full Pipeline (process_request) ===")
    request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
    }

    rust_result = rust_process_request("/v1/chat/completions", request)
    if rust_result is None:
        print("  Rust pipeline: UNAVAILABLE")
        return

    timings = benchmark(rust_process_request, ("/v1/chat/completions", request), {})
    p50 = percentile(timings, 50)
    p95 = percentile(timings, 95)
    p99 = percentile(timings, 99)
    throughput = 1_000_000 / p50
    print(f"  p50={p50:>8.0f}us  p95={p95:>8.0f}us  p99={p99:>8.0f}us  throughput={throughput:.0f} req/s")


def verify_parity():
    print("\n=== Parity Verification ===")
    mismatches = 0

    # Token counter parity
    for model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]:
        for text in ["hello", "A" * 1000, "unicode: áéíóú"]:
            py = python_token_counter(model=model, text=text)
            rust = try_rust_token_counter(model=model, text=text)
            if rust is not None and rust != py:
                print(f"  MISMATCH token_counter: model={model} text={text[:20]!r} py={py} rust={rust}")
                mismatches += 1

    # Cost calculator parity
    for model in ["gpt-4o", "gpt-4"]:
        py_prompt, py_completion = python_cost_per_token(model=model, prompt_tokens=100, completion_tokens=50)
        py_total = py_prompt + py_completion
        rust = try_rust_completion_cost(model=model, prompt_tokens=100, completion_tokens=50)
        if rust is not None and abs(rust - py_total) > 1e-12:
            print(f"  MISMATCH completion_cost: model={model} py={py_total} rust={rust}")
            mismatches += 1

    # Auth hash parity
    for token in ["sk-test", "sk-" + "a" * 100]:
        py = python_hash_token(token)
        rust = try_rust_hash_token(token)
        if rust is not None and rust != py:
            print(f"  MISMATCH hash_token: token={token[:20]!r} py={py} rust={rust}")
            mismatches += 1

    if mismatches == 0:
        print("  All parity checks passed")
    else:
        print(f"  {mismatches} parity mismatches found")


def main():
    print("Rust Pipeline Benchmark: Comprehensive Comparison")
    print("=" * 70)

    verify_parity()
    bench_token_counter()
    bench_cost_calculator()
    bench_auth_hash()
    bench_full_pipeline()

    print("\n" + "=" * 70)
    print("Notes:")
    print("  - All times in microseconds (us)")
    print("  - Speedup = Python p50 / Rust p50")
    print("  - 200 iterations per workload")
    print("  - UNAVAILABLE means Rust bridge not built or env var not set")


if __name__ == "__main__":
    main()
