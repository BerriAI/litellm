"""Load test: hammer the Rust bridge with concurrent requests.

Measures throughput and latency under concurrent load. Run with:
    python scripts/load_test_rust_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import time

os.environ["LITELLM_RUST_TOKEN_COUNTER"] = "1"
os.environ["LITELLM_RUST_COST_CALCULATOR"] = "1"
os.environ["LITELLM_RUST_AUTH"] = "1"

from litellm.rust_bridge.auth import try_rust_hash_token
from litellm.rust_bridge.cost_calculator import try_rust_completion_cost
from litellm.rust_bridge.token_counter import try_rust_token_counter


async def bench_concurrent(func, args: tuple, kwargs: dict, concurrency: int, iterations: int):
    """Run func concurrently and measure throughput."""
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    tasks = []
    for _ in range(iterations):
        tasks.append(loop.run_in_executor(None, lambda: func(*args, **kwargs)))
        if len(tasks) >= concurrency:
            await asyncio.gather(*tasks)
            tasks.clear()
    if tasks:
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    return iterations / elapsed


async def main():
    print("Rust Pipeline Load Test")
    print("=" * 60)

    concurrency_levels = [1, 10, 50, 100]

    # Token counter load test
    print("\n=== Token Counter (100 messages) ===")
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}."} for i in range(100)]
    kwargs = {"model": "gpt-4o", "messages": messages}

    for concurrency in concurrency_levels:
        throughput = await bench_concurrent(
            try_rust_token_counter, (), kwargs, concurrency, iterations=500
        )
        print(f"  concurrency={concurrency:>3}  throughput={throughput:>8.0f} req/s")

    # Cost calculator load test
    print("\n=== Cost Calculator ===")
    kwargs = {"model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": 500}

    for concurrency in concurrency_levels:
        throughput = await bench_concurrent(
            try_rust_completion_cost, (), kwargs, concurrency, iterations=2000
        )
        print(f"  concurrency={concurrency:>3}  throughput={throughput:>8.0f} req/s")

    # Auth hash load test
    print("\n=== Auth Hash (SHA-256) ===")
    token = "sk-" + "a" * 100

    for concurrency in concurrency_levels:
        throughput = await bench_concurrent(
            try_rust_hash_token, (token,), {}, concurrency, iterations=5000
        )
        print(f"  concurrency={concurrency:>3}  throughput={throughput:>8.0f} req/s")

    print("\n" + "=" * 60)
    print("Load test complete.")


if __name__ == "__main__":
    asyncio.run(main())
