"""
LiteLLM SDK hot-path benchmark.

Measures the Python overhead in the request processing pipeline by calling
litellm.acompletion() against a local mock server. This isolates the SDK
processing overhead (serialization, routing, logging, response parsing)
from network latency.

Usage:
    # Start mock server first:
    poetry run python tests/load_tests/mock_openai_server.py &
    
    # Run benchmark:
    poetry run python tests/load_tests/benchmark_sdk_hotpath.py
"""

import asyncio
import os
import statistics
import sys
import time

os.environ["LITELLM_LOG"] = "ERROR"
os.environ.setdefault("OPENAI_API_KEY", "fake-key")

import litellm

litellm.telemetry = False
litellm.drop_params = True
litellm.num_retries = 0


async def bench_acompletion_direct(
    n_requests: int = 200,
    concurrency: int = 50,
) -> dict:
    """Benchmark litellm.acompletion() calls against mock server."""
    
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async def single_request():
        nonlocal errors
        async with semaphore:
            t0 = time.perf_counter()
            try:
                resp = await litellm.acompletion(
                    model="openai/fake-model",
                    messages=[{"role": "user", "content": "Hello world test message"}],
                    max_tokens=10,
                    api_base="http://127.0.0.1:18888/",
                    api_key="fake-key",
                )
                latencies.append(time.perf_counter() - t0)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Error: {e}")

    t_start = time.perf_counter()
    tasks = [asyncio.create_task(single_request()) for _ in range(n_requests)]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t_start

    latencies.sort()
    return {
        "total_requests": n_requests,
        "concurrency": concurrency,
        "errors": errors,
        "total_time_s": total_time,
        "rps": n_requests / total_time,
        "avg_ms": statistics.mean(latencies) * 1000 if latencies else 0,
        "p50_ms": latencies[len(latencies) // 2] * 1000 if latencies else 0,
        "p95_ms": latencies[int(len(latencies) * 0.95)] * 1000 if latencies else 0,
        "p99_ms": latencies[int(len(latencies) * 0.99)] * 1000 if latencies else 0,
        "min_ms": min(latencies) * 1000 if latencies else 0,
        "max_ms": max(latencies) * 1000 if latencies else 0,
    }


async def bench_router_acompletion(
    n_requests: int = 200,
    concurrency: int = 50,
) -> dict:
    """Benchmark Router.acompletion() with routing logic."""
    
    router = litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/fake-model",
                    "api_key": "fake-key",
                    "api_base": "http://127.0.0.1:18888/",
                    "rpm": 10000,
                },
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/fake-model-2",
                    "api_key": "fake-key-2",
                    "api_base": "http://127.0.0.1:18888/",
                    "rpm": 5000,
                },
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
    )

    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async def single_request():
        nonlocal errors
        async with semaphore:
            t0 = time.perf_counter()
            try:
                resp = await router.acompletion(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello world test message"}],
                    max_tokens=10,
                )
                latencies.append(time.perf_counter() - t0)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Error: {e}")

    t_start = time.perf_counter()
    tasks = [asyncio.create_task(single_request()) for _ in range(n_requests)]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t_start

    latencies.sort()
    return {
        "total_requests": n_requests,
        "concurrency": concurrency,
        "errors": errors,
        "total_time_s": total_time,
        "rps": n_requests / total_time,
        "avg_ms": statistics.mean(latencies) * 1000 if latencies else 0,
        "p50_ms": latencies[len(latencies) // 2] * 1000 if latencies else 0,
        "p95_ms": latencies[int(len(latencies) * 0.95)] * 1000 if latencies else 0,
        "p99_ms": latencies[int(len(latencies) * 0.99)] * 1000 if latencies else 0,
        "min_ms": min(latencies) * 1000 if latencies else 0,
        "max_ms": max(latencies) * 1000 if latencies else 0,
    }


def print_result(label: str, result: dict):
    print(f"\n  {label}")
    print(f"  {'='*60}")
    print(f"  Requests:    {result['total_requests']} ({result['errors']} errors)")
    print(f"  Concurrency: {result['concurrency']}")
    print(f"  Total time:  {result['total_time_s']:.2f}s")
    print(f"  Throughput:  {result['rps']:.1f} req/s")
    print(f"  Avg:         {result['avg_ms']:.1f} ms")
    print(f"  P50:         {result['p50_ms']:.1f} ms")
    print(f"  P95:         {result['p95_ms']:.1f} ms")
    print(f"  P99:         {result['p99_ms']:.1f} ms")
    print(f"  Min:         {result['min_ms']:.1f} ms")
    print(f"  Max:         {result['max_ms']:.1f} ms")


async def main():
    import subprocess
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: __import__("urllib.request", fromlist=["urlopen"]).urlopen(
                "http://127.0.0.1:18888/health"
            ),
        )
        if resp.status != 200:
            raise Exception("Mock server not healthy")
    except Exception:
        print("ERROR: Mock OpenAI server not running on port 18888.")
        print("Start it with: poetry run python tests/load_tests/mock_openai_server.py &")
        return 1

    print()
    print("=" * 70)
    print("  LITELLM SDK HOT-PATH BENCHMARK")
    print("  (against local mock server, measuring Python overhead)")
    print("=" * 70)

    # Warmup
    print("\n  Warming up...")
    await bench_acompletion_direct(n_requests=10, concurrency=5)

    # Direct acompletion
    print("\n  Running litellm.acompletion() benchmark (500 reqs, 100 concurrent)...")
    direct_result = await bench_acompletion_direct(n_requests=500, concurrency=100)
    print_result("litellm.acompletion() - Direct SDK call", direct_result)

    # Router acompletion
    print("\n  Running Router.acompletion() benchmark (500 reqs, 100 concurrent)...")
    router_result = await bench_router_acompletion(n_requests=500, concurrency=100)
    print_result("Router.acompletion() - With routing + load balancing", router_result)

    # High concurrency direct
    print("\n  Running high-concurrency benchmark (1000 reqs, 200 concurrent)...")
    hc_result = await bench_acompletion_direct(n_requests=1000, concurrency=200)
    print_result("litellm.acompletion() - High concurrency (200)", hc_result)

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Direct SDK:     {direct_result['rps']:>8.1f} req/s | P50: {direct_result['p50_ms']:>6.1f}ms | P99: {direct_result['p99_ms']:>6.1f}ms")
    print(f"  Router:         {router_result['rps']:>8.1f} req/s | P50: {router_result['p50_ms']:>6.1f}ms | P99: {router_result['p99_ms']:>6.1f}ms")
    print(f"  High Concurrency:{hc_result['rps']:>7.1f} req/s | P50: {hc_result['p50_ms']:>6.1f}ms | P99: {hc_result['p99_ms']:>6.1f}ms")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
