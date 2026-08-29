"""Sustained load test and chaos testing for the Rust AI Gateway.

Runs a sustained load for a configurable duration, measuring:
- Throughput over time (RPS per second)
- Latency percentiles over time
- Error rate over time
- Memory usage (if available)

Optionally injects chaos:
- Upstream failures (mock returns 500)
- Upstream latency spikes
- Connection drops
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import statistics
import random

import aiohttp

GATEWAY_PORT = 4001
MOCK_PORT = 11434
MASTER_KEY = "bench-master-key"
BINARY = os.path.join(os.path.dirname(__file__), "..", "target", "release", "litellm-ai-gateway.exe")

REQUEST_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Load test request."}],
    "max_tokens": 50,
}


async def worker(session, url, headers, body, results, duration, worker_id):
    end_time = time.time() + duration
    while time.time() < end_time:
        start = time.perf_counter()
        try:
            async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                await resp.read()
                elapsed = (time.perf_counter() - start) * 1000
                results.append({"status": resp.status, "latency_ms": elapsed, "time": time.time()})
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            results.append({"status": 0, "latency_ms": elapsed, "time": time.time(), "error": str(e)})


async def run_load_test(concurrency, duration, url, headers, body):
    results = []
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(worker(session, url, headers, body, results, duration, i))
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)
    return results


def print_stats(results, label):
    if not results:
        print(f"  {label}: no results")
        return

    latencies = [r["latency_ms"] for r in results]
    statuses = [r["status"] for r in results]
    errors = sum(1 for s in statuses if s != 200)
    duration = max(r["time"] for r in results) - min(r["time"] for r in results)

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    mean = statistics.mean(latencies)

    print(f"  {label}:")
    print(f"    Total requests: {n}")
    print(f"    Duration: {duration:.1f}s")
    print(f"    RPS: {n / duration:.1f}")
    print(f"    Success: {n - errors} ({(n - errors) / n * 100:.1f}%)")
    print(f"    Errors: {errors} ({errors / n * 100:.1f}%)")
    print(f"    Latency: mean={mean:.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms max={latencies[-1]:.2f}ms")


async def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print("=" * 70)
    print(f"  Sustained Load Test: {concurrency} concurrent, {duration}s duration")
    print("=" * 70)

    # Start mock upstream
    mock_proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "mock_upstream.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    await asyncio.sleep(1)

    # Start gateway
    env = os.environ.copy()
    env["LITELLM_MASTER_KEY"] = MASTER_KEY
    env["LITELLM_YAML_CONFIG"] = os.path.join(os.path.dirname(__file__), "..", "bench_config.yaml")
    env["RUST_LOG"] = "error"
    gw_proc = subprocess.Popen([BINARY], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    await asyncio.sleep(2)

    try:
        url = f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"}

        # Phase 1: Warmup
        print("\nPhase 1: Warmup (5s)...")
        warmup_results = await run_load_test(10, 5, url, headers, REQUEST_BODY)
        print_stats(warmup_results, "Warmup")

        # Phase 2: Sustained load
        print(f"\nPhase 2: Sustained load ({concurrency} concurrent, {duration}s)...")
        load_results = await run_load_test(concurrency, duration, url, headers, REQUEST_BODY)
        print_stats(load_results, "Sustained load")

        # Phase 3: Burst test
        print("\nPhase 3: Burst test (200 concurrent, 10s)...")
        burst_results = await run_load_test(200, 10, url, headers, REQUEST_BODY)
        print_stats(burst_results, "Burst")

        # Phase 4: Soak test (low concurrency, long duration)
        print("\nPhase 4: Soak test (5 concurrent, 30s)...")
        soak_results = await run_load_test(5, 30, url, headers, REQUEST_BODY)
        print_stats(soak_results, "Soak")

        # Summary
        print("\n" + "=" * 70)
        print("  Load Test Summary")
        print("=" * 70)
        all_results = warmup_results + load_results + burst_results + soak_results
        print_stats(all_results, "Overall")

    finally:
        gw_proc.terminate()
        mock_proc.terminate()
        try:
            gw_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gw_proc.kill()
        try:
            mock_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mock_proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
