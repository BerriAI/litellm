#!/usr/bin/env python3
"""
Benchmark: Standard LiteLLM Proxy vs fast-litellm Accelerated Proxy

Measures pure proxy overhead using network_mock mode (no real API calls).
Tests both configurations back-to-back and produces a comparison report.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import statistics
import time

import aiohttp

REQUEST_BODY = {
    "model": "db-openai-endpoint",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 100,
    "user": "benchmark_user",
}

HEADERS = {
    "Authorization": "Bearer sk-1234",
    "Content-Type": "application/json",
}


async def wait_for_proxy(url: str, timeout: int = 120) -> bool:
    health_url = url.rsplit("/", 1)[0] + "/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def send_request(session, url, semaphore):
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(url, json=REQUEST_BODY, headers=HEADERS) as resp:
                await resp.read()
                elapsed = time.perf_counter() - start
                return elapsed if resp.status == 200 else None
        except Exception:
            return None


async def run_benchmark(url: str, n_requests: int, max_concurrent: int):
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(
        limit=min(max_concurrent * 2, 500),
        limit_per_host=max_concurrent,
        force_close=False,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        warmup_count = min(100, n_requests // 2)
        await asyncio.gather(
            *[send_request(session, url, semaphore) for _ in range(warmup_count)]
        )

        wall_start = time.perf_counter()
        results = await asyncio.gather(
            *[send_request(session, url, semaphore) for _ in range(n_requests)]
        )
        wall_elapsed = time.perf_counter() - wall_start

    latencies = sorted([r for r in results if r is not None])
    failures = sum(1 for r in results if r is None)

    if not latencies:
        return {
            "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
            "min_ms": 0, "max_ms": 0,
            "throughput_rps": 0, "failures": n_requests,
            "wall_time_s": wall_elapsed, "n_requests": n_requests,
            "max_concurrent": max_concurrent, "latencies": [],
        }

    n = len(latencies)
    return {
        "mean_ms": statistics.mean(latencies) * 1000,
        "p50_ms": latencies[n // 2] * 1000,
        "p95_ms": latencies[int(n * 0.95)] * 1000,
        "p99_ms": latencies[int(n * 0.99)] * 1000,
        "min_ms": latencies[0] * 1000,
        "max_ms": latencies[-1] * 1000,
        "throughput_rps": n_requests / wall_elapsed,
        "failures": failures,
        "wall_time_s": wall_elapsed,
        "n_requests": n_requests,
        "max_concurrent": max_concurrent,
        "latencies": latencies,
    }


def aggregate_runs(results: list[dict]) -> dict:
    all_latencies = []
    for r in results:
        all_latencies.extend(r["latencies"])
    all_latencies.sort()

    total_failures = sum(r["failures"] for r in results)
    total_requests = sum(r["n_requests"] for r in results)
    n = len(all_latencies)

    if not all_latencies:
        return {"error": f"All {total_requests} requests failed"}

    return {
        "total_requests": total_requests,
        "total_failures": total_failures,
        "mean_ms": statistics.mean(all_latencies) * 1000,
        "p50_ms": all_latencies[n // 2] * 1000,
        "p95_ms": all_latencies[int(n * 0.95)] * 1000,
        "p99_ms": all_latencies[int(n * 0.99)] * 1000,
        "min_ms": all_latencies[0] * 1000,
        "max_ms": all_latencies[-1] * 1000,
        "avg_throughput_rps": statistics.mean(r["throughput_rps"] for r in results),
    }


def print_results(label: str, agg: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if "error" in agg:
        print(f"  ERROR: {agg['error']}")
        return
    print(f"  Total Requests:  {agg['total_requests']}  (failures: {agg['total_failures']})")
    print(f"  Throughput:      {agg['avg_throughput_rps']:.0f} req/s")
    print(f"  Mean latency:    {agg['mean_ms']:.2f} ms")
    print(f"  P50  latency:    {agg['p50_ms']:.2f} ms")
    print(f"  P95  latency:    {agg['p95_ms']:.2f} ms")
    print(f"  P99  latency:    {agg['p99_ms']:.2f} ms")
    print(f"  Min  latency:    {agg['min_ms']:.2f} ms")
    print(f"  Max  latency:    {agg['max_ms']:.2f} ms")


def print_comparison(standard: dict, fast: dict):
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Standard vs fast-litellm")
    print(f"{'='*60}")

    if "error" in standard or "error" in fast:
        print("  Cannot compare — one or both benchmarks had all failures.")
        return

    metrics = [
        ("Mean latency", "mean_ms", "ms", True),
        ("P50  latency", "p50_ms", "ms", True),
        ("P95  latency", "p95_ms", "ms", True),
        ("P99  latency", "p99_ms", "ms", True),
        ("Throughput", "avg_throughput_rps", "req/s", False),
    ]

    print(f"  {'Metric':<18} {'Standard':>12} {'fast-litellm':>14} {'Change':>12}")
    print(f"  {'-'*18} {'-'*12} {'-'*14} {'-'*12}")

    for label, key, unit, lower_is_better in metrics:
        std_val = standard[key]
        fast_val = fast[key]

        if std_val == 0:
            pct = "N/A"
        else:
            change = ((fast_val - std_val) / std_val) * 100
            pct = f"{change:+.1f}%"

        print(f"  {label:<18} {std_val:>10.2f} {unit[0]}  {fast_val:>12.2f} {unit[0]}  {pct:>10}")

    if standard.get("avg_throughput_rps", 0) > 0:
        speedup = fast.get("avg_throughput_rps", 0) / standard["avg_throughput_rps"]
        print(f"\n  Throughput multiplier: {speedup:.2f}x")

    if standard.get("mean_ms", 0) > 0:
        latency_reduction = (1 - fast.get("mean_ms", 0) / standard["mean_ms"]) * 100
        print(f"  Mean latency reduction: {latency_reduction:.1f}%")


def start_proxy(config_path: str, port: int, use_fast_litellm: bool, num_workers: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["LITELLM_LOG"] = "ERROR"

    if use_fast_litellm:
        cmd = [
            sys.executable, "-c",
            f"import fast_litellm; from litellm.proxy.proxy_cli import run_server; run_server()",
            "--config", config_path,
            "--port", str(port),
            "--num_workers", str(num_workers),
        ]
    else:
        cmd = [
            sys.executable, "-m", "litellm",
            "--config", config_path,
            "--port", str(port),
            "--num_workers", str(num_workers),
        ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def stop_proxy(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


async def run_full_benchmark(
    config_path: str,
    port: int,
    use_fast_litellm: bool,
    n_requests: int,
    max_concurrent: int,
    n_runs: int,
    num_workers: int,
) -> dict:
    label = "fast-litellm" if use_fast_litellm else "Standard LiteLLM"
    print(f"\n{'#'*60}")
    print(f"  Starting {label} proxy on port {port} ({num_workers} workers)")
    print(f"{'#'*60}")

    proc = start_proxy(config_path, port, use_fast_litellm, num_workers)

    url = f"http://localhost:{port}/chat/completions"

    print(f"  Waiting for proxy to be ready...")
    ready = await wait_for_proxy(url, timeout=120)

    if not ready:
        print(f"  ERROR: Proxy did not start within 120s")
        stderr_output = ""
        if proc.poll() is not None:
            stderr_output = proc.stderr.read().decode("utf-8", errors="replace")[-2000:]
        stop_proxy(proc)
        print(f"  Proxy stderr: {stderr_output}")
        return {"error": "Proxy did not start"}

    print(f"  Proxy is ready! Running benchmark...")
    print(f"  {n_requests} requests, {max_concurrent} concurrency, {n_runs} run(s)")

    results = []
    for run_num in range(1, n_runs + 1):
        result = await run_benchmark(url, n_requests, max_concurrent)
        results.append(result)
        print(f"\n  Run {run_num}/{n_runs}: mean={result['mean_ms']:.2f}ms, "
              f"p50={result['p50_ms']:.2f}ms, p95={result['p95_ms']:.2f}ms, "
              f"throughput={result['throughput_rps']:.0f} req/s, "
              f"failures={result['failures']}")

    agg = aggregate_runs(results)
    print_results(label, agg)

    stop_proxy(proc)
    await asyncio.sleep(2)

    return agg


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Standard LiteLLM vs fast-litellm proxy")
    parser.add_argument("--config", default="benchmark_config.yaml")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--max-concurrent", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None, help="Write JSON results to file")
    args = parser.parse_args()

    print("=" * 60)
    print("  LiteLLM Proxy Benchmark: Standard vs fast-litellm")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Requests:     {args.requests}")
    print(f"  Concurrency:  {args.max_concurrent}")
    print(f"  Runs:         {args.runs}")
    print(f"  Workers:      {args.num_workers}")
    print(f"  Mode:         network_mock (pure proxy overhead)")

    standard_results = await run_full_benchmark(
        config_path=args.config,
        port=4000,
        use_fast_litellm=False,
        n_requests=args.requests,
        max_concurrent=args.max_concurrent,
        n_runs=args.runs,
        num_workers=args.num_workers,
    )

    fast_results = await run_full_benchmark(
        config_path=args.config,
        port=4001,
        use_fast_litellm=True,
        n_requests=args.requests,
        max_concurrent=args.max_concurrent,
        n_runs=args.runs,
        num_workers=args.num_workers,
    )

    print_comparison(standard_results, fast_results)

    if args.output:
        output_data = {
            "config": {
                "requests": args.requests,
                "concurrency": args.max_concurrent,
                "runs": args.runs,
                "workers": args.num_workers,
                "mode": "network_mock",
            },
            "standard_litellm": {k: v for k, v in standard_results.items() if k != "latencies"},
            "fast_litellm": {k: v for k, v in fast_results.items() if k != "latencies"},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
