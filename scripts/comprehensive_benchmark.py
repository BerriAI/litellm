#!/usr/bin/env python3
"""
Comprehensive benchmark comparing Standard LiteLLM vs fast-litellm proxy.

Tests multiple concurrency levels and produces a detailed comparison report.
"""

import asyncio
import json
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


async def run_benchmark(url: str, n_requests: int, max_concurrent: int, n_runs: int = 3):
    all_results = []

    for run_num in range(n_runs):
        semaphore = asyncio.Semaphore(max_concurrent)
        connector = aiohttp.TCPConnector(
            limit=min(max_concurrent * 2, 500),
            limit_per_host=max_concurrent,
            force_close=False,
            enable_cleanup_closed=True,
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            warmup = min(50, n_requests // 4)
            await asyncio.gather(
                *[send_request(session, url, semaphore) for _ in range(warmup)]
            )

            wall_start = time.perf_counter()
            results = await asyncio.gather(
                *[send_request(session, url, semaphore) for _ in range(n_requests)]
            )
            wall_elapsed = time.perf_counter() - wall_start

        latencies = sorted([r for r in results if r is not None])
        failures = sum(1 for r in results if r is None)
        n = len(latencies)

        if n > 0:
            run_result = {
                "mean_ms": statistics.mean(latencies) * 1000,
                "p50_ms": latencies[n // 2] * 1000,
                "p95_ms": latencies[int(n * 0.95)] * 1000,
                "p99_ms": latencies[int(n * 0.99)] * 1000,
                "throughput_rps": n_requests / wall_elapsed,
                "failures": failures,
                "wall_time_s": wall_elapsed,
            }
        else:
            run_result = {
                "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
                "throughput_rps": 0, "failures": n_requests, "wall_time_s": wall_elapsed,
            }
        all_results.append(run_result)

    means = [r["mean_ms"] for r in all_results if r["mean_ms"] > 0]
    p50s = [r["p50_ms"] for r in all_results if r["p50_ms"] > 0]
    p95s = [r["p95_ms"] for r in all_results if r["p95_ms"] > 0]
    p99s = [r["p99_ms"] for r in all_results if r["p99_ms"] > 0]
    thrpts = [r["throughput_rps"] for r in all_results if r["throughput_rps"] > 0]

    return {
        "concurrency": max_concurrent,
        "requests_per_run": n_requests,
        "runs": n_runs,
        "mean_ms": statistics.median(means) if means else 0,
        "p50_ms": statistics.median(p50s) if p50s else 0,
        "p95_ms": statistics.median(p95s) if p95s else 0,
        "p99_ms": statistics.median(p99s) if p99s else 0,
        "throughput_rps": statistics.median(thrpts) if thrpts else 0,
        "total_failures": sum(r["failures"] for r in all_results),
        "per_run": all_results,
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    concurrency_levels = [10, 50, 100, 200]
    results = {}

    print(f"\n{'='*70}")
    print(f"  Benchmarking: {args.label}")
    print(f"  URL: {args.url}")
    print(f"  Requests per level: {args.requests}, Runs per level: {args.runs}")
    print(f"  Concurrency levels: {concurrency_levels}")
    print(f"{'='*70}")

    for conc in concurrency_levels:
        print(f"\n  Concurrency={conc} ...", end=" ", flush=True)
        result = await run_benchmark(args.url, args.requests, conc, args.runs)
        results[conc] = result
        print(f"throughput={result['throughput_rps']:.0f} req/s, "
              f"mean={result['mean_ms']:.1f}ms, p50={result['p50_ms']:.1f}ms, "
              f"p95={result['p95_ms']:.1f}ms, p99={result['p99_ms']:.1f}ms")

    print(f"\n  {'Conc':>6} | {'Throughput':>12} | {'Mean':>10} | {'P50':>10} | {'P95':>10} | {'P99':>10}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for conc in concurrency_levels:
        r = results[conc]
        print(f"  {conc:>6} | {r['throughput_rps']:>9.0f} rps | {r['mean_ms']:>7.1f} ms | "
              f"{r['p50_ms']:>7.1f} ms | {r['p95_ms']:>7.1f} ms | {r['p99_ms']:>7.1f} ms")

    output_data = {
        "label": args.label,
        "url": args.url,
        "requests_per_level": args.requests,
        "runs_per_level": args.runs,
        "results": {str(k): {kk: vv for kk, vv in v.items() if kk != "per_run"} for k, v in results.items()},
        "per_run_details": {str(k): v["per_run"] for k, v in results.items()},
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
