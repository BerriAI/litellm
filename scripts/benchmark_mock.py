#!/usr/bin/env python3
"""Quick benchmark for network_mock proxy overhead measurement."""

import argparse
import asyncio
import time
import statistics

import aiohttp


REQUEST_BODY = {
    "model": "db-openai-endpoint",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 100,
    "user": "new_user",
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


async def run_benchmark(url, n_requests, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)
    connector_limit = min(max_concurrent * 2, 200)
    connector = aiohttp.TCPConnector(
        limit=connector_limit,
        limit_per_host=max_concurrent,
        force_close=False,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        # warmup
        await asyncio.gather(*[send_request(session, url, semaphore) for _ in range(min(50, n_requests))])

        # timed run
        wall_start = time.perf_counter()
        results = await asyncio.gather(*[send_request(session, url, semaphore) for _ in range(n_requests)])
        wall_elapsed = time.perf_counter() - wall_start

    latencies = [r for r in results if r is not None]
    failures = sum(1 for r in results if r is None)

    if not latencies:
        return {
            "mean": 0, "p50": 0, "p95": 0, "p99": 0,
            "throughput": 0, "failures": n_requests,
            "wall_time": wall_elapsed, "n_requests": n_requests,
            "max_concurrent": max_concurrent, "latencies": [],
        }

    latencies.sort()
    n = len(latencies)
    mean = statistics.mean(latencies) * 1000
    p50 = latencies[n // 2] * 1000
    p95 = latencies[int(n * 0.95)] * 1000
    p99 = latencies[int(n * 0.99)] * 1000
    throughput = n_requests / wall_elapsed

    return {
        "mean": mean, "p50": p50, "p95": p95, "p99": p99,
        "throughput": throughput, "failures": failures,
        "wall_time": wall_elapsed, "n_requests": n_requests,
        "max_concurrent": max_concurrent, "latencies": latencies,
    }


def print_run_results(run_num, total_runs, result):
    label = f"  Run {run_num}/{total_runs}" if total_runs > 1 else "  Results"
    print(f"\n{'='*60}")
    print(label)
    print(f"{'='*60}")
    print(f"  Requests:    {result['n_requests']}  (failures: {result['failures']})")
    print(f"  Concurrency: {result['max_concurrent']}")
    print(f"  Wall time:   {result['wall_time']:.2f}s")
    print(f"  Throughput:  {result['throughput']:.0f} req/s")
    print(f"  Mean:        {result['mean']:.2f} ms")
    print(f"  P50:         {result['p50']:.2f} ms")
    print(f"  P95:         {result['p95']:.2f} ms")
    print(f"  P99:         {result['p99']:.2f} ms")


def print_aggregate(results):
    all_latencies = []
    for r in results:
        all_latencies.extend(r["latencies"])
    all_latencies.sort()

    total_failures = sum(r["failures"] for r in results)
    total_requests = sum(r["n_requests"] for r in results)
    n = len(all_latencies)

    if not all_latencies:
        print(f"\n  Aggregate: all {total_requests} requests failed across {len(results)} runs")
        return

    mean = statistics.mean(all_latencies) * 1000
    p50 = all_latencies[n // 2] * 1000
    p95 = all_latencies[int(n * 0.95)] * 1000
    p99 = all_latencies[int(n * 0.99)] * 1000
    avg_throughput = statistics.mean(r["throughput"] for r in results)

    print(f"\n{'='*60}")
    print(f"  Aggregate ({len(results)} runs, {total_requests} total requests)")
    print(f"{'='*60}")
    print(f"  Failures:    {total_failures}")
    print(f"  Throughput:  {avg_throughput:.0f} req/s (avg across runs)")
    print(f"  Mean:        {mean:.2f} ms")
    print(f"  P50:         {p50:.2f} ms")
    print(f"  P95:         {p95:.2f} ms")
    print(f"  P99:         {p99:.2f} ms")

    # Run-to-run variance
    run_means = [r["mean"] for r in results]
    run_throughputs = [r["throughput"] for r in results]
    if len(run_means) > 1:
        cov_latency = statistics.stdev(run_means) / statistics.mean(run_means) * 100
        cov_throughput = statistics.stdev(run_throughputs) / statistics.mean(run_throughputs) * 100
        print(f"\n  Run-to-run variance:")
        print(f"    Latency CoV:    {cov_latency:.1f}%")
        print(f"    Throughput CoV: {cov_throughput:.1f}%")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:4000/chat/completions")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--max-concurrent", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    print(f"Benchmarking {args.url}")
    print(f"  {args.requests} requests, {args.max_concurrent} concurrency, {args.runs} run(s)")

    results = []
    for run_num in range(1, args.runs + 1):
        result = await run_benchmark(args.url, args.requests, args.max_concurrent)
        results.append(result)
        print_run_results(run_num, args.runs, result)

    if args.runs > 1:
        print_aggregate(results)


if __name__ == "__main__":
    asyncio.run(main())
