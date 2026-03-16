#!/usr/bin/env python3
"""Generate a comparison report from benchmark JSON files."""

import json
import sys


def load_results(path):
    with open(path) as f:
        return json.load(f)


def main():
    standard = load_results("benchmark_results/standard_comprehensive.json")
    fast = load_results("benchmark_results/fast_comprehensive.json")

    print("=" * 80)
    print("  BENCHMARK REPORT: Standard LiteLLM Proxy vs fast-litellm Accelerated Proxy")
    print("=" * 80)
    print()
    print("  Test Configuration:")
    print(f"    - Requests per concurrency level: {standard['requests_per_level']}")
    print(f"    - Runs per level (median taken):  {standard['runs_per_level']}")
    print(f"    - Mode: network_mock (pure proxy overhead, no real API calls)")
    print(f"    - Database: Local PostgreSQL (eliminates network DB latency)")
    print(f"    - Workers: 4 uvicorn workers")
    print()

    concurrency_levels = ["10", "50", "100", "200"]

    print("  " + "-" * 76)
    print(f"  {'Metric':<22} | {'Conc':>5} | {'Standard':>12} | {'fast-litellm':>14} | {'Diff':>10}")
    print("  " + "-" * 76)

    for conc in concurrency_levels:
        std_r = standard["results"][conc]
        fast_r = fast["results"][conc]

        metrics = [
            ("Throughput", "throughput_rps", "rps", False),
            ("Mean latency", "mean_ms", "ms", True),
            ("P50 latency", "p50_ms", "ms", True),
            ("P95 latency", "p95_ms", "ms", True),
            ("P99 latency", "p99_ms", "ms", True),
        ]

        for label, key, unit, lower_better in metrics:
            std_val = std_r[key]
            fast_val = fast_r[key]
            if std_val > 0:
                pct = ((fast_val - std_val) / std_val) * 100
                direction = "faster" if (pct < 0 and lower_better) or (pct > 0 and not lower_better) else "slower"
                diff_str = f"{pct:+.1f}%"
            else:
                diff_str = "N/A"

            print(f"  {label:<22} | {conc:>5} | {std_val:>9.1f} {unit:<3}| {fast_val:>11.1f} {unit:<3}| {diff_str:>10}")

        print("  " + "-" * 76)

    print()
    print("  SUMMARY TABLE (Throughput & Mean Latency)")
    print()
    print(f"  {'Concurrency':>12} | {'Std Throughput':>15} | {'Fast Throughput':>16} | {'Std Mean':>10} | {'Fast Mean':>11} | {'Speedup':>8}")
    print(f"  {'-'*12}-+-{'-'*15}-+-{'-'*16}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}")

    for conc in concurrency_levels:
        std_r = standard["results"][conc]
        fast_r = fast["results"][conc]

        std_tp = std_r["throughput_rps"]
        fast_tp = fast_r["throughput_rps"]
        speedup = fast_tp / std_tp if std_tp > 0 else 0

        print(f"  {conc:>12} | {std_tp:>12.0f} rps | {fast_tp:>13.0f} rps | "
              f"{std_r['mean_ms']:>7.1f} ms | {fast_r['mean_ms']:>8.1f} ms | {speedup:>6.2f}x")

    print()
    print("  KEY FINDINGS:")

    avg_std_tp = sum(standard["results"][c]["throughput_rps"] for c in concurrency_levels) / len(concurrency_levels)
    avg_fast_tp = sum(fast["results"][c]["throughput_rps"] for c in concurrency_levels) / len(concurrency_levels)
    overall_speedup = avg_fast_tp / avg_std_tp if avg_std_tp > 0 else 0

    avg_std_mean = sum(standard["results"][c]["mean_ms"] for c in concurrency_levels) / len(concurrency_levels)
    avg_fast_mean = sum(fast["results"][c]["mean_ms"] for c in concurrency_levels) / len(concurrency_levels)
    latency_diff_pct = ((avg_fast_mean - avg_std_mean) / avg_std_mean) * 100

    print(f"    - Overall throughput ratio: {overall_speedup:.2f}x (fast-litellm / standard)")
    print(f"    - Avg mean latency: standard={avg_std_mean:.1f}ms, fast-litellm={avg_fast_mean:.1f}ms ({latency_diff_pct:+.1f}%)")
    print()
    print("  NOTES:")
    print("    - network_mock mode eliminates real API calls, measuring pure proxy overhead")
    print("    - Local PostgreSQL eliminates network DB latency from the measurement")
    print("    - fast-litellm v0.1.6 with Rust acceleration via PyO3")
    print("    - Results may vary depending on hardware, OS, and Python version")
    print("    - fast-litellm patches: routing, token_counting, rate_limiting, connection_pooling")
    print("    - Some patches (SimpleRateLimiter, SimpleConnectionPool, count_tokens_batch)")
    print("      failed to apply since the target classes/functions were not found in this version")


if __name__ == "__main__":
    main()
