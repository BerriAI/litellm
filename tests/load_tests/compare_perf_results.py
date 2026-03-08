"""Compare baseline vs optimized locust load test results."""
import csv
import sys
import os


def parse_stats_csv(filepath):
    """Parse a locust stats CSV file."""
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") == "Aggregated":
                return {
                    "requests": int(row.get("Request Count", 0)),
                    "failures": int(row.get("Failure Count", 0)),
                    "median": float(row.get("Median Response Time", 0)),
                    "avg": float(row.get("Average Response Time", 0)),
                    "min": float(row.get("Min Response Time", 0)),
                    "max": float(row.get("Max Response Time", 0)),
                    "p50": float(row.get("50%", 0)),
                    "p66": float(row.get("66%", 0)),
                    "p75": float(row.get("75%", 0)),
                    "p80": float(row.get("80%", 0)),
                    "p90": float(row.get("90%", 0)),
                    "p95": float(row.get("95%", 0)),
                    "p98": float(row.get("98%", 0)),
                    "p99": float(row.get("99%", 0)),
                    "p999": float(row.get("99.9%", 0)),
                    "p9999": float(row.get("99.99%", 0)),
                    "rps": float(row.get("Requests/s", 0)),
                }
    return None


def print_comparison(label, baseline, optimized):
    print()
    print(f"  {label}")
    print("=" * 74)
    print(f"{'Metric':<25} {'Baseline':>12} {'Optimized':>12} {'Change':>12}  {'':>2}")
    print("=" * 74)

    metrics = [
        ("Total Requests", "requests", "", False),
        ("Failures", "failures", "", False),
        ("Failure %", None, "%", False),
        ("Requests/sec", "rps", " rps", True),
        ("Median (ms)", "median", " ms", False),
        ("Average (ms)", "avg", " ms", False),
        ("P50 (ms)", "p50", " ms", False),
        ("P75 (ms)", "p75", " ms", False),
        ("P90 (ms)", "p90", " ms", False),
        ("P95 (ms)", "p95", " ms", False),
        ("P98 (ms)", "p98", " ms", False),
        ("P99 (ms)", "p99", " ms", False),
        ("P99.9 (ms)", "p999", " ms", False),
        ("Max (ms)", "max", " ms", False),
    ]

    for label_m, key, unit, higher_is_better in metrics:
        if key is None:
            b_val = (baseline["failures"] / baseline["requests"] * 100) if baseline["requests"] else 0
            o_val = (optimized["failures"] / optimized["requests"] * 100) if optimized["requests"] else 0
        else:
            b_val = baseline[key]
            o_val = optimized[key]

        if b_val > 0:
            change = ((o_val - b_val) / b_val) * 100
            change_str = f"{change:+.1f}%"
            if higher_is_better:
                indicator = "+" if change > 2 else ("~" if change > -2 else "-")
            else:
                if key in ("requests", "failures", None):
                    indicator = ""
                else:
                    indicator = "+" if change < -2 else ("~" if change < 2 else "-")
        elif o_val == 0 and b_val == 0:
            change_str = "0.0%"
            indicator = "~"
        else:
            change_str = "N/A"
            indicator = ""
        print(
            f"{label_m:<25} {b_val:>10.1f}{unit:>2} {o_val:>10.1f}{unit:>2} {change_str:>10}  {indicator}"
        )

    print("=" * 74)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/load_tests/results"

    # 5000 user comparison
    baseline_5k = parse_stats_csv(os.path.join(results_dir, "baseline_stats.csv"))
    optimized_5k = parse_stats_csv(os.path.join(results_dir, "optimized_stats.csv"))

    # 2000 user comparison
    baseline_2k = parse_stats_csv(os.path.join(results_dir, "baseline_2k_stats.csv"))
    optimized_2k = parse_stats_csv(os.path.join(results_dir, "optimized_2k_stats.csv"))

    print()
    print("=" * 74)
    print("  LOCUST BENCHMARK: BASELINE vs OPTIMIZED (GIL perf fixes)")
    print("  60s run time per test, warmed up proxy, mock LLM backend")

    if baseline_5k and optimized_5k:
        print_comparison("5,000 CONCURRENT USERS (saturation test)", baseline_5k, optimized_5k)

    if baseline_2k and optimized_2k:
        print_comparison("2,000 CONCURRENT USERS (sub-saturation test)", baseline_2k, optimized_2k)

    if baseline_5k and optimized_5k:
        rps_5k = ((optimized_5k["rps"] - baseline_5k["rps"]) / baseline_5k["rps"]) * 100
        p50_5k = ((optimized_5k["p50"] - baseline_5k["p50"]) / baseline_5k["p50"]) * 100 if baseline_5k["p50"] else 0

        print()
        print("SUMMARY")
        print("-" * 74)
        print(f"  5k users: RPS {rps_5k:+.1f}% | P50 {p50_5k:+.1f}%")

    if baseline_2k and optimized_2k:
        rps_2k = ((optimized_2k["rps"] - baseline_2k["rps"]) / baseline_2k["rps"]) * 100
        p50_2k = ((optimized_2k["p50"] - baseline_2k["p50"]) / baseline_2k["p50"]) * 100 if baseline_2k["p50"] else 0
        p98_2k = ((optimized_2k["p98"] - baseline_2k["p98"]) / baseline_2k["p98"]) * 100 if baseline_2k["p98"] else 0
        print(f"  2k users: RPS {rps_2k:+.1f}% | P50 {p50_2k:+.1f}% | P98 {p98_2k:+.1f}%")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
