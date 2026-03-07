"""Compare baseline vs sidecar locust load test results."""
import csv
import sys
import os


def parse_stats_csv(filepath):
    """Parse a locust stats CSV file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
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


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/load_tests/results"

    baseline = parse_stats_csv(os.path.join(results_dir, "baseline_stats.csv"))
    sidecar = parse_stats_csv(os.path.join(results_dir, "sidecar_stats.csv"))

    if not baseline:
        print("No baseline results found")
        return

    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'Sidecar':>12} {'Change':>12}")
    print("=" * 70)

    if sidecar:
        metrics = [
            ("Total Requests", "requests", ""),
            ("Failures", "failures", ""),
            ("Requests/sec", "rps", " rps"),
            ("Median (ms)", "median", " ms"),
            ("Average (ms)", "avg", " ms"),
            ("P50 (ms)", "p50", " ms"),
            ("P90 (ms)", "p90", " ms"),
            ("P95 (ms)", "p95", " ms"),
            ("P99 (ms)", "p99", " ms"),
            ("P99.9 (ms)", "p999", " ms"),
            ("Max (ms)", "max", " ms"),
        ]

        for label, key, unit in metrics:
            b_val = baseline[key]
            s_val = sidecar[key]
            if b_val > 0:
                if key in ("rps", "requests"):
                    change = ((s_val - b_val) / b_val) * 100
                else:
                    change = ((s_val - b_val) / b_val) * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "N/A"
            print(f"{label:<25} {b_val:>10.1f}{unit:>2} {s_val:>10.1f}{unit:>2} {change_str:>12}")
    else:
        print("Sidecar results not available yet. Baseline only:")
        for key, val in baseline.items():
            print(f"  {key}: {val}")

    print("=" * 70)


if __name__ == "__main__":
    main()
