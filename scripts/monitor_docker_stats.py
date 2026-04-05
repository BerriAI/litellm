#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Sample:
    timestamp: dt.datetime
    elapsed_s: float
    cpu_percent: float
    mem_used_bytes: float


UNIT_FACTORS = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
    "tb": 1000**4,
    "tib": 1024**4,
}


def parse_size_to_bytes(size_str: str) -> float:
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*", size_str)
    if match is None:
        raise ValueError(f"Unable to parse size: {size_str}")

    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit not in UNIT_FACTORS:
        raise ValueError(f"Unsupported size unit: {unit}")
    return value * UNIT_FACTORS[unit]


def bytes_to_gib(value_bytes: float) -> float:
    return value_bytes / (1024**3)


def run_stats_command(compose_cmd: str, service: str) -> str:
    cmd = compose_cmd.split() + ["stats", service, "--no-stream"]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        msg = stderr or stdout or "Unknown error"
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{msg}")
    return completed.stdout


def parse_stats_output(raw_output: str) -> tuple[float, float]:
    lines = [line for line in raw_output.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Unexpected docker stats output (no data row found)")

    data_line = lines[-1]

    cpu_match = re.search(r"([0-9]*\.?[0-9]+)%", data_line)
    mem_match = re.search(
        r"([0-9]*\.?[0-9]+[A-Za-z]+)\s*/\s*([0-9]*\.?[0-9]+[A-Za-z]+)",
        data_line,
    )
    if cpu_match is None or mem_match is None:
        raise ValueError(f"Unable to parse stats row: {data_line}")

    cpu_percent = float(cpu_match.group(1))
    mem_used_bytes = parse_size_to_bytes(mem_match.group(1))
    return cpu_percent, mem_used_bytes


def summarize(samples: List[Sample]) -> dict:
    if not samples:
        raise ValueError("No samples collected")

    cpu_values = [s.cpu_percent for s in samples]
    mem_gib_values = [bytes_to_gib(s.mem_used_bytes) for s in samples]

    return {
        "samples": len(samples),
        "duration_seconds": samples[-1].elapsed_s,
        "cpu_avg_percent": statistics.mean(cpu_values),
        "cpu_peak_percent": max(cpu_values),
        "cpu_variance": statistics.pvariance(cpu_values) if len(cpu_values) > 1 else 0.0,
        "mem_avg_gib": statistics.mean(mem_gib_values),
        "mem_peak_gib": max(mem_gib_values),
        "mem_variance_gib2": (
            statistics.pvariance(mem_gib_values) if len(mem_gib_values) > 1 else 0.0
        ),
    }


def print_summary(summary: dict) -> None:
    print("\n=== Docker Stats Summary ===")
    print(f"Samples: {summary['samples']}")
    print(f"Duration: {summary['duration_seconds']:.1f}s")
    print(f"Average CPU: {summary['cpu_avg_percent']:.2f}%")
    print(f"Peak CPU: {summary['cpu_peak_percent']:.2f}%")
    print(f"CPU Variance: {summary['cpu_variance']:.4f} (%^2)")
    print(f"Average Memory: {summary['mem_avg_gib']:.3f} GiB")
    print(f"Peak Memory: {summary['mem_peak_gib']:.3f} GiB")
    print(f"Memory Variance: {summary['mem_variance_gib2']:.6f} (GiB^2)")


def write_csv(samples: List[Sample], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "elapsed_seconds",
                "cpu_percent",
                "mem_used_gib",
            ]
        )
        for s in samples:
            writer.writerow(
                [
                    s.timestamp.isoformat(),
                    f"{s.elapsed_s:.3f}",
                    f"{s.cpu_percent:.4f}",
                    f"{bytes_to_gib(s.mem_used_bytes):.6f}",
                ]
            )


def plot_memory(samples: List[Sample], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    x = [s.elapsed_s for s in samples]
    y = [bytes_to_gib(s.mem_used_bytes) for s in samples]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory usage (GiB)")
    plt.title("Memory Usage vs Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def should_stop(start_monotonic: float, duration_s: Optional[float]) -> bool:
    if duration_s is not None and (time.monotonic() - start_monotonic) >= duration_s:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample 'docker compose stats <service> --no-stream' and summarize usage",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=45.0,
        help="Sampling interval in seconds (default: 45)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Total duration in seconds. If omitted, runs until Ctrl+C.",
    )
    parser.add_argument(
        "--service",
        type=str,
        default="litellm",
        help="Compose service name (default: litellm)",
    )
    parser.add_argument(
        "--compose-cmd",
        type=str,
        default="docker compose",
        help="Compose command prefix (default: 'docker compose')",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional output CSV path for raw samples",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional PNG path for memory-vs-time plot",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        print("--interval must be > 0", file=sys.stderr)
        return 2

    if args.duration is None:
        print("Running until Ctrl+C. Use --duration for automatic stop.")

    samples: List[Sample] = []
    start_monotonic = time.monotonic()
    next_run = start_monotonic

    try:
        while True:
            now = time.monotonic()
            if now < next_run:
                time.sleep(next_run - now)

            timestamp = dt.datetime.now(dt.timezone.utc)
            elapsed_s = time.monotonic() - start_monotonic

            try:
                output = run_stats_command(args.compose_cmd, args.service)
                cpu, mem_used_bytes = parse_stats_output(output)
                sample = Sample(
                    timestamp=timestamp,
                    elapsed_s=elapsed_s,
                    cpu_percent=cpu,
                    mem_used_bytes=mem_used_bytes,
                )
                samples.append(sample)
                print(
                    f"[{timestamp.isoformat()}] sample={len(samples)} "
                    f"cpu={cpu:.2f}% mem={bytes_to_gib(mem_used_bytes):.3f}GiB"
                )
            except Exception as e:
                print(f"Warning: failed to sample stats: {e}", file=sys.stderr)

            if should_stop(start_monotonic, args.duration):
                break

            next_run += args.interval

    except KeyboardInterrupt:
        print("\nStopping on Ctrl+C...")

    if not samples:
        print("No valid samples collected.", file=sys.stderr)
        return 1

    summary = summarize(samples)
    print_summary(summary)

    if args.csv:
        write_csv(samples, args.csv)
        print(f"Wrote samples CSV: {args.csv}")

    if args.plot:
        try:
            plot_memory(samples, args.plot)
            print(f"Wrote memory plot: {args.plot}")
        except Exception as e:
            print(f"Failed to generate plot: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
