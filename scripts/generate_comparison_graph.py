#!/usr/bin/env python3
"""
Generate before/after comparison graph from memory leak repro CSV data.

Usage:
    poetry run python scripts/generate_comparison_graph.py
"""
import csv
import io
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


BEFORE_CSV = """elapsed_s,rss_mb,fd_count,requests_sent,rps
0.0,569.34,70,0,0.0
0.0,569.34,70,0,0.0
15.5,619.24,120,1450,93.8
30.5,620.84,120,2899,96.6
45.5,622.13,120,4299,93.3
60.5,621.38,118,5599,86.6
75.9,632.70,118,6950,87.4
90.9,633.59,118,8363,94.2
105.9,634.92,118,9800,95.8
120.9,636.50,118,11199,93.2
136.3,637.33,118,12609,91.6
151.3,639.59,118,14050,96.0
166.3,639.79,118,15499,96.6
181.3,639.81,118,16900,93.4
196.8,640.68,118,18236,86.2
211.9,640.88,118,19656,94.6
226.9,640.00,118,21101,96.3
241.9,640.02,118,22516,94.3
257.4,640.89,118,24031,97.5
272.4,641.12,118,25450,94.6
287.4,640.14,118,26900,96.6
302.4,640.16,68,28100,80.0"""

AFTER_CSV = """elapsed_s,rss_mb,fd_count,requests_sent,rps
0.0,568.89,70,0,0.0
0.0,568.89,70,0,0.0
15.5,610.83,120,1406,91.0
30.5,613.89,120,2800,92.9
45.5,615.44,120,4200,93.3
60.5,616.84,118,5650,96.6
75.9,628.53,118,7100,94.2
90.9,629.23,118,8450,90.0
105.9,630.47,118,9800,90.0
120.9,631.61,118,11200,93.2
136.2,631.72,121,12604,91.5
151.2,631.80,118,14018,94.2
166.2,632.07,118,15400,92.1
181.2,632.08,118,16821,94.7
196.8,633.13,118,18307,95.3
211.8,633.13,118,19713,93.7
226.8,633.16,118,21200,99.1
241.8,633.30,118,22600,93.3
257.2,634.55,118,24050,94.5
272.2,634.57,118,25481,95.4
287.2,634.77,118,26900,94.6
302.2,634.91,68,28150,83.3"""


def parse_csv(csv_text):
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    data = {"elapsed_s": [], "rss_mb": [], "requests_sent": []}
    for row in reader:
        elapsed = float(row["elapsed_s"])
        if elapsed == 0.0 and len(data["elapsed_s"]) > 0 and data["elapsed_s"][-1] == 0.0:
            continue  # skip duplicate 0.0 entry
        data["elapsed_s"].append(elapsed)
        data["rss_mb"].append(float(row["rss_mb"]))
        data["requests_sent"].append(int(row["requests_sent"]))
    return data


def main():
    before = parse_csv(BEFORE_CSV)
    after = parse_csv(AFTER_CSV)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("LiteLLM Memory Leak: Before vs After Fix", fontsize=14, fontweight="bold")

    # --- Plot 1: RSS over time ---
    ax1.plot(before["elapsed_s"], before["rss_mb"], "r-o", markersize=3, label="Before fix", linewidth=1.5)
    ax1.plot(after["elapsed_s"], after["rss_mb"], "g-o", markersize=3, label="After fix", linewidth=1.5)
    ax1.set_xlabel("Elapsed time (seconds)")
    ax1.set_ylabel("RSS Memory (MB)")
    ax1.set_title("RSS Memory Over Time (~90 rps sustained traffic)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotations for final values
    ax1.annotate(
        f"{before['rss_mb'][-1]:.1f} MB",
        xy=(before["elapsed_s"][-1], before["rss_mb"][-1]),
        xytext=(-60, 10), textcoords="offset points",
        fontsize=9, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
    )
    ax1.annotate(
        f"{after['rss_mb'][-1]:.1f} MB",
        xy=(after["elapsed_s"][-1], after["rss_mb"][-1]),
        xytext=(-60, -20), textcoords="offset points",
        fontsize=9, color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
    )

    # --- Plot 2: RSS over request count ---
    ax2.plot(before["requests_sent"], before["rss_mb"], "r-o", markersize=3, label="Before fix", linewidth=1.5)
    ax2.plot(after["requests_sent"], after["rss_mb"], "g-o", markersize=3, label="After fix", linewidth=1.5)
    ax2.set_xlabel("Requests sent")
    ax2.set_ylabel("RSS Memory (MB)")
    ax2.set_title("RSS Memory vs Request Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:.0f}K"))

    # Steady-state comparison box
    # Before: 636.50 → 640.16 over 120s-302s
    # After:  631.61 → 634.91 over 120s-302s
    before_steady = before["rss_mb"][-1] - before["rss_mb"][9]  # ~120s mark
    after_steady = after["rss_mb"][-1] - after["rss_mb"][9]   # ~120s mark
    improvement = (1 - after_steady / before_steady) * 100 if before_steady > 0 else 0

    textstr = (
        f"Steady-state growth (120s→300s):\n"
        f"  Before: +{before_steady:.1f} MB\n"
        f"  After:  +{after_steady:.1f} MB\n"
        f"  Improvement: {improvement:.0f}%\n\n"
        f"Peak RSS:\n"
        f"  Before: {max(before['rss_mb']):.1f} MB\n"
        f"  After:  {max(after['rss_mb']):.1f} MB\n"
        f"  Saved: {max(before['rss_mb']) - max(after['rss_mb']):.1f} MB"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=8,
             verticalalignment="top", bbox=props)

    plt.tight_layout()
    output_path = "/opt/cursor/artifacts/memory_leak_before_after.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Graph saved to {output_path}")

    # Also save to repo for commit
    repo_path = "/workspace/scripts/memory_leak_before_after.png"
    plt.savefig(repo_path, dpi=150, bbox_inches="tight")
    print(f"Graph also saved to {repo_path}")


if __name__ == "__main__":
    main()
