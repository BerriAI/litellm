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
0.0,570.18,70,0,0.0
0.0,570.18,70,0,0.0
15.4,612.54,120,1501,97.6
30.4,608.96,120,3000,99.9
45.4,615.77,120,4550,103.3
60.4,609.82,118,6001,96.7
75.7,616.18,118,7550,101.3
90.7,610.00,118,9050,100.0
105.7,613.53,118,10600,103.3
120.7,608.31,118,12056,97.0
136.0,612.61,118,13602,101.3
151.0,607.01,118,15101,99.9
166.0,612.73,118,16650,103.2
181.0,618.67,118,18199,103.2
196.3,612.33,118,19699,97.7
211.3,616.50,118,21050,90.1
226.3,611.43,118,22550,100.0
241.3,617.59,118,24050,100.0
256.7,610.75,118,25591,100.4
271.7,617.40,118,27100,100.6
286.7,612.33,118,28599,99.9
301.7,614.95,68,29950,90.0
317.0,606.12,68,29950,0.0"""


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

    # Comparison stats
    before_final = before["rss_mb"][-1]
    after_final = after["rss_mb"][-1]
    before_peak = max(before["rss_mb"][2:])  # skip initial 0s
    after_peak = max(after["rss_mb"][2:])
    before_start = before["rss_mb"][0]
    after_start = after["rss_mb"][0]
    before_growth = before_final - before_start
    after_growth = after_final - after_start

    textstr = (
        f"Total RSS growth:\n"
        f"  Before: +{before_growth:.1f} MB (monotonic)\n"
        f"  After:  +{after_growth:.1f} MB (oscillating)\n"
        f"  Reduction: {(1 - after_growth/before_growth)*100:.0f}%\n\n"
        f"Peak RSS:\n"
        f"  Before: {before_peak:.1f} MB\n"
        f"  After:  {after_peak:.1f} MB\n"
        f"  Saved: {before_peak - after_peak:.1f} MB\n\n"
        f"Key: malloc_trim releases\n"
        f"memory back to OS every 10s"
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
