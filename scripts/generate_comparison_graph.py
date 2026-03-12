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
0.0,569.50,70,0,0.0
0.0,569.50,70,0,0.0
15.4,615.90,121,1471,95.4
30.4,617.38,121,2999,101.8
45.4,618.59,121,4501,100.1
60.4,619.44,119,6050,103.2
75.8,630.95,119,7600,101.0
90.8,631.15,119,9110,100.6
105.8,631.46,119,10650,102.6
120.8,631.48,119,12136,99.0
136.1,631.78,119,13700,101.9
151.1,632.49,119,15199,99.9
166.1,632.96,119,16526,88.4
181.2,633.00,119,17920,92.9
196.8,633.18,119,19400,94.8
211.8,633.19,119,20820,94.6
226.8,633.27,119,22197,91.8
241.8,633.72,119,23554,90.4
257.2,633.89,119,25099,100.2
272.2,634.10,119,26508,93.9
287.2,634.30,119,28037,101.9
302.2,634.48,69,29350,87.5"""

AFTER_CSV = """elapsed_s,rss_mb,fd_count,requests_sent,rps
0.0,569.42,70,0,0.0
0.0,569.42,70,0,0.0
15.4,610.18,120,1499,97.3
30.4,617.07,120,2999,100.0
45.4,605.94,120,4500,100.0
60.4,618.02,118,6003,100.2
76.0,629.87,118,7550,99.3
91.0,630.23,118,9050,100.0
106.0,606.46,118,10050,66.6
121.0,617.81,118,11399,89.8
136.4,630.02,118,12872,95.9
151.4,630.68,118,14300,95.2
166.4,607.14,118,15750,96.6
181.4,619.03,118,17296,103.0
196.8,631.25,118,18792,97.4
211.8,631.54,118,19950,77.2
226.8,605.76,118,21300,90.0
241.8,618.51,118,22801,100.0
257.2,631.75,118,24317,98.5
272.2,631.98,118,25800,98.8
287.2,632.02,118,27000,80.0
302.2,619.48,68,28300,86.6"""


def parse_csv(csv_text):
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    data = {"elapsed_s": [], "rss_mb": [], "requests_sent": [], "rps": []}
    for row in reader:
        elapsed = float(row["elapsed_s"])
        if elapsed == 0.0 and len(data["elapsed_s"]) > 0 and data["elapsed_s"][-1] == 0.0:
            continue  # skip duplicate 0.0 entry
        data["elapsed_s"].append(elapsed)
        data["rss_mb"].append(float(row["rss_mb"]))
        data["requests_sent"].append(int(row["requests_sent"]))
        data["rps"].append(float(row["rps"]))
    return data


def main():
    before = parse_csv(BEFORE_CSV)
    after = parse_csv(AFTER_CSV)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle("LiteLLM Memory Leak: Before vs After Fix (malloc_trim only, 60s interval)", fontsize=14, fontweight="bold")

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

    # --- Plot 3: RPS over time ---
    # Filter out 0 rps points (start/end)
    before_rps = [(t, r) for t, r in zip(before["elapsed_s"], before["rps"]) if r > 0]
    after_rps = [(t, r) for t, r in zip(after["elapsed_s"], after["rps"]) if r > 0]
    
    if before_rps:
        ax3.plot([t for t, _ in before_rps], [r for _, r in before_rps], "r-o", markersize=3, label="Before fix", linewidth=1.5)
    if after_rps:
        ax3.plot([t for t, _ in after_rps], [r for _, r in after_rps], "g-o", markersize=3, label="After fix", linewidth=1.5)
    ax3.set_xlabel("Elapsed time (seconds)")
    ax3.set_ylabel("Requests per second")
    ax3.set_title("RPS Over Time (throughput impact check)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(50, 120)

    # Add avg RPS annotations
    before_avg_rps = sum(r for _, r in before_rps) / len(before_rps) if before_rps else 0
    after_avg_rps = sum(r for _, r in after_rps) / len(after_rps) if after_rps else 0
    rps_diff = ((after_avg_rps - before_avg_rps) / before_avg_rps * 100) if before_avg_rps else 0
    
    rps_text = (
        f"Avg RPS:\n"
        f"  Before: {before_avg_rps:.1f}\n"
        f"  After:  {after_avg_rps:.1f}\n"
        f"  Impact: {rps_diff:+.1f}%"
    )
    ax3.text(0.02, 0.98, rps_text, transform=ax3.transAxes, fontsize=8,
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
