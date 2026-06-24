"""
Render the before/after timeline of TCPConnector slot occupancy captured by
aiohttp_pool_leak_repro.py. Produces a single PNG with two stacked subplots,
plus a small bar chart of recovery-probe outcome.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PHASE_COLORS = {
    "idle": "#cccccc",
    "leaking": "#f4a261",
    "recovery_probe": "#2a9d8f",
    "done": "#dddddd",
}


def _phase_spans(samples: list[dict]) -> list[tuple[str, float, float]]:
    spans: list[tuple[str, float, float]] = []
    if not samples:
        return spans
    current = samples[0]["phase"]
    start = samples[0]["t"]
    for s in samples[1:]:
        if s["phase"] != current:
            spans.append((current, start, s["t"]))
            current = s["phase"]
            start = s["t"]
    spans.append((current, start, samples[-1]["t"]))
    return spans


def _plot_scenario(ax, report: dict, title: str) -> None:
    samples = report["samples"]
    ts = [s["t"] for s in samples]
    acquired = [s["acquired"] for s in samples]
    limit = report["pool_limit"]

    for phase, lo, hi in _phase_spans(samples):
        ax.axvspan(lo, hi, color=PHASE_COLORS.get(phase, "#eeeeee"), alpha=0.35)

    ax.plot(ts, acquired, drawstyle="steps-post", color="#264653", linewidth=2)
    ax.axhline(
        limit,
        color="#e63946",
        linestyle="--",
        linewidth=1.2,
        label=f"pool_limit = {limit}",
    )
    ax.set_ylim(-0.3, limit + 0.7)
    ax.set_xlim(0, max(ts) if ts else 1.0)
    ax.set_ylabel("connector\nslots in use")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    probe_ok = report["recovery_ok"]
    probe_elapsed = report["recovery_elapsed_s"]
    err = report["recovery_error"]
    badge = (
        f"recovery probe: OK in {probe_elapsed*1000:.0f} ms"
        if probe_ok
        else f"recovery probe: FAILED after {probe_elapsed:.2f}s\n{err}"
    )
    ax.text(
        0.99,
        0.97,
        badge,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "#d8f3dc" if probe_ok else "#ffccd5",
            "edgecolor": "#888888",
            "boxstyle": "round,pad=0.3",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=pathlib.Path, required=True)
    parser.add_argument("--after", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())

    fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    _plot_scenario(
        ax_before,
        before,
        "Before #30271 (no finally block): TCPConnector slots leaked permanently",
    )
    _plot_scenario(
        ax_after,
        after,
        "After #30271 (finally: response.close()): slots released, pool recovers",
    )
    ax_after.set_xlabel("seconds since test start")

    legend_handles = [
        mpatches.Patch(
            color=PHASE_COLORS["leaking"],
            alpha=0.35,
            label="cancelled streams (1 per slot)",
        ),
        mpatches.Patch(
            color=PHASE_COLORS["recovery_probe"],
            alpha=0.35,
            label="recovery probe (normal GET)",
        ),
        mpatches.Patch(color="#e63946", label="pool limit"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "aiohttp connector pool leak repro (LiteLLMAiohttpTransport, pool=4)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
