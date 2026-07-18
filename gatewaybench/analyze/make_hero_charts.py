"""Generate the GatewayBench hero charts (CursorBench-styled scatter plots).

The numbers here are ILLUSTRATIVE PLACEHOLDERS to convey the chart design; they
are not measured. Replace `_DATA` with real rows from results/*.jsonl once the
harness has run, then regenerate:

    python analyze/make_hero_charts.py

Outputs analyze/throughput_vs_memory.png and analyze/ttft_vs_memory.png.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_OUT_DIR = Path(__file__).resolve().parent
_INK = "#1f2933"
_MUTED = "#6b7280"
_GRID = "#e5e7eb"


@dataclass(frozen=True)
class Point:
    label: str
    memory_mb: float
    throughput_rps: float
    ttft_overhead_ms: float
    color: str
    note: str = ""


# ILLUSTRATIVE PLACEHOLDER DATA -- not measured.
_DATA: tuple[Point, ...] = (
    Point("LiteLLM Proxy", 450, 1800, 12.0, "#0b7285"),
    Point("Bifrost (Go)", 180, 4200, 3.0, "#e8590c"),
    Point("Portkey (OSS)", 260, 2600, 7.0, "#ae3ec9"),
    Point(
        "LiteLLM SDK",
        120,
        6000,
        1.5,
        "#2f9e44",
        note="in-process, no HTTP hop",
    ),
)


def _style_axes(ax: "plt.Axes", title: str, ylabel: str, xlabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=_INK, pad=18)
    ax.set_ylabel(ylabel, fontsize=11, color=_MUTED)
    ax.set_xlabel(xlabel, fontsize=11, color=_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=10)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _annotate(ax: "plt.Axes", p: Point, dx: float, dy: float) -> None:
    text = p.label if not p.note else f"{p.label}"
    ax.annotate(
        text,
        xy=(p.memory_mb, _y(ax, p)),
        xytext=(p.memory_mb + dx, _y(ax, p) + dy),
        fontsize=11,
        color=_INK,
        fontweight="medium",
    )
    if p.note:
        ax.annotate(
            p.note,
            xy=(p.memory_mb, _y(ax, p)),
            xytext=(p.memory_mb + dx, _y(ax, p) + dy - _note_gap(ax)),
            fontsize=8.5,
            color=_MUTED,
            style="italic",
        )


_MODE = {"metric": "throughput_rps"}


def _y(ax: "plt.Axes", p: Point) -> float:
    return getattr(p, _MODE["metric"])


def _note_gap(ax: "plt.Axes") -> float:
    lo, hi = ax.get_ylim()
    return (hi - lo) * 0.045


def _frontier(ax: "plt.Axes", pts: list[Point], color: str) -> None:
    xs = [p.memory_mb for p in pts]
    ys = [_y(ax, p) for p in pts]
    ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.55, zorder=2)


def _scatter(pts: tuple[Point, ...]) -> None:
    for p in pts:
        ax = plt.gca()
        ax.scatter([p.memory_mb], [_y(ax, p)], s=90, color=p.color, zorder=3, edgecolors="white", linewidths=1.2)


def make_throughput_chart() -> Path:
    _MODE["metric"] = "throughput_rps"
    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(80, 520)
    ax.set_ylim(1000, 6800)
    _style_axes(
        ax,
        "GatewayBench: throughput vs memory",
        "sustained throughput (req/s)  higher is better ->",
        "peak resident memory (MB)  <- lower is better",
    )
    _scatter(_DATA)
    frontier_pts = sorted(_DATA, key=lambda p: p.memory_mb)
    _frontier(ax, frontier_pts, "#0b7285")
    offsets = {
        "LiteLLM Proxy": (10, 120),
        "Bifrost (Go)": (10, 180),
        "Portkey (OSS)": (14, 150),
        "LiteLLM SDK": (-40, -320),
    }
    for p in _DATA:
        dx, dy = offsets[p.label]
        _annotate(ax, p, dx, dy)
    ax.annotate(
        "efficiency frontier",
        xy=(120, 6000),
        xytext=(300, 6300),
        fontsize=12,
        color=_MUTED,
        fontweight="bold",
    )
    ax.annotate(
        "ILLUSTRATIVE PLACEHOLDER DATA -- not measured",
        xy=(0, 0),
        xytext=(0.99, -0.13),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color="#b91c1c",
        style="italic",
    )
    out = _OUT_DIR / "throughput_vs_memory.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_ttft_chart() -> Path:
    _MODE["metric"] = "ttft_overhead_ms"
    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(80, 520)
    ax.set_ylim(0, 14)
    _style_axes(
        ax,
        "GatewayBench: streaming TTFT overhead vs memory",
        "p99 added time-to-first-chunk (ms)  <- lower is better",
        "peak resident memory (MB)  <- lower is better",
    )
    _scatter(_DATA)
    frontier_pts = sorted(_DATA, key=lambda p: p.memory_mb)
    _frontier(ax, frontier_pts, "#0b7285")
    offsets = {
        "LiteLLM Proxy": (10, 0.4),
        "Bifrost (Go)": (10, 0.4),
        "Portkey (OSS)": (10, 0.4),
        "LiteLLM SDK": (-30, 0.6),
    }
    for p in _DATA:
        dx, dy = offsets[p.label]
        _annotate(ax, p, dx, dy)
    ax.annotate(
        "ideal: fast + lean (bottom-left)",
        xy=(120, 1.5),
        xytext=(150, 11.5),
        fontsize=11,
        color=_MUTED,
        fontweight="bold",
    )
    ax.annotate(
        "ILLUSTRATIVE PLACEHOLDER DATA -- not measured",
        xy=(0, 0),
        xytext=(0.99, -0.13),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color="#b91c1c",
        style="italic",
    )
    out = _OUT_DIR / "ttft_vs_memory.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    _ = font_manager  # ensure font cache import for consistent rendering
    t = make_throughput_chart()
    s = make_ttft_chart()
    print(f"wrote {t}")
    print(f"wrote {s}")


if __name__ == "__main__":
    main()
