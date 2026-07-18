"""Generate the GatewayBench hero chart: a simple overhead comparison.

Two horizontal-bar panels comparing gateways head-to-head on overhead. The
numbers here are ILLUSTRATIVE PLACEHOLDERS to convey the design; they are not
measured. Replace `_DATA` with real rows from results/*.jsonl once the harness
has run, then regenerate:

    python analyze/make_hero_charts.py

Outputs analyze/overhead_comparison.png.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_OUT_DIR = Path(__file__).resolve().parent
_INK = "#1f2933"
_MUTED = "#6b7280"
_GRID = "#e5e7eb"
_DARK = "#26272b"
_HILITE = "#f2530b"


@dataclass(frozen=True)
class Gateway:
    label: str
    added_latency_ms: float
    memory_mb: float
    is_litellm: bool


# ILLUSTRATIVE PLACEHOLDER DATA -- not measured.
_DATA: tuple[Gateway, ...] = (
    Gateway("LiteLLM (Rust)", 2.0, 90, True),
    Gateway("Bifrost", 3.0, 180, False),
    Gateway("Portkey", 7.0, 260, False),
    Gateway("LiteLLM (Python v1)", 12.0, 450, True),
)


def _panel(ax: "plt.Axes", values: list[float], labels: list[str], colors: list[str], xlabel: str) -> None:
    y = range(len(labels))
    ax.barh(list(y), values, color=colors, height=0.62, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11, color=_INK)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11, color=_MUTED, labelpad=10)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(axis="x", colors=_MUTED, labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.02, i, f"{v:g}", va="center", fontsize=10, color=_MUTED)


def make_chart() -> Path:
    ordered = sorted(_DATA, key=lambda g: g.added_latency_ms)
    labels = [g.label for g in ordered]
    colors = [_HILITE if g.is_litellm else _DARK for g in ordered]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=130)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Gateway overhead, lower is better",
        x=0.06,
        y=1.02,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=_INK,
    )
    _panel(ax_l, [g.added_latency_ms for g in ordered], labels, colors, "p99 added latency (ms)")
    _panel(
        ax_r,
        [g.memory_mb for g in ordered],
        ["" for _ in ordered],
        colors,
        "peak memory (MB)",
    )
    fig.text(
        0.99,
        -0.04,
        "ILLUSTRATIVE PLACEHOLDER DATA -- not measured",
        ha="right",
        fontsize=8.5,
        color="#b91c1c",
        style="italic",
    )
    out = _OUT_DIR / "overhead_comparison.png"
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    out = make_chart()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
