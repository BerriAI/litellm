"""Turns run directories from run_bench.py into per-run metrics, pairwise deltas with bootstrap CIs, and one HTML
report. Pairs are `old:new` variant names, e.g. `A0:A1`; an A/A pair (`A0:A0b`) gives the noise floor."""

from __future__ import annotations

import argparse
import html
import json
import random
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import takewhile
from pathlib import Path
from typing import Final

WARMUP_SEQ_START: Final = 900_000
BOOTSTRAP_ROUNDS: Final = 2000
RAMP_P99_LIMIT_S: Final = 0.5
RAMP_ERROR_LIMIT: Final = 0.005


@dataclass(frozen=True, slots=True)
class Step:
    rps: float
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    send_at_s: float
    ok: bool
    stream: bool
    rust: bool
    overhead_s: float | None
    ttft_overhead_s: float | None
    header_overhead_ms: float | None


@dataclass(frozen=True, slots=True)
class RunMetrics:
    variant: str
    run: str
    measured: int
    errors: int
    rust_share: float
    window_s: float
    cpu_ms_per_request: float
    loadgen_cpu_share: float
    peak_rss_mb: float
    rss_growth_mb: float
    prometheus_matches: bool
    s3_matches: bool
    overheads_s: tuple[float, ...]
    ttft_overheads_s: tuple[float, ...]
    header_overheads_ms: tuple[float, ...]
    sustained_rps: float | None


def _interpolate(samples: Sequence[Mapping[str, float]], key: str, at: float) -> float:
    before: Final = max((s for s in samples if s["at"] <= at), key=lambda s: s["at"], default=samples[0])
    after: Final = min((s for s in samples if s["at"] >= at), key=lambda s: s["at"], default=samples[-1])
    if after["at"] == before["at"]:
        return before[key]
    weight: Final = (at - before["at"]) / (after["at"] - before["at"])
    return before[key] + weight * (after[key] - before[key])


def percentile(values: Sequence[float], q: float) -> float:
    ordered: Final = sorted(values)
    if not ordered:
        return float("nan")
    rank: Final = (len(ordered) - 1) * q / 100
    low: Final = int(rank)
    high: Final = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def request_metrics(
    record: Mapping[str, object], send_at_s: float, mock: Mapping[str, Mapping[str, float]]
) -> RequestMetrics:
    upstream: Final = mock.get(str(record["marker"]))
    ok: Final = record["ok"] is True
    total: Final = float(str(record["total_s"]))
    first_token: Final = record.get("first_token_s")
    header_ms: Final = record.get("litellm_overhead_ms")
    return RequestMetrics(
        send_at_s=send_at_s,
        ok=ok,
        stream=record["stream"] is True,
        rust=record.get("litellm_rust") is True,
        overhead_s=None if upstream is None or not ok else total - upstream["delay_s"] - upstream["generation_s"],
        ttft_overhead_s=(
            float(str(first_token)) - upstream["delay_s"]
            if upstream is not None and ok and record["stream"] is True and first_token is not None
            else None
        ),
        header_overhead_ms=float(str(header_ms)) if isinstance(header_ms, (int, float)) else None,
    )


def sustained(requests: Sequence[RequestMetrics], steps: Sequence[Step]) -> float | None:
    """The highest ramp step that, with every step before it, kept p99 overhead and errors under the limits."""
    held: Final = tuple(
        takewhile(
            lambda step: _step_holds(tuple(r for r in requests if step.start_s <= r.send_at_s < step.end_s)), steps
        )
    )
    return held[-1].rps if held else None


def _step_holds(requests: Sequence[RequestMetrics]) -> bool:
    if not requests:
        return False
    errors: Final = sum(1 for r in requests if not r.ok) / len(requests)
    overheads: Final = tuple(r.overhead_s for r in requests if r.overhead_s is not None)
    return errors <= RAMP_ERROR_LIMIT and percentile(overheads, 99) <= RAMP_P99_LIMIT_S


def plan_steps(plan_path: Path) -> tuple[Step, ...]:
    return tuple(Step(**step) for step in json.loads(plan_path.with_suffix(".steps.json").read_text()))


def run_metrics(run_dir: Path) -> RunMetrics:
    meta: Final = json.loads((run_dir / "meta.json").read_text())
    plan: Final = json.loads(Path(meta["plan"]).read_text())
    send_at: Final = {int(item["seq"]): float(item["send_at_s"]) for item in plan}
    records: Final = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines() if line]
    mock: Final = {
        str(entry["marker"]): {"delay_s": float(entry["delay_s"]), "generation_s": float(entry.get("generation_s", 0))}
        for entry in json.loads((run_dir / "mock.json").read_text())["requests"]
        if entry["marker"]
    }
    samples: Final = json.loads((run_dir / "samples.json").read_text())
    t0: Final = statistics.median(float(r["started_at"]) - send_at[int(r["seq"])] for r in records)
    measured_sends: Final = tuple(s for seq, s in send_at.items() if seq < WARMUP_SEQ_START)
    start: Final = t0 + min(measured_sends)
    end: Final = t0 + max(measured_sends)
    measured: Final = tuple(
        request_metrics(r, send_at[int(r["seq"])], mock) for r in records if int(r["seq"]) < WARMUP_SEQ_START
    )
    window_s: Final = end - start
    cpu_s: Final = _interpolate(samples, "proxy_cpu_s", end) - _interpolate(samples, "proxy_cpu_s", start)
    loadgen_s: Final = _interpolate(samples, "loadgen_cpu_s", end) - _interpolate(samples, "loadgen_cpu_s", start)
    ok: Final = tuple(r for r in measured if r.ok)
    window_rss: Final = tuple(s["proxy_rss_bytes"] for s in samples if start <= s["at"] <= end)
    steps: Final = plan_steps(Path(meta["plan"]))
    return RunMetrics(
        variant=str(meta["variant"]),
        run=run_dir.name,
        measured=len(measured),
        errors=len(measured) - len(ok),
        rust_share=sum(1 for r in ok if r.rust) / max(1, len(ok)),
        window_s=window_s,
        cpu_ms_per_request=1000 * cpu_s / max(1, sum(1 for r in ok)),
        loadgen_cpu_share=loadgen_s / window_s if window_s else float("nan"),
        peak_rss_mb=max(window_rss, default=0) / 2**20,
        rss_growth_mb=(window_rss[-1] - window_rss[0]) / 2**20 if window_rss else 0.0,
        prometheus_matches=round(float(meta["prometheus_requests_delta"])) == int(meta["ok"]),
        s3_matches=int(meta["s3_objects"]) >= int(meta["ok"]),
        overheads_s=tuple(r.overhead_s for r in ok if r.overhead_s is not None),
        ttft_overheads_s=tuple(r.ttft_overhead_s for r in ok if r.ttft_overhead_s is not None),
        header_overheads_ms=tuple(r.header_overhead_ms for r in ok if r.header_overhead_ms is not None),
        sustained_rps=sustained(measured, steps) if len(steps) > 1 else None,
    )


RunStat = Callable[[Sequence[RunMetrics], random.Random | None], float]


def _run_level(pick: Callable[[RunMetrics], float]) -> RunStat:
    def stat(runs: Sequence[RunMetrics], _rng: random.Random | None) -> float:
        return statistics.fmean(pick(run) for run in runs)

    return stat


def _pooled_percentile(pick: Callable[[RunMetrics], Sequence[float]], q: float) -> RunStat:
    """Percentile of requests pooled across runs; with an rng, requests are also resampled within each run."""

    def stat(runs: Sequence[RunMetrics], rng: random.Random | None) -> float:
        pooled: Final = tuple(
            value for run in runs for value in (pick(run) if rng is None else rng.choices(pick(run), k=len(pick(run))))
        )  # comprehension-ok: hierarchical resample flattens runs into one pool
        return percentile(pooled, q) * 1000

    return stat


METRICS: Final[Mapping[str, RunStat]] = {
    "cpu_ms_per_request": _run_level(lambda run: run.cpu_ms_per_request),
    "overhead_p50_ms": _pooled_percentile(lambda run: run.overheads_s, 50),
    "overhead_p95_ms": _pooled_percentile(lambda run: run.overheads_s, 95),
    "overhead_p99_ms": _pooled_percentile(lambda run: run.overheads_s, 99),
    "ttft_overhead_p50_ms": _pooled_percentile(lambda run: run.ttft_overheads_s, 50),
    "ttft_overhead_p99_ms": _pooled_percentile(lambda run: run.ttft_overheads_s, 99),
    "header_overhead_p50_ms": _pooled_percentile(lambda run: tuple(v / 1000 for v in run.header_overheads_ms), 50),
    "peak_rss_mb": _run_level(lambda run: run.peak_rss_mb),
}


@dataclass(frozen=True, slots=True)
class Delta:
    metric: str
    old: float
    new: float
    change_pct: float
    ci_low_pct: float
    ci_high_pct: float


def delta(metric: str, old: Sequence[RunMetrics], new: Sequence[RunMetrics], seed: int) -> Delta:
    stat: Final = METRICS[metric]
    rng: Final = random.Random(seed)
    point_old: Final = stat(old, None)
    point_new: Final = stat(new, None)
    ratios: Final = sorted(
        stat(rng.choices(new, k=len(new)), rng) / stat(rng.choices(old, k=len(old)), rng) - 1
        for _ in range(BOOTSTRAP_ROUNDS)
    )
    return Delta(
        metric=metric,
        old=point_old,
        new=point_new,
        change_pct=100 * (point_new / point_old - 1),
        ci_low_pct=100 * percentile(ratios, 2.5),
        ci_high_pct=100 * percentile(ratios, 97.5),
    )


def _cell(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def html_report(runs: Sequence[RunMetrics], deltas: Mapping[str, Sequence[Delta]], noise: Mapping[str, float]) -> str:
    run_rows: Final = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(v))}</td>"
            for v in (
                run.run,
                run.measured,
                run.errors,
                _cell(run.rust_share),
                _cell(run.cpu_ms_per_request),
                _cell(percentile(run.overheads_s, 50) * 1000, 1),
                _cell(percentile(run.overheads_s, 99) * 1000, 1),
                _cell(run.loadgen_cpu_share * 100, 0),
                _cell(run.peak_rss_mb, 0),
                run.prometheus_matches and run.s3_matches,
                run.sustained_rps if run.sustained_rps is not None else "",
            )
        )
        + "</tr>"
        for run in runs
    )
    pair_tables: Final = "".join(
        f"<h2>{html.escape(pair)}</h2><table><tr><th>metric</th><th>old</th><th>new</th><th>change</th>"
        f"<th>95% CI</th><th>A/A floor</th><th>verdict</th></tr>"
        + "".join(
            f"<tr><td>{d.metric}</td><td>{_cell(d.old)}</td><td>{_cell(d.new)}</td><td>{d.change_pct:+.1f}%</td>"
            f"<td>{d.ci_low_pct:+.1f}% to {d.ci_high_pct:+.1f}%</td><td>±{noise.get(d.metric, float('nan')):.1f}%</td>"
            f"<td>{verdict(d, noise.get(d.metric))}</td></tr>"
            for d in pair_deltas
        )
        + "</table>"
        for pair, pair_deltas in deltas.items()
    )
    return (
        "<!doctype html><meta charset=utf-8><title>Rust vs Python bench</title>"
        "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;margin:8px 0 24px}"
        "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}</style>"
        f"{pair_tables}<h2>Runs</h2><table><tr><th>run</th><th>measured</th><th>errors</th><th>rust header share</th>"
        "<th>cpu ms/req</th><th>p50 ms</th><th>p99 ms</th><th>loadgen cpu %</th><th>peak rss MB</th>"
        f"<th>side effects ok</th><th>sustained rps</th></tr>{run_rows}</table>"
    )


def verdict(d: Delta, floor: float | None) -> str:
    if d.ci_low_pct == d.ci_high_pct:
        return "single run, no CI"
    excludes_zero: Final = d.ci_low_pct > 0 or d.ci_high_pct < 0
    above_floor: Final = floor is None or abs(d.change_pct) > floor
    if not (excludes_zero and above_floor):
        return "no change"
    return "higher" if d.change_pct > 0 else "lower"


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("results", type=Path)
    _ = parser.add_argument("--pairs", required=True, help="comma separated old:new variant pairs")
    _ = parser.add_argument("--noise-pair", help="old:new A/A pair whose |change| is the noise floor")
    _ = parser.add_argument("--seed", type=int, default=7)
    args: Final = parser.parse_args()
    results: Final[Path] = args.results  # pyright: ignore[reportAny]
    runs: Final = tuple(run_metrics(path) for path in sorted(results.iterdir()) if (path / "meta.json").exists())
    by_variant: Final = {v: tuple(r for r in runs if r.variant == v) for v in {r.variant for r in runs}}
    pairs: Final = tuple(tuple(pair.split(":")) for pair in str(args.pairs).split(","))  # pyright: ignore[reportAny]
    noise_spec: Final[str | None] = args.noise_pair  # pyright: ignore[reportAny]
    noise_deltas: Final = (
        tuple(
            delta(m, by_variant[noise_spec.split(":")[0]], by_variant[noise_spec.split(":")[1]], int(args.seed))
            for m in METRICS
        )  # pyright: ignore[reportAny]
        if noise_spec
        else ()
    )
    noise: Final = {d.metric: max(abs(d.ci_low_pct), abs(d.ci_high_pct)) for d in noise_deltas}
    deltas: Final = {
        f"{old} -> {new}": tuple(delta(m, by_variant[old], by_variant[new], int(args.seed)) for m in METRICS)  # pyright: ignore[reportAny]
        for old, new in pairs
    }
    _ = (results / "report.json").write_text(
        json.dumps(
            {
                "noise_floor_pct": noise,
                "deltas": {pair: [asdict(d) for d in ds] for pair, ds in deltas.items()},
                "runs": [
                    {k: v for k, v in asdict(r).items() if not k.endswith("_s") and not k.endswith("_ms")} for r in runs
                ],
            },
            indent=2,
        )
    )
    _ = (results / "report.html").write_text(html_report(runs, deltas, noise))
    for pair, ds in deltas.items():
        _ = sys.stdout.write(pair + "\n")
        for d in ds:
            _ = sys.stdout.write(
                f"  {d.metric:24} {d.old:9.2f} -> {d.new:9.2f}  {d.change_pct:+6.1f}%  CI [{d.ci_low_pct:+.1f}, {d.ci_high_pct:+.1f}]  {verdict(d, noise.get(d.metric))}"
                + "\n"
            )


if __name__ == "__main__":
    main()
