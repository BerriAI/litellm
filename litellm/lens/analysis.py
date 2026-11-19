from __future__ import annotations

import json
import math
import shlex
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from statistics import fmean
from typing import Final

from pydantic import TypeAdapter
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from litellm.lens.core import CaseRow, EventRow, RunStore, TargetInfo
from litellm.lens.terminal import ACCENT, BORDER, GREEN, INK, MUTED, money


@dataclass(frozen=True, slots=True)
class TargetMetrics:
    target: str
    attempts: int
    passed: int
    failed: int
    errors: int
    pass_rate: float | None
    mean_agent_cost: float | None
    mean_duration: float | None
    p95_duration: float | None
    cost_per_pass: float | None
    known_cost_count: int
    missing_cost_count: int
    telemetry_complete: bool


@dataclass(frozen=True, slots=True)
class ToolMetrics:
    target: str
    tool: str
    attempts: int
    calls: int
    calls_per_attempt: float
    errors: int
    repeated_attempts: int
    extra_calls: int
    extra_call_rate: float
    telemetry_complete: bool


@dataclass(frozen=True, slots=True)
class RepeatExample:
    case: CaseRow
    tool: str
    arguments: str
    calls: int


@dataclass(frozen=True, slots=True)
class ToolReport:
    metrics: tuple[ToolMetrics, ...]
    repeats: tuple[RepeatExample, ...]
    failures: tuple[CaseRow, ...]
    unobserved_targets: tuple[str, ...]
    unmatched_errors: int


@dataclass(frozen=True, slots=True)
class _ToolUse:
    case: CaseRow
    call: EventRow
    arguments: str | None
    error: bool


@dataclass(frozen=True, slots=True)
class _Attempt:
    case: CaseRow
    events: tuple[EventRow, ...]
    uses: tuple[_ToolUse, ...]
    repeats: tuple[RepeatExample, ...]


def _target_metrics(target: TargetInfo, rows: tuple[CaseRow, ...]) -> TargetMetrics:
    selected: Final = tuple(row for row in rows if row.target == target.label)
    durations: Final = tuple(sorted(row.duration for row in selected))
    costs: Final = tuple(row.agent_cost for row in selected if row.agent_cost is not None)
    passed: Final = sum(row.status == "pass" for row in selected)
    total_cost: Final = sum(costs) if selected and len(costs) == len(selected) else None
    return TargetMetrics(
        target.label,
        len(selected),
        passed,
        sum(row.status == "fail" for row in selected),
        sum(row.status == "error" for row in selected),
        passed / len(selected) if selected else None,
        total_cost / len(selected) if total_cost is not None else None,
        fmean(durations) if durations else None,
        durations[math.ceil(len(durations) * 0.95) - 1] if durations else None,
        total_cost / passed if total_cost is not None and passed else None,
        len(costs),
        len(selected) - len(costs),
        target.telemetry_complete,
    )


def target_metrics(store: RunStore) -> tuple[TargetMetrics, ...]:
    rows: Final = store.rows()
    return tuple(_target_metrics(target, rows) for target in store.manifest.targets)


def quality_extremes(
    metrics: tuple[TargetMetrics, ...],
) -> tuple[tuple[TargetMetrics, ...], tuple[TargetMetrics, ...]]:
    measured: Final = tuple(metric for metric in metrics if metric.pass_rate is not None)
    rates: Final = tuple(metric.pass_rate for metric in measured if metric.pass_rate is not None)
    if not rates:
        return (), ()
    return (
        tuple(metric for metric in measured if metric.pass_rate == max(rates)),
        tuple(metric for metric in measured if metric.pass_rate == min(rates)),
    )


def _case_rank(case: CaseRow) -> tuple[int, float, str, str, int]:
    status: Final = {"pass": 0, "fail": 1, "error": 2}
    return (
        status.get(case.status, 2),
        case.duration,
        case.task_id,
        case.target,
        case.trial,
    )


def ranked_cases(store: RunStore) -> tuple[CaseRow, ...]:
    return tuple(sorted(store.rows(), key=_case_rank))


def _arguments(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed: Final = TypeAdapter(object).validate_json(value)
    except ValueError:
        return None
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _failed_call(event: EventRow, calls: tuple[EventRow, ...]) -> int | None:
    if event.kind != "ToolResult" or not event.is_error or event.call_id is None:
        return None
    return next(
        (call.sequence for call in reversed(calls) if call.call_id == event.call_id and call.sequence < event.sequence),
        None,
    )


def _attempt(case: CaseRow, events: tuple[EventRow, ...]) -> _Attempt:
    calls: Final = tuple(event for event in events if event.kind == "ToolCall")
    failed: Final = frozenset(sequence for event in events if (sequence := _failed_call(event, calls)) is not None)
    uses: Final = tuple(_ToolUse(case, call, _arguments(call.arguments), call.sequence in failed) for call in calls)
    counts: Final = Counter((use.call.name or "unknown", use.arguments) for use in uses if use.arguments is not None)
    repeats: Final = tuple(
        RepeatExample(case, tool, arguments, count) for (tool, arguments), count in counts.items() if count > 1
    )
    return _Attempt(case, events, uses, repeats)


def _tool_metric(
    target: TargetInfo,
    tool: str | None,
    attempts: tuple[_Attempt, ...],
    uses: tuple[_ToolUse, ...],
) -> ToolMetrics:
    target_uses: Final = tuple(use for use in uses if use.case.target == target.label)
    selected: Final = (
        target_uses if tool is None else tuple(use for use in target_uses if (use.call.name or "unknown") == tool)
    )
    cases: Final = tuple(attempt for attempt in attempts if attempt.case.target == target.label)
    groups: Final = tuple(attempt.repeats for attempt in cases)
    repeats: Final = tuple(example for example in chain.from_iterable(groups) if tool is None or example.tool == tool)
    extra_calls: Final = sum(example.calls - 1 for example in repeats)
    return ToolMetrics(
        target.label,
        tool if tool is not None else "All tools",
        len(cases),
        len(selected),
        len(selected) / len(cases),
        sum(use.error for use in selected),
        len(frozenset(example.case.run_id for example in repeats)),
        extra_calls,
        extra_calls / len(selected) if selected else 0,
        target.telemetry_complete,
    )


def _target_tools(
    target: TargetInfo, attempts: tuple[_Attempt, ...], uses: tuple[_ToolUse, ...]
) -> tuple[ToolMetrics, ...]:
    cases: Final = tuple(attempt for attempt in attempts if attempt.case.target == target.label)
    selected: Final = tuple(use for use in uses if use.case.target == target.label)
    if not cases or (not selected and not target.telemetry_complete):
        return ()
    names: Final = tuple(dict.fromkeys(use.call.name or "unknown" for use in selected))
    return (_tool_metric(target, None, attempts, uses),) + tuple(
        _tool_metric(target, name, attempts, uses) for name in names
    )


def tool_report(store: RunStore) -> ToolReport:
    attempts: Final = tuple(_attempt(case, store.events(case.run_id)) for case in store.rows())
    uses: Final = tuple(chain.from_iterable(attempt.uses for attempt in attempts))
    metrics: Final = tuple(
        chain.from_iterable(_target_tools(target, attempts, uses) for target in store.manifest.targets)
    )
    repeats: Final = tuple(chain.from_iterable(attempt.repeats for attempt in attempts))
    return ToolReport(
        metrics,
        repeats,
        tuple(
            attempt.case
            for attempt in attempts
            if attempt.case.status != "pass" or any(use.error for use in attempt.uses)
        ),
        tuple(
            target.label
            for target in store.manifest.targets
            if not target.telemetry_complete and not any(use.case.target == target.label for use in uses)
        ),
        sum(_unmatched_errors(attempt) for attempt in attempts),
    )


def _unmatched_errors(attempt: _Attempt) -> int:
    calls: Final = tuple(use.call for use in attempt.uses)
    return sum(
        event.kind == "ToolResult" and event.is_error and _failed_call(event, calls) is None for event in attempt.events
    )


def _duration(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}s"


def _metrics_table(metrics: tuple[TargetMetrics, ...]) -> Table:
    table: Final = Table(box=None, padding=(0, 2), pad_edge=False, header_style=MUTED)
    for label in ("TARGET", "RATE", "MEAN", "P95", "$/ATTEMPT", "$/PASS"):
        table.add_column(label)
    for metric in metrics:
        table.add_row(
            Text(metric.target, style=ACCENT),
            f"{metric.pass_rate:.0%}" if metric.pass_rate is not None else "unknown",
            _duration(metric.mean_duration),
            _duration(metric.p95_duration),
            money(metric.mean_agent_cost),
            money(metric.cost_per_pass),
        )
    return table


def _plot_cell(x: int, y: int, positions: tuple[tuple[int, int, str], ...]) -> str:
    markers: Final = tuple(marker for px, py, marker in positions if px == x and py == y)
    return "*" if len(markers) > 1 else markers[0] if markers else " "


def _plot_row(y: int, positions: tuple[tuple[int, int, str], ...], ymax: float, percent: bool) -> Text:
    label: Final = f"{ymax:.0%}" if percent else f"{ymax:.2f}s"
    edge: Final = label if y == 0 else "0" if y == 8 else ""
    cells: Final = "".join(_plot_cell(x, y, positions) for x in range(40))
    return Text.assemble((f"{edge:>7} │", BORDER), (cells, ACCENT))


def _scatter(metrics: tuple[TargetMetrics, ...], *, latency: bool) -> Group:
    measured: Final = tuple(metric for metric in metrics if metric.mean_agent_cost is not None)
    xmax: Final = max((metric.mean_agent_cost or 0 for metric in measured), default=0)
    ymax: Final = max((metric.mean_duration or 0 for metric in measured), default=0) if latency else 1.0
    positions: Final = tuple(_point(metric, index, xmax, ymax, latency) for index, metric in enumerate(measured))
    return Group(
        Text("Cost vs latency" if latency else "Cost vs quality", style=f"bold {INK}"),
        Text(
            "Mean duration (s)" if latency else "Pass rate (errors included)",
            style=MUTED,
        ),
        *tuple(_plot_row(y, positions, ymax, not latency) for y in range(9)),
        Text("        └" + "─" * 40, style=BORDER),
        Text(f"         $0{' ' * 28}{money(xmax)}", style=MUTED),
        Text("         Mean agent cost / attempt (USD)", style=MUTED),
    )


def _point(metric: TargetMetrics, index: int, xmax: float, ymax: float, latency: bool) -> tuple[int, int, str]:
    yvalue: Final = (metric.mean_duration or 0) if latency else (metric.pass_rate or 0)
    x: Final = round((metric.mean_agent_cost or 0) / xmax * 39) if xmax else 0
    y: Final = 8 - (round(yvalue / ymax * 8) if ymax else 0)
    return x, y, str(index + 1) if index < 9 else "+"


def _charts(console: Console, metrics: tuple[TargetMetrics, ...]) -> None:
    known: Final = tuple(metric for metric in metrics if metric.mean_agent_cost is not None)
    missing: Final = tuple(metric for metric in metrics if metric.mean_agent_cost is None)
    if known:
        plots: Final = (
            _scatter(metrics, latency=False),
            _scatter(metrics, latency=True),
        )
        if console.width >= 110:
            layout: Final = Table.grid(padding=(0, 4))
            layout.add_row(*plots)
            console.print(layout)
        else:
            for plot in plots:
                console.print(plot)
                console.print()
        console.print(
            Text(
                "  ".join(f"{index + 1} {metric.target}" for index, metric in enumerate(known)),
                style=ACCENT,
            )
        )
        console.print(
            Text(
                "* overlapping points. + target 10 or later. Lower cost and latency are better.",
                style=MUTED,
            )
        )
    if missing:
        console.print(
            Text(
                "Chart omitted unknown costs: "
                + ", ".join(
                    f"{metric.target} ({metric.missing_cost_count}/{metric.attempts} missing)" for metric in missing
                ),
                style=MUTED,
            )
        )


def _tool_table(report: ToolReport, redundancy: bool) -> Table:
    table: Final = Table(box=None, padding=(0, 1), pad_edge=False, header_style=MUTED)
    labels: Final = ("TARGET", "TOOL", "CALLS", "AVG/CASE", "ERRORS")
    for label in labels + (("REPEATS", "EXTRA") if redundancy else ()):
        table.add_column(label)
    for metric in report.metrics:
        table.add_row(*_tool_row(metric, redundancy))
    return table


def _tool_row(metric: ToolMetrics, redundancy: bool) -> tuple[str, ...]:
    values: Final = (
        metric.target,
        metric.tool,
        str(metric.calls),
        f"{metric.calls_per_attempt:.2f}",
        str(metric.errors),
    )
    extra: Final = (
        str(metric.repeated_attempts),
        f"{metric.extra_calls} ({metric.extra_call_rate:.0%})",
    )
    return values + (extra if redundancy else ())


def _inspect_hint(store: RunStore, case: CaseRow) -> str:
    return (
        f"/inspect {shlex.quote(store.run_id)} --case {shlex.quote(case.task_id)} "
        f"--target {shlex.quote(case.target)} --trial {case.trial}"
    )


def _highlights(console: Console, store: RunStore, metrics: tuple[TargetMetrics, ...]) -> None:
    best, worst = quality_extremes(metrics)
    if best:
        console.print(
            Text(
                "Highest pass rate: " + ", ".join(metric.target for metric in best),
                style=GREEN,
            )
        )
        console.print(
            Text(
                "Lowest pass rate: " + ", ".join(metric.target for metric in worst),
                style=MUTED,
            )
        )
    cases: Final = ranked_cases(store)
    if cases:
        console.print(
            Text(
                "Case order uses pass, fail, error, then duration. Ties use case/target/trial IDs.",
                style=MUTED,
            )
        )
        console.print(
            Text(
                f"Best observed: {cases[0].status} · {_duration(cases[0].duration)} · {_inspect_hint(store, cases[0])}"
            )
        )
        console.print(
            Text(
                f"Worst observed: {cases[-1].status} · {_duration(cases[-1].duration)} · {_inspect_hint(store, cases[-1])}"
            )
        )


def print_analysis(
    console: Console,
    store: RunStore,
    *,
    tools: bool = True,
    redundancy: bool = True,
    charts: bool = True,
) -> None:
    metrics: Final = target_metrics(store)
    console.print()
    console.print(Text("Saved run analysis", style=f"bold {INK}"))
    console.print(Text("Completed attempts only. No model calls or reruns.", style=MUTED))
    if store.manifest.fixture:
        console.print(Text("Fixture metrics are simulated. No model bill.", style=MUTED))
    console.print(_metrics_table(metrics))
    console.print(
        Text(
            "Pass rate includes errors. P95 uses nearest rank. $/pass includes all agent attempt costs.",
            style=MUTED,
        )
    )
    console.print(
        Text(
            "Unknown costs stay unknown. Model-step counts are unavailable.",
            style=MUTED,
        )
    )
    if charts:
        console.print()
        _charts(console, metrics)
    console.print()
    _highlights(console, store, metrics)
    if not tools and not redundancy:
        return
    report: Final = tool_report(store)
    console.print()
    console.print(Text("Observed tool calls across completed attempts", style=f"bold {INK}"))
    console.print(_tool_table(report, redundancy))
    incomplete: Final = tuple(target.label for target in store.manifest.targets if not target.telemetry_complete)
    if incomplete:
        console.print(
            Text(
                "Partial telemetry; observed counts are lower bounds: " + ", ".join(incomplete),
                style=MUTED,
            )
        )
    if report.unobserved_targets:
        console.print(
            Text(
                "No tool events recorded; tool usage unknown: " + ", ".join(report.unobserved_targets),
                style=MUTED,
            )
        )
    if report.unmatched_errors:
        console.print(
            Text(
                f"{report.unmatched_errors} tool errors have no matching call ID; excluded from per-tool errors.",
                style=MUTED,
            )
        )
    if redundancy:
        console.print(
            Text(
                "Repeats mean the same tool and normalized arguments within one attempt. Extra rate = extra/calls.",
                style=MUTED,
            )
        )
        for example in report.repeats[:3]:
            console.print(
                Text(
                    f"Repeated {example.tool} ×{example.calls} · {_inspect_hint(store, example.case)}",
                    style="yellow",
                )
            )
            console.print(Text(f"  Arguments: {example.arguments[:160]}", style=MUTED))
    for case in report.failures[:3]:
        console.print(
            Text(
                f"{case.status.upper()} · {case.detail[:160]} · {_inspect_hint(store, case)}",
                style=MUTED,
            )
        )
