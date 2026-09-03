from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.sdk_function_trace.fixtures import ROUTE_SPECS
from tests.sdk_function_trace.profiler import FunctionTraceEvent
from tests.sdk_function_trace.runtime import (
    TraceDiff,
    TraceFailed,
    TraceOk,
    TraceRun,
    TraceSkipped,
    attempt_trace,
    trace_diff,
)
from tests.sdk_function_trace.steps import Engine, pipeline_issues, pipeline_steps
from tests.sdk_function_trace.table import format_trace_table

_PYTHON_ONLY_COLOR: Final = "\033[34m"
_RUST_ONLY_COLOR: Final = "\033[33m"
_RESET: Final = "\033[0m"

_ENGINE_COLOR: Final[dict[Engine, str]] = {"python": _PYTHON_ONLY_COLOR, "rust": _RUST_ONLY_COLOR}


@dataclass(frozen=True, slots=True)
class EngineReport:
    engine: Engine
    run: TraceRun
    events: tuple[FunctionTraceEvent, ...]
    steps: tuple[FunctionTraceEvent, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Comparison:
    route: str
    label: str
    asynchronous: bool
    engines: tuple[EngineReport, ...]
    diff: TraceDiff

    @property
    def comparable(self) -> bool:
        return self.route != "audio_transcription" and all(isinstance(report.run, TraceOk) for report in self.engines)

    @property
    def passed(self) -> bool:
        return (
            (not self.comparable or self.diff.matches)
            and not any(report.issues for report in self.engines)
            and all(not isinstance(report.run, TraceFailed) for report in self.engines)
        )


def _events(run: TraceRun) -> tuple[FunctionTraceEvent, ...]:
    match run:
        case TraceOk(events=events):
            return events
        case TraceSkipped() | TraceFailed():
            return ()


def _engine_report(route: str, engine: Engine, run: TraceRun) -> EngineReport:
    events: Final = _events(run)
    steps: Final = pipeline_steps(route, engine, events)
    issues: Final = pipeline_issues(route, engine, steps) if isinstance(run, TraceOk) else ()
    return EngineReport(engine=engine, run=run, events=events, steps=steps, issues=issues)


def compare(route: str, *, asynchronous: bool) -> Comparison:
    runs: Final = {
        engine: attempt_trace(route, engine=engine, asynchronous=asynchronous) for engine in ("python", "rust")
    }
    engines: Final = tuple(_engine_report(route, engine, run) for engine, run in runs.items())
    return Comparison(
        route=route,
        label=ROUTE_SPECS[route].label,
        asynchronous=asynchronous,
        engines=engines,
        diff=trace_diff(engines[0].steps, engines[1].steps),
    )


def _tree_line(event: FunctionTraceEvent, only: frozenset[str], marker: str, color: str, *, colorize: bool) -> str:
    line: Final = f"{'  ' * event.depth}{event.function}" + (f"  {marker}" if event.function in only else "")
    return f"{color}{line}{_RESET}\n" if colorize and event.function in only else f"{line}\n"


def _tree_lines(
    events: tuple[FunctionTraceEvent, ...],
    only: frozenset[str],
    marker: str,
    color: str,
    *,
    colorize: bool,
) -> tuple[str, ...]:
    return tuple(_tree_line(event, only, marker, color, colorize=colorize) for event in events)


def _engine_lines(
    report: EngineReport, diff: TraceDiff, *, comparable: bool, full: bool, colorize: bool
) -> tuple[str, ...]:
    match report.run:
        case TraceSkipped(reason=reason):
            return (f"{report.engine}: SKIP ({reason})\n\n",)
        case TraceFailed(reason=reason):
            return (f"{report.engine}: FAIL ({reason})\n\n",)
        case TraceOk():
            shown: Final = report.events if full else report.steps
            only: Final = (
                () if full or not comparable else (diff.python_only if report.engine == "python" else diff.rust_only)
            )
            return (
                f"{report.engine} ({len(shown)} steps)\n\n",
                *_tree_lines(
                    shown,
                    frozenset(only),
                    f"<- {report.engine} only",
                    _ENGINE_COLOR[report.engine],
                    colorize=colorize,
                ),
                "\n",
            )


def _parity_lines(comparison: Comparison) -> tuple[str, ...]:
    if not comparison.comparable:
        if comparison.route == "audio_transcription":
            return ("step parity: UNAVAILABLE (Bedrock transcription has no independent Python implementation)\n",)
        return ("step parity: UNAVAILABLE (both engines must complete)\n",)
    diff: Final = comparison.diff
    order: Final = "the same" if diff.shared_order_matches else "a different"
    return (
        "diff\n\n",
        f"shared steps appear in {order} order\n",
        f"python-only: {', '.join(diff.python_only) or 'none'}\n",
        f"rust-only: {', '.join(diff.rust_only) or 'none'}\n\n",
        f"step parity: {'PASS' if diff.matches else 'FAIL'}\n",
    )


def _stage_lines(comparison: Comparison) -> tuple[str, ...]:
    return tuple(
        f"{report.engine} "
        f"{'SDK dispatch only' if comparison.route == 'audio_transcription' and report.engine == 'python' else 'pipeline'}: "
        f"{'FAIL: ' + '; '.join(report.issues) if report.issues else 'PASS'}\n"
        for report in comparison.engines
        if isinstance(report.run, TraceOk)
    )


def render(comparison: Comparison, *, full: bool, colorize: bool) -> str:
    mode: Final = "async" if comparison.asynchronous else "sync"
    traces: Final = (
        (format_trace_table(comparison.engines[0].steps, comparison.engines[1].steps, colorize=colorize) + "\n\n",)
        if not full and all(isinstance(report.run, TraceOk) for report in comparison.engines)
        else tuple(
            line
            for report in comparison.engines
            for line in _engine_lines(
                report, comparison.diff, comparable=comparison.comparable, full=full, colorize=colorize
            )
        )
    )
    return "".join(
        (
            f"route: {comparison.route}    provider: {comparison.label}    mode: {mode}\n\n",
            *traces,
            *_parity_lines(comparison),
            *_stage_lines(comparison),
            "Each successful invocation issued exactly one local provider request\n\n",
        )
    )
