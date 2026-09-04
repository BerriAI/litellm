from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import AbstractContextManager
from textwrap import indent
from types import TracebackType
from typing import TYPE_CHECKING, Final

from litellm._logging import handler as litellm_log_handler

from .models import HarnessRun, RunStatus, Strategy
from .rendering import ReportSection

if TYPE_CHECKING:
    from rich.live import Live


class HarnessOutputFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        noisy_prefixes: Final = (
            "OCR cost:",
            "LoggingWorker: event loop changed;",
        )
        return not (record.name == "LiteLLM" and record.getMessage().startswith(noisy_prefixes))


_HARNESS_OUTPUT_FILTER: Final = HarnessOutputFilter()


def _start_output_filtering() -> None:
    litellm_log_handler.addFilter(_HARNESS_OUTPUT_FILTER)


def _stop_output_filtering() -> None:
    litellm_log_handler.removeFilter(_HARNESS_OUTPUT_FILTER)


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def _summary(run: HarnessRun) -> tuple[int, int, int, int]:
    outcomes: dict[str, RunStatus] = {}
    for result in run.results.values():
        outcomes.update(result.outcomes)
    values: Final = tuple(outcomes.values())
    return (
        values.count(RunStatus.PASSED),
        values.count(RunStatus.FAILED),
        values.count(RunStatus.ERROR),
        values.count(RunStatus.SKIPPED),
    )


def _strategy_state(statuses: tuple[RunStatus, ...], outcomes: tuple[RunStatus, ...]) -> str:
    for status in (
        RunStatus.ERROR,
        RunStatus.FAILED,
        RunStatus.MISSING,
        RunStatus.RUNNING,
        RunStatus.QUEUED,
    ):
        if status in statuses:
            return status.value
    if RunStatus.NOT_IMPLEMENTED in statuses:
        return RunStatus.NOT_IMPLEMENTED.value
    if outcomes and all(outcome is RunStatus.SKIPPED for outcome in outcomes):
        return RunStatus.SKIPPED.value
    if statuses and all(status is RunStatus.SKIPPED for status in statuses):
        return RunStatus.SKIPPED.value
    for status in (RunStatus.PASSED, RunStatus.SKIPPED):
        if status in statuses:
            return status.value
    return RunStatus.NOT_RUN.value


def _strategy_line(strategy: Strategy, run: HarnessRun) -> str:
    results: Final = tuple(run.results[case.key] for case in strategy.cases if case.key in run.results)
    outcomes: dict[str, RunStatus] = {}
    collected: set[str] = set()
    for result in results:
        outcomes.update(result.outcomes)
        collected.update(result.collected)
    values: Final = tuple(outcomes.values())
    statuses: Final = tuple(result.status for result in results)
    state: Final = _strategy_state(statuses, values)
    completed: Final = len(outcomes)
    total: Final = len(collected)
    progress: Final = f", {completed}/{total} checks" if total else ""
    counts: Final = (
        f", {values.count(RunStatus.PASSED)} passed, "
        f"{values.count(RunStatus.FAILED) + values.count(RunStatus.ERROR)} failed, "
        f"{values.count(RunStatus.SKIPPED)} skipped"
        if total
        else ""
    )
    duration: Final = run.strategy_durations.get(strategy.id, 0.0)
    return f"- {strategy.label}: {state}{progress}{counts}, {_format_duration(duration)}"


def _rendered_sections(run: HarnessRun, strategies: Sequence[Strategy]) -> tuple[ReportSection, ...]:
    return tuple(
        section
        for strategy in strategies
        if any(case.key in run.results for case in strategy.cases)
        for section in strategy.definition.render(
            tuple(run.results[case.key] for case in strategy.cases if case.key in run.results)
        )
    )


def _format_section(section: ReportSection) -> str:
    return f"{section.title}\n" + ("\n\n".join(section.blocks) or "- No results")


def _run_result(run: HarnessRun, exit_code: int) -> str:
    statuses: Final = tuple(result.status for result in run.results.values())
    if exit_code:
        return "FAILED"
    if not statuses or all(status is RunStatus.NOT_IMPLEMENTED for status in statuses):
        return "NOT RUN"
    if all(status is RunStatus.SKIPPED for status in statuses):
        return "SKIPPED"
    return "PASSED"


def final_report(run: HarnessRun, exit_code: int, strategies: Sequence[Strategy]) -> str:
    passed, failed, errors, skipped = _summary(run)
    run_result: Final = _run_result(run, exit_code)
    statuses: Final = tuple(case_result.status for case_result in run.results.values())
    not_implemented: Final = statuses.count(RunStatus.NOT_IMPLEMENTED)
    implemented: Final = len(statuses) - not_implemented
    skipped_cells: Final = statuses.count(RunStatus.SKIPPED)
    failure_lines: Final = tuple(
        f"{index}. {nodeid}\n{indent(detail.strip(), '    ')}"
        for index, (nodeid, detail) in enumerate(run.failures[:5], start=1)
    )
    rendered: Final = tuple(_format_section(section) for section in _rendered_sections(run, strategies))
    summary: Final = (
        "Rust <-> Python parity report\n\n"
        f"Result: {run_result}\n"
        f"Harness support: {implemented}/{len(statuses)} cases implemented\n"
        f"Cases: {len(statuses)} selected, {not_implemented} not implemented, {skipped_cells} skipped\n"
        f"Checks: {run.completed_checks}/{run.unique_checks} completed, {passed} passed, "
        f"{failed} failed, {errors} errors, {skipped} skipped\n"
        f"Duration: {_format_duration(run.duration)}\n"
        f"Exit code: {exit_code}"
    )
    failures: Final = (
        (f"Failures (showing {len(failure_lines)} of {len(run.failures)})\n" + "\n\n".join(failure_lines))
        if failure_lines
        else ""
    )
    return "\n\n".join((summary, *rendered, *((failures,) if failures else ())))


class RichDashboard(AbstractContextManager["RichDashboard"]):
    def __init__(self, strategies: Sequence[Strategy]) -> None:
        from rich.console import Console
        from rich.live import Live

        self.strategies = strategies
        self.console = Console()
        self.live: Live = Live(console=self.console, refresh_per_second=12, transient=True)
        self._live_active = False

    def __enter__(self) -> RichDashboard:
        _start_output_filtering()
        self.live.__enter__()
        self._live_active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _stop_output_filtering()
        if self._live_active:
            self.live.__exit__(exc_type, exc_value, traceback)
            self._live_active = False

    def update(self, run: HarnessRun) -> None:
        from rich.markup import escape

        active: Final = run.current_nodeid or "Waiting for test events..."
        available_width: Final = max(40, self.console.width - 10)
        visible_active: Final = active if len(active) <= available_width else f"...{active[-(available_width - 3) :]}"
        passed, failed, errors, skipped = _summary(run)
        total: Final = run.unique_checks
        percentage: Final = round(100 * run.completed_checks / total) if total else 0
        strategy_lines: Final = "\n".join(escape(_strategy_line(strategy, run)) for strategy in self.strategies)
        self.live.update(
            "[bold]Running Rust <-> Python parity[/bold]\n"
            f"Progress: [bold]{run.completed_checks}/{total} ({percentage}%)[/bold] | "
            f"[green]{passed} passed[/green] | [red]{failed + errors} failed[/red] | "
            f"[yellow]{skipped} skipped[/yellow] | [dim]{_format_duration(run.duration)}[/dim]\n"
            f"Strategies:\n{strategy_lines}\n"
            f"Current: [dim]{escape(visible_active)}[/dim]"
        )

    def finish(self, run: HarnessRun, exit_code: int) -> None:
        if self._live_active:
            self.live.stop()
            self._live_active = False
        print(final_report(run, exit_code, self.strategies), flush=True)  # noqa: T201  # CLI output


class PlainDashboard(AbstractContextManager["PlainDashboard"]):
    def __init__(self, strategies: Sequence[Strategy]) -> None:
        self.strategies = strategies
        self._seen: dict[str, tuple[RunStatus, int]] = {}

    def __enter__(self) -> PlainDashboard:
        _start_output_filtering()
        print("Running Rust <-> Python parity", flush=True)  # noqa: T201  # CLI output
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        _stop_output_filtering()

    def update(self, run: HarnessRun) -> None:
        for key, result in run.results.items():
            self._update_result(key, result.case.display_name, result.status, len(result.completed), result.total)

    def _update_result(self, key: str, label: str, status: RunStatus, completed: int, total: int) -> None:
        state: Final = (status, completed)
        previous: Final = self._seen.get(key)
        self._seen[key] = state
        visible: Final = status not in {
            RunStatus.NOT_RUN,
            RunStatus.QUEUED,
            RunStatus.NOT_IMPLEMENTED,
            RunStatus.SKIPPED,
        }
        should_print: Final = visible and (
            previous is None or previous[0] is not status or (completed > 0 and completed % 25 == 0)
        )
        if should_print:
            progress: Final = f" {completed}/{total}" if total else ""
            print(  # noqa: T201  # CLI output
                f"{label}: {status.value}{progress}", flush=True
            )

    def finish(self, run: HarnessRun, exit_code: int) -> None:
        print(  # noqa: T201  # CLI output
            f"\n{final_report(run, exit_code, self.strategies)}", flush=True
        )


def make_dashboard(strategies: Sequence[Strategy]) -> PlainDashboard:
    return PlainDashboard(strategies)
