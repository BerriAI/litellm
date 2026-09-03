from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .models import (
    Coverage,
    HarnessRun,
    RunStatus,
    SDK_FUNCTIONS,
    Strategy,
    section_confidence,
)

STATUS_GLYPHS = {
    RunStatus.NOT_RUN: "·",
    RunStatus.QUEUED: "○",
    RunStatus.RUNNING: "◉",
    RunStatus.PASSED: "✓",
    RunStatus.FAILED: "✗",
    RunStatus.SKIPPED: "↷",
    RunStatus.ERROR: "!",
    RunStatus.MISSING: "?",
    RunStatus.PLANNED: "—",
    RunStatus.NOT_APPLICABLE: "n/a",
}

STATUS_STYLES = {
    RunStatus.QUEUED: "dim",
    RunStatus.RUNNING: "bold cyan",
    RunStatus.PASSED: "bold green",
    RunStatus.FAILED: "bold red",
    RunStatus.SKIPPED: "yellow",
    RunStatus.ERROR: "bold red",
    RunStatus.MISSING: "magenta",
    RunStatus.PLANNED: "dim",
    RunStatus.NOT_APPLICABLE: "dim",
}


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def _rerun_command(nodeid: str) -> str:
    return f"poetry run pytest {shlex.quote(nodeid)} -q"


def _summary(run: HarnessRun) -> tuple[int, int, int, int]:
    outcomes: dict[str, RunStatus] = {}
    for result in run.results.values():
        outcomes.update(result.outcomes)
    return (
        list(outcomes.values()).count(RunStatus.PASSED),
        list(outcomes.values()).count(RunStatus.FAILED),
        list(outcomes.values()).count(RunStatus.ERROR),
        list(outcomes.values()).count(RunStatus.SKIPPED),
    )


def _cell_text(run: HarnessRun, strategy_id: str, sdk_function: str) -> tuple[str, str]:
    result = run.results.get(f"{strategy_id}:{sdk_function}")
    if result is None:
        return "", ""
    counts = ""
    if result.total:
        counts = f" {len(result.completed)}/{result.total}"
    coverage = " ◐" if result.case.coverage is Coverage.PARTIAL else ""
    return f"{STATUS_GLYPHS[result.status]}{counts}{coverage}", STATUS_STYLES.get(
        result.status, ""
    )


class RichDashboard(AbstractContextManager["RichDashboard"]):
    def __init__(
        self,
        strategies: Sequence[Strategy],
        confidence_strategies: Sequence[Strategy],
    ) -> None:
        from rich.console import Console
        from rich.live import Live

        self.strategies = strategies
        self.confidence_strategies = confidence_strategies
        self.console = Console()
        self.live: Any = Live(
            console=self.console, refresh_per_second=12, transient=False
        )

    def _table(self, run: HarnessRun) -> Any:
        from rich import box
        from rich.table import Table
        from rich.text import Text

        narrow = self.console.width < 96
        if narrow:
            table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=False)
            table.add_column("Strategy", ratio=3)
            table.add_column("Results", ratio=5)
            for strategy in self.strategies:
                values = []
                for sdk_function in SDK_FUNCTIONS:
                    value, style = _cell_text(run, strategy.id, sdk_function)
                    if value:
                        values.append(
                            Text.assemble((f"{sdk_function} ", "dim"), (value, style))
                        )
                table.add_row(strategy.label, Text("  ").join(values))
            return table

        table = Table(box=box.ROUNDED, expand=True, title="Strategy × SDK function")
        table.add_column("Strategy", ratio=3)
        for label in SDK_FUNCTIONS:
            table.add_column(label, justify="center", ratio=1)
        for strategy in self.strategies:
            cells = []
            for sdk_function in SDK_FUNCTIONS:
                value, style = _cell_text(run, strategy.id, sdk_function)
                cells.append(Text(value, style=style))
            table.add_row(strategy.label, *cells)
        return table

    def __enter__(self) -> "RichDashboard":
        self.live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.live.__exit__(*args)

    def update(self, run: HarnessRun) -> None:
        from rich.markup import escape
        from rich.panel import Panel

        active = run.current_nodeid or "Waiting for test events…"
        if len(active) > max(40, self.console.width - 16):
            active = f"…{active[-(self.console.width - 17):]}"
        passed, failed, errors, skipped = _summary(run)
        progress = (
            f"[bold]{run.completed_tests}/{run.unique_tests}[/bold] tests  "
            f"[green]{passed} passed[/green]  [red]{failed + errors} failed[/red]  "
            f"[yellow]{skipped} skipped[/yellow]  [dim]{_format_duration(run.duration)}[/dim]"
        )
        legend = "✓ pass  ✗ fail  ! error  ↷ skip\n? configured test missing  — planned  ◐ partial coverage"
        self.live.update(
            Panel(
                self._table(run),
                title="⚡ Rust ↔ Python parity lab",
                subtitle=f"{progress}\n[dim]{escape(active)}[/dim]\n{legend}",
                border_style="cyan",
            )
        )

    def finish(self, run: HarnessRun, exit_code: int) -> None:
        self.update(run)
        if run.failures:
            from rich.markup import escape
            from rich.panel import Panel

            for nodeid, detail in run.failures[:5]:
                rerun = _rerun_command(nodeid)
                self.console.print(
                    Panel(
                        f"{escape(detail)}\n\n[bold]Rerun just this test[/bold]\n"
                        f"[cyan]{escape(rerun)}[/cyan]",
                        title=f"✗ {escape(nodeid)}",
                        border_style="red",
                    )
                )
        durations: dict[str, float] = {}
        for result in run.results.values():
            for nodeid, duration in result.durations.items():
                durations[nodeid] = max(duration, durations.get(nodeid, 0.0))
        if durations:
            slow = sorted(durations.items(), key=lambda item: item[1], reverse=True)[:3]
            self.console.print(
                "[bold]Slowest tests[/bold]  "
                + "  •  ".join(
                    f"{Path(nodeid).name} [dim]{_format_duration(duration)}[/dim]"
                    for nodeid, duration in slow
                )
            )
        from rich import box
        from rich.table import Table

        confidence_table = Table(
            title="Port confidence by SDK section", box=box.ROUNDED, expand=True
        )
        confidence_table.add_column("SDK section")
        confidence_table.add_column("Score", justify="right")
        confidence_table.add_column("Confidence")
        confidence_table.add_column("Strategy evidence", ratio=4)
        confidence_styles = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}
        for score in section_confidence(run, self.confidence_strategies):
            confidence_table.add_row(
                score.sdk_function,
                f"{score.verified_strategies}/{score.required_strategies}  {score.percentage}%",
                f"[{confidence_styles[score.level.value]}]{score.level.value}[/]",
                "  ".join(score.details),
            )
        self.console.print(confidence_table)
        self.console.print(
            "[dim]Score = required strategies with passing evidence. "
            "LOC coverage remains a separate report.[/dim]"
        )
        style = "green" if exit_code == 0 else "red"
        self.console.print(
            f"[{style}]Harness finished in {_format_duration(run.duration)} "
            f"(exit {exit_code})[/{style}]"
        )


class PlainDashboard(AbstractContextManager["PlainDashboard"]):
    def __init__(
        self,
        strategies: Sequence[Strategy],
        confidence_strategies: Sequence[Strategy],
    ) -> None:
        self.strategies = strategies
        self.confidence_strategies = confidence_strategies
        self._seen: dict[str, tuple[RunStatus, int]] = {}

    def __enter__(self) -> "PlainDashboard":
        print("Rust <-> Python SDK parity harness", flush=True)
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self, run: HarnessRun) -> None:
        for key, result in run.results.items():
            state = (result.status, len(result.completed))
            if self._seen.get(key) != state:
                self._seen[key] = state
                progress = (
                    f" {len(result.completed)}/{result.total}" if result.total else ""
                )
                print(
                    f"{STATUS_GLYPHS[result.status]} {key}: {result.status.value}{progress}",
                    flush=True,
                )

    def finish(self, run: HarnessRun, exit_code: int) -> None:
        self.update(run)
        passed, failed, errors, skipped = _summary(run)
        print(
            f"Summary: {passed} passed, {failed} failed, {errors} errors, "
            f"{skipped} skipped in {_format_duration(run.duration)}",
            flush=True,
        )
        for nodeid, _ in run.failures[:5]:
            print(f"Rerun: {_rerun_command(nodeid)}", flush=True)
        print("Port confidence by SDK section", flush=True)
        for score in section_confidence(run, self.confidence_strategies):
            print(
                f"  {score.sdk_function:12} "
                f"{score.verified_strategies}/{score.required_strategies} "
                f"{score.percentage:3}% {score.level.value:6}  "
                f"{' | '.join(score.details)}",
                flush=True,
            )
        print(
            "  Score = required strategies with passing evidence; LOC is reported separately.",
            flush=True,
        )
        print(f"Harness finished with exit code {exit_code}", flush=True)


def make_dashboard(
    strategies: Sequence[Strategy],
    plain: bool = False,
    confidence_strategies: Sequence[Strategy] | None = None,
) -> RichDashboard | PlainDashboard:
    confidence_strategies = confidence_strategies or strategies
    interactive_terminal = (
        sys.stdout.isatty()
        and not os.environ.get("CI")
        and os.environ.get("TERM") != "dumb"
    )
    if not plain and interactive_terminal:
        try:
            import rich  # noqa: F401

            return RichDashboard(strategies, confidence_strategies)
        except ImportError:
            pass
    return PlainDashboard(strategies, confidence_strategies)
