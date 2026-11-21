from __future__ import annotations

import time
from pathlib import Path
from typing import Final

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from litellm.harness.evaluation import EvalProgress
from litellm.harness.types import Done, ToolCall, ToolResult
from litellm.harness.types import Text as AgentText
from litellm.lens.core import CaseRow, RunStore, TargetInfo, repeated_calls

ACCENT: Final = "#7eabff"
INK: Final = "#eef3f7"
MUTED: Final = "#a9b9c1"
BORDER: Final = "#526873"
GREEN: Final = "#91ddaf"
BACKGROUND: Final = "#17272e"
_GLYPHS: Final = (
    ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
)


def money(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def divider(console: Console, *, accent: bool = False) -> Text:
    line: Final = Text("─" * max(1, console.width - 2), style=BORDER)
    if accent:
        line.stylize("#347cff", 0, min(16, console.width))
    return line


def _hero_row(index: int) -> tuple[Text, Text]:
    pixels: Final = "  ".join("".join("█" if bit == "1" else " " for bit in letter[index]) for letter in _GLYPHS)
    copy: Final = {
        0: Text("Lens CLI", style=f"bold {INK}"),
        2: Text("Agent evals from your terminal.", style=MUTED),
        4: Text("Local by default. Lens is optional.", style=MUTED),
    }.get(index, Text(""))
    return Text(pixels, style="#347cff" if index < 6 else "#2462c9"), copy


def welcome(console: Console) -> None:
    header: Final = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(
        Text(f"{Path.cwd().name} % lens", style=MUTED),
        Text("LENS / LOCAL", style=f"bold {ACCENT}"),
    )
    console.print()
    console.print(header)
    console.print(divider(console, accent=True))
    console.print()
    hero: Final = Table.grid(padding=(0, 8))
    hero.add_column(no_wrap=True)
    hero.add_column()
    for logo, copy in tuple(_hero_row(index) for index in range(7)):
        hero.add_row(logo, copy)
    console.print(hero)
    console.print()
    console.print(divider(console))
    console.print()
    console.print(Text.assemble(("❯ ", ACCENT), ("/run tasks.csv --config evals.py", f"bold {INK}")))
    console.print()
    console.print(
        Text.assemble(
            ("✦ ", ACCENT),
            ("Your agents, harnesses, and models. One comparison.", f"bold {INK}"),
        )
    )
    console.print()
    console.print(Text("  Define tasks and checks in Python. Bring your own harness.", style=MUTED))
    console.print(Text("  Compare targets, inspect tool calls, interrupt and resume.", style=MUTED))
    console.print()
    console.print(Text("  /demo    Try a local comparison with fixture data", style=GREEN))
    console.print(
        Text(
            "  /init    Create tasks.csv and evals.py     /help    All commands",
            style=MUTED,
        )
    )
    console.print()


def registrations(console: Console, store: RunStore) -> None:
    agents: Final = frozenset(target.agent for target in store.manifest.targets)
    console.print()
    console.print(
        Text.assemble(
            ("✦ ", ACCENT),
            (
                f"Found {len(store.manifest.tasks):,} tasks and {len(agents)} registered agent{'s' if len(agents) != 1 else ''}",
                f"bold {INK}",
            ),
        )
    )
    console.print(
        Text(
            f"  {store.run_id}  ·  {'Fixture data · simulated metrics · $0 billed' if store.manifest.fixture else 'Live evaluation'}",
            style=MUTED,
        )
    )
    console.print()
    table: Final = Table(box=None, padding=(0, 1), pad_edge=False, header_style=MUTED)
    for label in ("  AGENT", "HARNESS", "MODEL", "TARGET"):
        table.add_column(label, no_wrap=True)
    for target in store.manifest.targets:
        table.add_row(
            f"  {target.agent}",
            target.harness,
            target.model,
            Text(target.label, style=ACCENT),
        )
    console.print(table)
    console.print()
    console.print(
        Text(
            f"  {store.manifest.total:,} case runs planned. Results stay local.",
            style=GREEN,
        )
    )
    if store.completed_count:
        console.print(
            Text(
                f"  {store.completed_count:,} completed attempts restored. Resuming remaining cases.",
                style=MUTED,
            )
        )
    console.print()
    console.print(divider(console))
    console.print()


def comparison(store: RunStore) -> Table:
    table: Final = Table(box=None, padding=(0, 1), pad_edge=False, header_style=MUTED)
    for label in (
        "  AGENT / TARGET",
        "PASS",
        "FAIL",
        "ERROR",
        "RATE",
        "AGENT $",
        "JUDGE $",
    ):
        table.add_column(label)
    for target, row in zip(store.manifest.targets, store.summary()):
        table.add_row(
            Text.assemble(
                (f"  {target.agent} / {row.target}\n", INK),
                (f"  {target.harness}\n  {target.model}", MUTED),
            ),
            Text(f"{row.passed}/{row.completed}", style=GREEN),
            Text(str(row.failed), style="yellow"),
            Text(str(row.errors), style="red"),
            f"{row.passed / row.completed:.0%}" if row.completed else "—",
            money(row.agent_cost),
            money(row.judge_cost),
        )
    return table


def _step(progress: EvalProgress) -> str:
    if isinstance(progress.event, ToolCall):
        return f"calling {progress.event.name}"
    if isinstance(progress.event, ToolResult):
        return "tool error" if progress.event.is_error else "tool returned"
    if isinstance(progress.event, AgentText):
        return "writing answer"
    if isinstance(progress.event, Done):
        return "checking answer"
    return "starting harness"


def _target_progress(
    target: TargetInfo,
    rows: tuple[CaseRow, ...],
    events: tuple[EvalProgress, ...],
    total: int,
    width: int,
) -> tuple[Text, Text, Text, Text]:
    completed: Final = sum(row.target == target.label for row in rows)
    active: Final = tuple(item for item in events if item.target == target.label)
    current: Final = active[-1] if active else None
    percent: Final = completed / total
    size: Final = 12 if width >= 105 else 0
    filled: Final = int(percent * size)
    bar: Final = Text.assemble(
        ("━" * filled, ACCENT),
        ("─" * (size - filled), BORDER),
        (f" {percent:4.0%}  {completed}/{total}", INK),
    )
    return (
        Text.assemble(
            (f"  {target.agent} / {target.label}\n", INK),
            (f"  {target.harness}\n  {target.model}", MUTED),
        ),
        Text(
            (current.task_id if current else "complete" if completed == total else "queued"),
            style=ACCENT,
        ),
        Text(
            _step(current) if current else "done" if completed == total else "waiting",
            style=MUTED,
        ),
        bar,
    )


class Progress:
    def __init__(self, console: Console, store: RunStore, live: Live) -> None:
        self.console = console
        self.store = store
        self.live = live
        self.started = time.monotonic()
        self.initial = store.completed_count
        self.done = self.initial
        self.active: tuple[EvalProgress, ...] = ()
        self.rows = store.rows()
        self.notice = ""
        self.last_refresh = 0.0

    def render(self) -> Group:
        elapsed: Final = time.monotonic() - self.started
        fresh: Final = self.done - self.initial
        eta: Final = f"{elapsed / fresh * (self.store.manifest.total - self.done):.0f}s" if fresh else "estimating"
        total: Final = len(self.store.manifest.tasks) * self.store.manifest.trials
        table: Final = Table(box=None, padding=(0, 1), pad_edge=False, header_style=MUTED)
        for label in ("  AGENT / TARGET", "TASK", "STEP", "PROGRESS"):
            table.add_column(label)
        for target in self.store.manifest.targets:
            table.add_row(*_target_progress(target, self.rows, self.active, total, self.console.width))
        costs: Final = tuple(row.agent_cost for row in self.rows)
        spend: Final = (
            money(sum(value for value in costs if value is not None))
            if costs and all(value is not None for value in costs)
            else "unknown"
        )
        return Group(
            Text.assemble(("✦ ", ACCENT), ("Evaluating now", f"bold {INK}")),
            Text(""),
            table,
            Text(""),
            Text(
                f"  {self.done:,}/{self.store.manifest.total:,} completed  ·  ETA {eta}  ·  "
                f"{'Simulated cost' if self.store.manifest.fixture else 'Agent spend'} {spend}",
                style=MUTED,
            ),
            Text(f"  {self.notice}" if self.notice else "", style="yellow"),
            Text("  ctrl+c to pause · completed cases are saved", style=MUTED),
            Text(""),
        )

    def progress(self, event: EvalProgress) -> None:
        self.active = tuple(item for item in self.active if item.run_id != event.run_id) + (event,)
        if isinstance(event.event, ToolCall):
            repeats: Final = repeated_calls(self.store.events(event.run_id))
            if repeats:
                name, _, count = repeats[-1]
                self.notice = f"Repeated lookup · {event.target} / {event.task_id} · {name} ×{count}, same arguments"
        if isinstance(event.event, ToolResult) and event.event.is_error:
            self.notice = f"Tool error · {event.target} / {event.task_id} · inspect this case after the run"
        self.refresh()

    def result(self, row: CaseRow) -> None:
        self.done += 1
        self.rows = (*self.rows, row)
        self.active = tuple(item for item in self.active if item.run_id != row.run_id)
        self.refresh()

    def refresh(self) -> None:
        if time.monotonic() - self.last_refresh < 0.1:
            return
        self.last_refresh = time.monotonic()
        self.live.update(self.render())
