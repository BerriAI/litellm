from __future__ import annotations

import argparse
import asyncio
import shlex
import time
import webbrowser
from collections.abc import Sequence
from contextlib import closing
from enum import Enum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.styles import Style
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
except ModuleNotFoundError as error:
    if error.name is not None and error.name.startswith(("rich", "prompt_toolkit")):
        raise SystemExit("Lens needs terminal dependencies. Install with pip install 'litellm[cli]'.") from None
    raise

from litellm.lens.analysis import print_analysis
from litellm.lens.core import (
    RunStore,
    SuiteConfig,
    load_config,
    repeated_calls,
    run_suite,
)
from litellm.lens.demo import create_demo
from litellm.lens.handoff import export_handoff
from litellm.lens.terminal import (
    ACCENT,
    BACKGROUND,
    BORDER,
    INK,
    MUTED,
    Progress,
    comparison,
    divider,
    money,
    registrations,
    welcome,
)

_ROOT: Final = Path(".lens/runs")
_PURPLE: Final = ACCENT
_HELP: Final = """/demo                      Run 12 fixture tasks against two targets. No model calls.
/demo --count 3000 --fast    Try progress and resume at scale.
/init my-eval               Create tasks.csv and an editable Python harness example.
/run tasks.csv --config evals.py
/resume <run_id>            Skip completed attempts, retry interrupted ones.
/compare [run_id]           Compare pass, fail, errors and known spend.
/report [run_id]            Tradeoff graphs, tool counts and repeated calls.
/inspect [run_id] --case ticket-0001 --target model-b
/runs                      List local runs.
/publish [run_id]           Save a local metadata-only handoff preview.
/open [run_id] --case ticket-0001 --target model-b
/exit                      Leave the shell.

Live Lens upload is not connected in this prototype. Saved results stay local.
evals.py runs trusted Python with your permissions. Custom runner metrics default to unknown."""


class _Command(Enum):
    DEMO = "demo"
    INIT = "init"
    RUN = "run"
    RESUME = "resume"
    COMPARE = "compare"
    REPORT = "report"
    INSPECT = "inspect"
    PUBLISH = "publish"
    OPEN = "open"
    RUNS = "runs"
    HELP = "help"


class _Args(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    command: _Command | None = None
    directory: Path | None = None
    tasks: Path = Path("tasks.csv")
    config: Path = Path("evals.py")
    run_id: str | None = None
    count: int = Field(default=12, ge=1)
    fast: bool = False
    delay: float = Field(default=0.35, ge=0)
    concurrency: int = Field(default=2, ge=1)
    quiet: bool = False
    trials: int = Field(default=1, ge=1)
    case: str | None = None
    target: str | None = None
    trial: int = Field(default=1, ge=1)
    include_content: bool = False
    tools: bool = False
    redundancy: bool = False
    charts: bool = False


def _parse(parser: argparse.ArgumentParser, arguments: Sequence[str] | None = None) -> _Args:
    return _Args.model_validate(vars(parser.parse_args(arguments)))


def _run(console: Console, store: RunStore, args: _Args, config: SuiteConfig) -> None:
    registrations(console, store)
    try:
        if args.quiet:
            asyncio.run(run_suite(store, config, max_concurrency=args.concurrency))
        else:
            with Live(console=console, refresh_per_second=8, transient=True) as live:
                display: Final = Progress(console, store, live)
                live.update(display.render())
                asyncio.run(
                    run_suite(
                        store,
                        config,
                        max_concurrency=args.concurrency,
                        on_progress=display.progress,
                        on_result=display.result,
                    )
                )
    except KeyboardInterrupt:
        console.print(
            Text(
                f"\nPaused. {store.completed_count:,} completed attempts saved.",
                style="yellow",
            )
        )
        console.print(Text(f"Resume with /resume {store.run_id}"))
        raise SystemExit(130) from None
    console.print(Text("  Comparison complete", style=f"bold {INK}"))
    console.print()
    console.print(comparison(store))
    if store.manifest.fixture:
        console.print(
            Text(
                "  Fixture metrics are simulated. No model calls. $0 billed.",
                style=MUTED,
            )
        )
    console.print()
    for row in tuple(item for item in store.rows() if item.status != "pass")[:3]:
        console.print(
            Text(
                f"  {row.status.upper()}  /inspect {store.run_id} --case {shlex.quote(row.task_id)} "
                f"--target {shlex.quote(row.target)} --trial {row.trial}",
                style=ACCENT,
            )
        )
    print_analysis(
        console,
        store,
        tools=args.tools or args.command is _Command.DEMO,
        redundancy=args.redundancy or args.command is _Command.DEMO,
        charts=args.charts or args.command is _Command.DEMO,
    )
    console.print(
        Text(
            f"\n  /inspect {store.run_id}    /report {store.run_id}    /publish {store.run_id}",
            style=MUTED,
        )
    )
    if any(row.status != "pass" for row in store.rows()):
        raise SystemExit(1)


def _inspect(console: Console, store: RunStore, case: str | None, target: str | None, trial: int) -> None:
    rows: Final = tuple(
        row
        for row in store.rows()
        if (case is None or row.task_id == case) and (target is None or row.target == target) and row.trial == trial
    )
    selected: Final = next((row for row in rows if row.status != "pass"), rows[0] if rows else None)
    if selected is None:
        raise ValueError("No saved attempt matches. Use /compare or /runs to find a run.")
    console.rule(
        f"{selected.task_id} / {selected.target} / trial {selected.trial}",
        style=_PURPLE,
    )
    console.print(
        Text(
            f"{selected.status.upper()}  {selected.detail}",
            style="green" if selected.status == "pass" else "yellow",
        )
    )
    task: Final = next(item for item in store.manifest.tasks if item.id == selected.task_id)
    console.print(
        Text(f"Prompt    {task.prompt}\nExpected  {task.expected}\nActual    {selected.text or '(no answer)'}")
    )
    console.print(
        Text(
            f"Agent {money(selected.agent_cost)}  Judge {money(selected.judge_cost)}  "
            f"Model calls {selected.model_calls if selected.model_calls is not None else 'unknown'}"
        )
    )
    for event in store.events(selected.run_id):
        console.print(
            Text(
                f"  {event.sequence + 1:02d}  {event.kind}  {event.name or ''}  "
                f"{event.arguments or event.output or ''}",
                style="red" if event.is_error else "dim",
            )
        )
    for name, arguments, count in repeated_calls(store.events(selected.run_id)):
        console.print(Text(f"Repeated {name} ×{count} with {arguments}", style="yellow"))


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(
        prog="lens",
        description="Local agent evaluations. Low-fidelity Lens CLI prototype.",
    )
    commands: Final = parser.add_subparsers(dest="command")
    for name in ("demo", "init"):
        command = commands.add_parser(name)
        command.add_argument("directory", nargs="?", type=Path)
        command.add_argument("--count", type=int, default=12)
        command.add_argument("--fast", action="store_true")
        command.add_argument("--delay", type=float, default=0.35)
        command.add_argument("--concurrency", type=int, default=2)
        command.add_argument("--quiet", action="store_true")
        command.add_argument("--tools", action="store_true")
        command.add_argument("--redundancy", action="store_true")
        command.add_argument("--charts", action="store_true")
    run: Final = commands.add_parser("run")
    run.add_argument("tasks", type=Path)
    run.add_argument("--config", type=Path, default=Path("evals.py"))
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--tools", action="store_true")
    run.add_argument("--redundancy", action="store_true")
    run.add_argument("--charts", action="store_true")
    resume: Final = commands.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--concurrency", type=int, default=2)
    resume.add_argument("--quiet", action="store_true")
    resume.add_argument("--tools", action="store_true")
    resume.add_argument("--redundancy", action="store_true")
    resume.add_argument("--charts", action="store_true")
    for name in ("compare", "report", "inspect", "publish", "open"):
        command = commands.add_parser(name)
        command.add_argument("run_id", nargs="?")
        command.add_argument("--case")
        command.add_argument("--target")
        command.add_argument("--trial", type=int, default=1)
        command.add_argument("--include-content", action="store_true")
        command.add_argument("--tools", action="store_true")
        command.add_argument("--redundancy", action="store_true")
        command.add_argument("--charts", action="store_true")
    commands.add_parser("runs")
    commands.add_parser("help")
    return parser


def _dispatch(console: Console, args: _Args) -> None:
    if args.command is _Command.HELP:
        console.print(Text(_HELP))
        return
    if args.command in (_Command.DEMO, _Command.INIT):
        directory: Final = args.directory or Path("lens-demo") / f"demo-{time.time_ns()}"
        tasks, config = create_demo(directory, count=args.count, delay=0 if args.fast else args.delay)
        console.print(Text(f"Created {tasks} and {config}. Edit evals.py to define your harness and targets."))
        if args.command is _Command.INIT:
            console.print(Text(f"/run {tasks} --config {config}", style=_PURPLE))
            return
        plan: Final = load_config(tasks, config)
        with closing(RunStore.create(_ROOT, plan)) as store:
            _run(console, store, args, plan)
        return
    if args.command is _Command.RUN:
        live_plan: Final = load_config(args.tasks, args.config)
        with closing(RunStore.create(_ROOT, live_plan, args.trials)) as store:
            _run(console, store, args, live_plan)
        return
    if args.command is _Command.RUNS:
        for manifest in RunStore.list_runs(_ROOT):
            console.print(
                Text(
                    f"{manifest.run_id}  {manifest.total:,} planned  "
                    f"{'fixture' if manifest.fixture else 'live'}  {manifest.tasks_path}"
                )
            )
        return
    manifests: Final = RunStore.list_runs(_ROOT)
    run_id: Final = args.run_id or (manifests[0].run_id if manifests else None)
    if run_id is None:
        raise ValueError("No local runs yet. Start with /demo or /run.")
    with closing(RunStore.open(_ROOT, run_id)) as store:
        if args.command is _Command.RESUME:
            _run(
                console,
                store,
                args,
                load_config(Path(store.manifest.tasks_path), Path(store.manifest.config_path)),
            )
        elif args.command is _Command.COMPARE:
            console.print(Text(run_id, style=_PURPLE))
            console.print(comparison(store))
        elif args.command is _Command.REPORT:
            console.print(comparison(store))
            requested: Final = args.tools or args.redundancy or args.charts
            print_analysis(
                console,
                store,
                tools=args.tools or not requested,
                redundancy=args.redundancy or not requested,
                charts=args.charts or not requested,
            )
        elif args.command is _Command.INSPECT:
            _inspect(console, store, args.case, args.target, args.trial)
        else:
            preview: Final = export_handoff(store, include_content=args.include_content)
            console.print(
                Text(
                    "Local handoff preview saved. No upload or model calls.",
                    style=_PURPLE,
                )
            )
            console.print(Text(str(preview.json_path)))
            url: Final = preview.case_url(args.case, args.target, args.trial) if args.case else preview.report_url
            console.print(Text(url))
            if args.command is _Command.OPEN:
                webbrowser.open(url)


def _prompt(console: Console) -> FormattedText:
    return FormattedText(
        [
            ("class:rule", "─" * max(console.width - 2, 1)),
            ("", "\n"),
            ("class:prompt", "❯ "),
        ]
    )


def _read_command(
    console: Console,
    parser: argparse.ArgumentParser,
    session: PromptSession[str] | None = None,
) -> bool:
    if session is None:
        console.print(divider(console))
    line: Final = (session.prompt(lambda: _prompt(console)) if session is not None else console.input("❯ ")).strip()
    if line in ("/exit", "exit", "/quit", "quit"):
        return False
    if line:
        _dispatch(console, _parse(parser, shlex.split(line.removeprefix("/")) or ["help"]))
    return True


def main() -> None:
    console: Final = Console(highlight=False, style=INK)
    parser: Final = _parser()
    try:
        args: Final = _parse(parser)
    except (ValueError, TypeError) as error:
        console.print(Text(str(error), style="red"))
        raise SystemExit(2) from error
    if args.command is not None:
        try:
            _dispatch(console, args)
        except (OSError, ValueError, TypeError) as error:
            console.print(Text(str(error), style="red"))
            raise SystemExit(1) from error
        return
    welcome(console)
    session: Final = (
        PromptSession[str](
            completer=WordCompleter(tuple(f"/{command.value}" for command in _Command) + ("/exit",)),
            complete_while_typing=False,
            bottom_toolbar=lambda: FormattedText([("", f"{Path.cwd().name}  ·  Lens off     Local prototype")]),
            style=Style.from_dict(
                {
                    "": f"{INK} bg:{BACKGROUND}",
                    "prompt": ACCENT,
                    "rule": BORDER,
                    "bottom-toolbar": f"{MUTED} bg:{BACKGROUND}",
                }
            ),
        )
        if console.is_terminal
        else None
    )
    while True:
        try:
            if not _read_command(console, parser, session):
                return
        except EOFError:
            return
        except KeyboardInterrupt:
            console.print(Text("\nUse /exit to leave."))
        except SystemExit:
            continue
        except (OSError, ValueError, TypeError) as error:
            console.print(Text(str(error), style="red"))


if __name__ == "__main__":
    main()
