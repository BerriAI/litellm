from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import replace
from io import StringIO
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import pytest
from rich.console import Console
from rich.live import Live

from litellm.harness.evaluation import EvalProgress
from litellm.harness.types import ToolCall
from litellm.lens.core import CaseRow, RunStore, load_config, run_suite
from litellm.lens.demo import create_demo
from litellm.lens.terminal import Progress, registrations


@pytest.mark.parametrize("width", (80, 132))
def test_should_show_saved_agent_harness_model_and_target_registrations(tmp_path: Path, width: int) -> None:
    tasks, config = create_demo(tmp_path / "suite", count=3, delay=0)
    suite: Final = replace(load_config(tasks, config), agent_name="SupportAgent")
    output: Final = StringIO()
    console: Final = Console(file=output, width=width, color_system=None)
    with closing(RunStore.create(tmp_path / "runs", suite)) as store:
        registrations(console, store)
        rendered: Final = output.getvalue()
        assert "3 tasks" in rendered
        assert "6 case runs planned" in rendered
        for target in store.manifest.targets:
            assert target.agent in rendered
            assert target.harness in rendered
            assert target.model in rendered
            assert target.label in rendered


async def test_should_show_each_target_current_task_and_checkpointed_percentage(
    tmp_path: Path,
) -> None:
    tasks, config = create_demo(tmp_path / "suite", count=3, delay=0)
    suite: Final = replace(load_config(tasks, config), agent_name="SupportAgent")
    output: Final = StringIO()
    console: Final = Console(file=output, width=132, color_system=None)
    snapshots: Final[SimpleQueue[str]] = SimpleQueue()
    with closing(RunStore.create(tmp_path / "runs", suite)) as store:

        def checkpoint(row: CaseRow) -> None:
            if store.completed_count == 1:
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await run_suite(store, suite, max_concurrency=1, on_result=checkpoint)
        display: Final = Progress(console, store, Live(console=console, auto_refresh=False))

        def observe(progress: EvalProgress) -> None:
            display.progress(progress)
            if isinstance(progress.event, ToolCall):
                console.print(display.render())
                snapshots.put(output.getvalue())
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await run_suite(
                store,
                suite,
                max_concurrency=1,
                on_progress=observe,
                on_result=display.result,
            )
        rendered: Final = snapshots.get_nowait()
        assert "SupportAgent" in rendered
        assert "model-a" in rendered and "model-b" in rendered
        assert "ticket-0001" in rendered
        assert "calling lookup_ticket" in rendered
        assert "33%" in rendered and "1/3" in rendered
        assert "0%" in rendered and "0/3" in rendered
        assert "1/6 completed" in rendered
