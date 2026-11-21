from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from litellm.harness.evaluation import EvalProgress
from litellm.lens.core import CaseRow, RunStore, load_config, repeated_calls, run_suite
from litellm.lens.demo import create_demo


def test_should_store_registered_agent_harness_and_model_names(tmp_path: Path) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=1, delay=0)
    suite: Final = load_config(tasks, config)
    store: Final = RunStore.create(tmp_path / "runs", suite)
    registered: Final = tuple((target.agent, target.harness, target.model) for target in store.manifest.targets)
    assert registered == (
        ("SupportAgent", "run_agent", "fixture/model-a"),
        ("SupportAgent", "run_agent", "fixture/model-b"),
    )
    run_id: Final = store.run_id
    store.close()
    reopened: Final = RunStore.open(tmp_path / "runs", run_id)
    assert reopened.manifest.targets == store.manifest.targets
    reopened.close()


async def test_should_checkpoint_then_resume_only_unfinished_attempts(
    tmp_path: Path,
) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=12, delay=0)
    suite: Final = load_config(tasks, config)
    store: Final = RunStore.create(tmp_path / "runs", suite, trials=2)

    def stop_after_three(row: CaseRow) -> None:
        if store.completed_count == 3:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_suite(store, suite, max_concurrency=1, on_result=stop_after_three)
    saved: Final = store.rows()
    run_id: Final = store.run_id
    store.close()
    resumed: Final = RunStore.open(tmp_path / "runs", run_id)
    starts: Final[asyncio.Queue[tuple[str, str, int]]] = asyncio.Queue()

    def observe(progress: EvalProgress) -> None:
        if progress.event is None:
            starts.put_nowait((progress.task_id, progress.target, progress.trial))

    await run_suite(resumed, suite, max_concurrency=4, on_progress=observe)
    rows: Final = resumed.rows()
    executed: Final = frozenset(starts.get_nowait() for _ in range(starts.qsize()))
    assert resumed.completed_count == 48
    assert all(row in rows for row in saved)
    assert not executed.intersection((row.task_id, row.target, row.trial) for row in saved)
    assert len(executed) == 45
    assert len({(row.task_id, row.target, row.trial) for row in rows}) == len(rows)
    assert {row.status for row in rows} == {"pass", "fail", "error"}
    resumed.close()


async def test_should_record_repeated_lookup_before_case_completion(
    tmp_path: Path,
) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=1, delay=0)
    suite: Final = load_config(tasks, config)
    store: Final = RunStore.create(tmp_path / "runs", suite)
    observations: Final[asyncio.Queue[tuple[int, int]]] = asyncio.Queue()

    def observe(progress: EvalProgress) -> None:
        duplicates: Final = repeated_calls(store.events(progress.run_id))
        if duplicates:
            observations.put_nowait((duplicates[0][2], store.completed_count))

    await run_suite(store, suite, max_concurrency=1, on_progress=observe)
    assert observations.get_nowait() == (2, 1)
    assert store.rows()[1].status == "fail"
    assert repeated_calls(store.events(store.rows()[1].run_id))[0][2] == 2
    store.close()


async def test_should_keep_custom_adapter_telemetry_unknown_unless_declared(
    tmp_path: Path,
) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=1, delay=0)
    suite: Final = replace(load_config(tasks, config), telemetry_complete=frozenset())
    store: Final = RunStore.create(tmp_path / "runs", suite)
    await run_suite(store, suite)
    assert all(row.agent_cost is None and row.model_calls is None for row in store.rows())
    assert all(row.agent_cost is None for row in store.summary())
    assert all(row.judge_cost == 0 for row in store.rows())
    assert all(row.tool_calls is None for row in store.rows())
    assert repeated_calls(store.events(store.rows()[1].run_id))[0][2] == 2
    store.close()


async def test_should_reject_changed_dataset_before_resuming(tmp_path: Path) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=1, delay=0)
    suite: Final = load_config(tasks, config)
    store: Final = RunStore.create(tmp_path / "runs", suite)
    tasks.write_text(tasks.read_text().replace("charged twice", "charged three times"))
    with pytest.raises(ValueError, match="changed"):
        await run_suite(store, load_config(tasks, config))
    assert store.completed_count == 0
    store.close()


async def test_should_record_zero_judge_cost_when_runtime_failure_skips_grading(
    tmp_path: Path,
) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=11, delay=0)
    suite: Final = load_config(tasks, config)
    store: Final = RunStore.create(tmp_path / "runs", suite)
    await run_suite(store, suite)
    failed: Final = tuple(row for row in store.rows() if row.status == "error")
    assert len(failed) == 1
    assert failed[0].judge_cost == 0
    assert all(summary.judge_cost == 0 for summary in store.summary())
    store.close()


async def test_should_process_3000_task_fixture_and_noop_a_completed_resume(
    tmp_path: Path,
) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=3000, delay=0)
    suite: Final = load_config(tasks, config)
    store: Final = RunStore.create(tmp_path / "runs", suite)
    await run_suite(store, suite, max_concurrency=20)
    rows: Final = store.rows()
    assert len(rows) == 6000
    assert len({(row.task_id, row.target, row.trial) for row in rows}) == 6000
    assert tuple(summary.completed for summary in store.summary()) == (3000, 3000)

    def no_rerun(progress: EvalProgress) -> None:
        pytest.fail("Completed resume must not execute another attempt")

    await run_suite(store, suite, on_progress=no_rerun)
    assert store.rows() == rows
    store.close()


def test_should_reject_duplicate_csv_ids_and_preserve_demo_files(
    tmp_path: Path,
) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=2, delay=0)
    original: Final = tasks.read_text()
    with pytest.raises(FileExistsError):
        create_demo(tmp_path / "demo")
    assert tasks.read_text() == original
    tasks.write_text(original.replace("ticket-0002", "ticket-0001"))
    with pytest.raises(ValueError, match="unique"):
        load_config(tasks, config)


async def test_should_import_a_custom_python_harness_beside_config(
    tmp_path: Path,
) -> None:
    tasks, config = create_demo(tmp_path / "demo", count=1, delay=0)
    harness: Final = config.parent / "my_agent.py"
    harness.write_text(
        "from litellm.lens.demo import fixture_runner\n"
        "async def run_custom(context):\n"
        "    return await fixture_runner(context, delay=0)\n"
    )
    config.write_text(
        "from litellm import EvalTarget, Harness\n"
        "from my_agent import run_custom\n"
        "targets = [EvalTarget(Harness.TOOL_LOOP, 'fixture/one', runner=run_custom)]\n"
    )
    previous_path: Final = tuple(sys.path)
    suite: Final = load_config(tasks, config)
    assert tuple(sys.path) == previous_path
    store: Final = RunStore.create(tmp_path / "runs", suite)
    await run_suite(store, suite)
    assert tuple(row.status for row in store.rows()) == ("pass",)
    assert store.rows()[0].agent_cost is None
    store.close()
