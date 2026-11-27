"""Evaluate independent task attempts through the public agent runtime."""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.harness.errors import OutputInvalid
from litellm.harness.evaluation.checks import run_check
from litellm.harness.evaluation.types import (
    AgentInput,
    AgentOutput,
    CheckResult,
    CommandCheck,
    EvalAgent,
    EvalRun,
    EvalScorecard,
    EvalStatus,
    EvalTarget,
    EvalTask,
    JudgeCheck,
)
from litellm.harness.runtime import aagent, aagent_session
from litellm.harness.sandbox import local
from litellm.harness.sandbox.base import Sandbox
from litellm.harness.sync import run_sync
from litellm.harness.types import Result

SandboxFactory: TypeAlias = Callable[[Path], Sandbox]


@dataclass(frozen=True, slots=True)
class _Settings:
    max_turns: int | None
    timeout: float
    setup_timeout: float
    sandbox: SandboxFactory
    metadata: Mapping[str, object]
    api_base: str | None
    api_key: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Outcome:
    status: EvalStatus
    detail: str
    result: Result | AgentOutput | None = None
    check: CheckResult | None = None


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_timeout(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _settings(
    max_turns: int | None,
    timeout: float,
    setup_timeout: float,
    sandbox: SandboxFactory,
    metadata: Mapping[str, object] | None,
    api_base: str | None,
    api_key: str | None,
) -> _Settings:
    if max_turns is not None:
        _positive_integer("max_turns", max_turns)
    _positive_timeout("timeout", timeout)
    _positive_timeout("setup_timeout", setup_timeout)
    copied_metadata: Final = MappingProxyType(dict(metadata or {}))
    json.dumps(dict(copied_metadata), allow_nan=False)
    return _Settings(max_turns, timeout, setup_timeout, sandbox, copied_metadata, api_base, api_key)


def _validate_task(task: EvalTask) -> None:
    if task.workspace is not None and not Path(task.workspace).is_dir():
        raise ValueError(f"task {task.id!r} workspace must be an existing directory")
    for command in task.setup:
        CommandCheck(command)
    _validate_files(task.files)


def _validate_files(files: Mapping[str, object]) -> None:
    for name, content in files.items():
        path = Path(name)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("task files must have workspace-relative paths")
        if not isinstance(content, str | bytes):
            raise TypeError("task file contents must be str or bytes")


async def _validate_targets(targets: Sequence[EvalTarget | EvalAgent], settings: _Settings) -> None:
    builtins: Final = tuple(target for target in targets if isinstance(target, EvalTarget))
    if not builtins:
        return
    with tempfile.TemporaryDirectory(prefix="litellm-eval-validate-") as directory:
        box: Final = local(directory)
        try:
            for target in builtins:
                session = aagent_session(
                    target.harness,
                    sandbox=box,
                    model=target.model,
                    api_key=settings.api_key,
                    api_base=settings.api_base,
                    tools=target.tools,
                    options=target.options,
                    max_turns=settings.max_turns,
                    timeout=settings.timeout,
                    metadata=settings.metadata,
                )
                await session.aclose()
        finally:
            await box.close()


def _prepare_workspace(task: EvalTask, directory: Path) -> None:
    if task.workspace is not None:
        source: Final = Path(task.workspace)
        if source.is_symlink():
            raise ValueError("task workspace root must not be a symbolic link")
        shutil.copytree(
            source,
            directory,
            symlinks=True,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        if any(path.is_symlink() for path in directory.rglob("*")):
            raise ValueError("task workspace must not contain symbolic links")
    for name, content in task.files.items():
        destination = directory / name
        if not destination.resolve().is_relative_to(directory.resolve()):
            raise ValueError("task file escapes the run workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content.encode() if isinstance(content, str) else content)


async def _prepare(task: EvalTask, directory: Path) -> None:
    preparation: Final = asyncio.create_task(asyncio.to_thread(_prepare_workspace, task, directory))
    try:
        await asyncio.shield(preparation)
    except asyncio.CancelledError:
        await preparation
        raise


async def _setup(task: EvalTask, box: Sandbox) -> str | None:
    for command in task.setup:
        completed = await box.run(command)
        if completed.exit_code != 0:
            return f"setup exited {completed.exit_code}: {completed.stderr or completed.stdout}"
    return None


def _custom_output(value: object) -> AgentOutput:
    if not isinstance(value, AgentOutput):
        raise TypeError("custom evaluation agents must return AgentOutput")
    return value


async def _agent_result(
    task: EvalTask,
    target: EvalTarget | EvalAgent,
    box: Sandbox,
    metadata: Mapping[str, object],
    settings: _Settings,
    trial: int,
) -> Result | AgentOutput:
    if isinstance(target, EvalAgent):
        output: Final = await asyncio.wait_for(
            target.run(AgentInput(task.id, task.prompt, box, trial, metadata)), timeout=settings.timeout
        )
        return _custom_output(output)
    pending: Final = aagent(
        target.harness,
        task.prompt,
        sandbox=box,
        model=target.model,
        options=target.options,
        tools=target.tools,
        max_turns=settings.max_turns,
        timeout=settings.timeout,
        metadata=metadata,
        api_key=settings.api_key,
        api_base=settings.api_base,
    )
    if not isinstance(pending, Coroutine):
        raise TypeError("agent evaluation requires a non-streaming agent result")
    return await pending


def _failure(stage: str, error: Exception, result: Result | AgentOutput | None = None) -> _Outcome:
    return _Outcome(EvalStatus.ERROR, f"{stage}: {type(error).__name__}: {error}", result)


async def _execute(
    task: EvalTask,
    target: EvalTarget | EvalAgent,
    box: Sandbox,
    metadata: Mapping[str, object],
    settings: _Settings,
    trial: int,
) -> _Outcome:
    try:
        setup_error: Final = await asyncio.wait_for(_setup(task, box), timeout=settings.setup_timeout)
    except Exception as error:  # noqa: BLE001  # User setup failures become error rows.
        return _failure("setup", error)
    if setup_error is not None:
        return _Outcome(EvalStatus.ERROR, setup_error)
    try:
        result: Final = await _agent_result(task, target, box, metadata, settings, trial)
    except OutputInvalid as error:
        return _failure("agent", error, error.result)
    except Exception as error:  # noqa: BLE001  # Harness failures must not abort sibling runs.
        return _failure("agent", error)
    if result.stop_reason != "done":
        status: Final = EvalStatus.FAIL if result.stop_reason == "max_turns" else EvalStatus.ERROR
        return _Outcome(status, f"agent stopped with {result.stop_reason}", result)
    try:
        checked: Final = await run_check(
            task.check,
            result,
            box,
            metadata,
            prompt=task.prompt,
            api_base=settings.api_base,
            api_key=settings.api_key,
        )
    except Exception as error:  # noqa: BLE001  # User-defined graders can raise arbitrary exceptions.
        failure: Final = _failure("check", error, result)
        if isinstance(task.check, JudgeCheck):
            return replace(
                failure,
                check=CheckResult(False, failure.detail, judge_usage=None, judge_cost=None, error=failure.detail),
            )
        return failure
    if checked.error is not None:
        return _Outcome(EvalStatus.ERROR, checked.error, result, checked)
    return _Outcome(EvalStatus.PASS if checked.passed else EvalStatus.FAIL, checked.detail, result, checked)


async def _close(box: Sandbox) -> str | None:
    try:
        await box.close()
    except Exception as error:  # noqa: BLE001  # Custom sandbox cleanup failures become error rows.
        return f"cleanup: {type(error).__name__}: {error}"
    return None


async def _attempt(
    task: EvalTask,
    target: EvalTarget | EvalAgent,
    metadata: Mapping[str, object],
    settings: _Settings,
    trial: int,
) -> _Outcome:
    with tempfile.TemporaryDirectory(prefix="litellm-eval-") as directory:
        try:
            await _prepare(task, Path(directory))
            box: Final = settings.sandbox(Path(directory))
        except Exception as error:  # noqa: BLE001  # Custom sandbox factories can raise arbitrary exceptions.
            return _failure("workspace", error)
        try:
            outcome: Final = await _execute(task, target, box, metadata, settings, trial)
        finally:
            close_error: Final = await _close(box)
        return replace(outcome, status=EvalStatus.ERROR, detail=close_error) if close_error else outcome


async def _run_attempt(
    task: EvalTask,
    target: EvalTarget | EvalAgent,
    trial: int,
    suite_id: str,
    settings: _Settings,
) -> EvalRun:
    started: Final = time.perf_counter()
    run_id: Final = uuid.uuid4().hex
    metadata: Final = MappingProxyType(
        {
            **settings.metadata,
            "agent_eval_suite": suite_id,
            "agent_eval_run": run_id,
            "agent_eval_task": task.id,
            "agent_eval_target": target.name,
            "agent_eval_trial": trial,
        }
    )
    try:
        outcome: Final = await _attempt(task, target, metadata, settings, trial)
    except Exception as error:  # noqa: BLE001  # Workspace cleanup failures become error rows.
        failed: Final = _failure("workspace cleanup", error)
        return EvalRun(
            run_id,
            task.id,
            target.name,
            trial,
            failed.status,
            None,
            None,
            time.perf_counter() - started,
            failed.detail,
        )
    return EvalRun(
        run_id,
        task.id,
        target.name,
        trial,
        outcome.status,
        outcome.result,
        outcome.check,
        time.perf_counter() - started,
        outcome.detail,
    )


async def _run(
    task: EvalTask,
    target: EvalTarget | EvalAgent,
    trial: int,
    suite_id: str,
    semaphore: asyncio.Semaphore,
    settings: _Settings,
) -> EvalRun:
    async with semaphore:
        return await _run_attempt(task, target, trial, suite_id, settings)


async def aagent_eval_attempt(
    task: EvalTask,
    target: EvalTarget | EvalAgent,
    *,
    trial: int = 1,
    suite_id: str | None = None,
    max_turns: int | None = None,
    timeout: float = 300,
    setup_timeout: float = 120,
    sandbox: SandboxFactory = local,
    metadata: Mapping[str, object] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> EvalRun:
    """Run and grade one independent attempt for a caller that owns scheduling and persistence."""
    _positive_integer("trial", trial)
    if suite_id is not None and not suite_id.strip():
        raise ValueError("suite_id must be a non-empty string")
    _validate_task(task)
    settings: Final = _settings(max_turns, timeout, setup_timeout, sandbox, metadata, api_base, api_key)
    await _validate_targets((target,), settings)
    return await _run_attempt(task, target, trial, suite_id or uuid.uuid4().hex, settings)


async def aagent_eval(
    tasks: Sequence[EvalTask],
    targets: Sequence[EvalTarget | EvalAgent],
    *,
    trials: int = 1,
    max_concurrency: int = 1,
    max_turns: int | None = None,
    timeout: float = 300,
    setup_timeout: float = 120,
    sandbox: SandboxFactory = local,
    metadata: Mapping[str, object] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> EvalScorecard:
    """Compare harness/model targets on independent, graded task attempts."""
    _positive_integer("trials", trials)
    _positive_integer("max_concurrency", max_concurrency)
    task_items: Final = tuple(tasks)
    target_items: Final = tuple(targets)
    if not task_items or not target_items:
        raise ValueError("agent_eval requires at least one task and target")
    if len(frozenset(task.id for task in task_items)) != len(task_items):
        raise ValueError("task ids must be unique")
    if len(frozenset(target.name for target in target_items)) != len(target_items):
        raise ValueError("target labels must be unique")
    for task in task_items:
        _validate_task(task)
    settings: Final = _settings(max_turns, timeout, setup_timeout, sandbox, metadata, api_base, api_key)
    await _validate_targets(target_items, settings)
    suite_id: Final = uuid.uuid4().hex
    semaphore: Final = asyncio.Semaphore(max_concurrency)
    pending: Final = tuple(
        asyncio.create_task(_run(task, target, trial, suite_id, semaphore, settings))
        for task, target, trial in itertools.product(task_items, target_items, range(1, trials + 1))
    )
    try:
        runs: Final = await asyncio.gather(*pending)
    except BaseException:
        for attempt in pending:
            attempt.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    return EvalScorecard(suite_id, tuple(runs))


def agent_eval(
    tasks: Sequence[EvalTask],
    targets: Sequence[EvalTarget | EvalAgent],
    *,
    trials: int = 1,
    max_concurrency: int = 1,
    max_turns: int | None = None,
    timeout: float = 300,
    setup_timeout: float = 120,
    sandbox: SandboxFactory = local,
    metadata: Mapping[str, object] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> EvalScorecard:
    """Synchronous counterpart of aagent_eval."""
    return run_sync(
        aagent_eval(
            tasks,
            targets,
            trials=trials,
            max_concurrency=max_concurrency,
            max_turns=max_turns,
            timeout=timeout,
            setup_timeout=setup_timeout,
            sandbox=sandbox,
            metadata=metadata,
            api_base=api_base,
            api_key=api_key,
        ),
        "agent_eval",
    )
