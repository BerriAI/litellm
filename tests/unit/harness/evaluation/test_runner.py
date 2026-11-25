from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest

from litellm import (
    AgentInput,
    AgentOutput,
    CommandCheck,
    EvalAgent,
    EvalOutput,
    EvalStatus,
    EvalTarget,
    EvalTask,
    PredicateCheck,
    aagent_eval,
    aagent_eval_attempt,
    agent_eval,
)
from litellm.harness.errors import SandboxError
from litellm.harness.options import ToolLoopOptions
from litellm.harness.sandbox.base import CompletedRun, Process
from litellm.harness.sandbox.local import LocalSandbox
from litellm.harness.types import FileChange, Harness, ToolCall, ToolResult, Usage
from litellm.types.utils import ModelResponse

CommandHandler = Callable[["RecordingSandbox", Sequence[str]], Awaitable[CompletedRun]]


class RecordingSandbox:
    def __init__(self, workspace: Path, command_handler: CommandHandler | None = None) -> None:
        self.workdir = str(workspace)
        self.local = LocalSandbox(workspace)
        self.command_handler = command_handler
        self.commands: tuple[tuple[str, ...], ...] = ()
        self.close_count = 0

    async def exec(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> Process:
        raise AssertionError("An in-process harness must not start a subprocess")

    async def run(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CompletedRun:
        self.commands = (*self.commands, tuple(cmd))
        if self.command_handler is None:
            return CompletedRun(stdout="ok", stderr="", exit_code=0)
        return await self.command_handler(self, cmd)

    async def read(self, path: str) -> bytes:
        return await self.local.read(path)

    async def write(self, path: str, data: bytes) -> None:
        await self.local.write(path, data)

    def host_url(self, port: int) -> str:
        return self.local.host_url(port)

    async def which(self, binary: str) -> str | None:
        return None

    async def snapshot(self) -> Mapping[str, str]:
        return await self.local.snapshot()

    async def tempdir(self) -> str:
        return await self.local.tempdir()

    async def close(self) -> None:
        self.close_count += 1
        await self.local.close()


class RecordingFactory:
    def __init__(self, command_handler: CommandHandler | None = None) -> None:
        self.command_handler = command_handler
        self.boxes: tuple[RecordingSandbox, ...] = ()

    def __call__(self, workspace: Path) -> RecordingSandbox:
        box: Final = RecordingSandbox(workspace, self.command_handler)
        self.boxes = (*self.boxes, box)
        return box


def _target(label: str = "agent", response: str | ModelResponse | Exception = "done") -> EvalTarget:
    return EvalTarget(
        Harness.TOOL_LOOP,
        "openai/eval-agent",
        label=label,
        options=ToolLoopOptions(completion_kwargs={"mock_response": response}),
    )


def _passes(result: EvalOutput) -> bool:
    return result.text == "done"


async def test_should_evaluate_a_text_task_with_default_execution_settings() -> None:
    report: Final = await aagent_eval(
        (EvalTask("answer", "Answer", check=PredicateCheck(_passes)),),
        (_target(),),
    )

    assert len(report.runs) == 1
    assert report.runs[0].status is EvalStatus.PASS
    assert report.runs[0].trial == 1
    assert report.runs[0].stop_reason == "done"
    assert report.runs[0].result is not None
    assert report.runs[0].result.text == "done"
    assert report.summaries[0].total == 1


async def test_should_grade_a_single_attempt_before_cleaning_its_independent_workspace(tmp_path: Path) -> None:
    original: Final = tmp_path / "project"
    original.mkdir()
    (original / "state.txt").write_text("original")

    async def command(box: RecordingSandbox, cmd: Sequence[str]) -> CompletedRun:
        if tuple(cmd) == ("prepare",):
            assert await box.read("state.txt") == b"original"
            await box.write("state.txt", b"changed")
            return CompletedRun(stdout="prepared", stderr="", exit_code=0)
        assert await box.read("state.txt") == b"changed"
        assert await box.read("nested/input.txt") == b"overlay"
        assert box.close_count == 0
        return CompletedRun(stdout="verified", stderr="", exit_code=0)

    factory: Final = RecordingFactory(command)
    run: Final = await aagent_eval_attempt(
        EvalTask(
            "edit",
            "Complete the task",
            check=CommandCheck(("verify",)),
            workspace=original,
            files={"nested/input.txt": "overlay"},
            setup=(("prepare",),),
        ),
        _target("candidate"),
        trial=3,
        suite_id="resumable-suite",
        sandbox=factory,
    )

    assert (run.task_id, run.target, run.trial, run.status) == ("edit", "candidate", 3, EvalStatus.PASS)
    assert run.check is not None and run.check.detail == "exit 0: verified"
    assert run.stop_reason == "done"
    assert len(factory.boxes) == 1
    assert factory.boxes[0].commands == (("prepare",), ("verify",))
    assert factory.boxes[0].close_count == 1
    assert not Path(factory.boxes[0].workdir).exists()
    assert (original / "state.txt").read_text() == "original"


async def test_should_preserve_unknown_metrics_when_a_single_attempt_cannot_start() -> None:
    def unavailable(workspace: Path) -> RecordingSandbox:
        raise SandboxError("workspace unavailable")

    run: Final = await aagent_eval_attempt(
        EvalTask("answer", "Answer", check=PredicateCheck(_passes)), _target(), sandbox=unavailable
    )

    assert run.status is EvalStatus.ERROR
    assert run.result is None and run.check is None
    assert run.cost is None and run.usage is None and run.tool_calls is None
    assert "workspace unavailable" in run.detail


async def test_should_cleanup_a_single_attempt_when_its_caller_cancels() -> None:
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def blocked(box: RecordingSandbox, cmd: Sequence[str]) -> CompletedRun:
        entered.set()
        await release.wait()
        return CompletedRun(stdout="ok", stderr="", exit_code=0)

    factory: Final = RecordingFactory(blocked)
    pending: Final = asyncio.create_task(
        aagent_eval_attempt(EvalTask("blocked", "Answer", check=CommandCheck(("verify",))), _target(), sandbox=factory)
    )
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert len(factory.boxes) == 1
    assert factory.boxes[0].close_count == 1
    assert not Path(factory.boxes[0].workdir).exists()


@pytest.mark.parametrize(
    ("trial", "suite_id", "turns", "timeout", "setup_timeout"),
    ((0, None, None, 1, 1), (1, " ", None, 1, 1), (1, None, 0, 1, 1), (1, None, None, 0, 1), (1, None, None, 1, 0)),
)
async def test_should_reject_invalid_attempt_options_before_creating_a_sandbox(
    trial: int, suite_id: str | None, turns: int | None, timeout: float, setup_timeout: float
) -> None:
    factory: Final = RecordingFactory()
    with pytest.raises(ValueError, match=r"positive|greater than zero|non-empty"):
        await aagent_eval_attempt(
            EvalTask("answer", "Answer", check=PredicateCheck(_passes)),
            _target(),
            trial=trial,
            suite_id=suite_id,
            max_turns=turns,
            timeout=timeout,
            setup_timeout=setup_timeout,
            sandbox=factory,
        )
    assert factory.boxes == ()


async def test_should_copy_each_workspace_and_run_setup_and_checks_in_the_same_sandbox(tmp_path: Path) -> None:
    original: Final = tmp_path / "project"
    original.mkdir()
    (original / "state.txt").write_text("original")
    observed: Final[asyncio.Queue[bytes]] = asyncio.Queue()

    async def command(box: RecordingSandbox, cmd: Sequence[str]) -> CompletedRun:
        if tuple(cmd) == ("prepare",):
            observed.put_nowait(await box.read("state.txt"))
            await box.write("state.txt", b"changed")
            return CompletedRun(stdout="prepared", stderr="", exit_code=0)
        assert await box.read("state.txt") == b"changed"
        assert await box.read("nested/input.txt") == b"overlay"
        return CompletedRun(stdout="verified", stderr="", exit_code=0)

    factory: Final = RecordingFactory(command)
    task: Final = EvalTask(
        "edit",
        "Complete the task",
        check=CommandCheck(("verify",)),
        workspace=original,
        files={"nested/input.txt": "overlay"},
        setup=(("prepare",),),
    )
    report: Final = await aagent_eval(
        (task,), (_target("first"), _target("second")), trials=2, max_concurrency=2, sandbox=factory
    )

    assert tuple(run.status for run in report.runs) == (EvalStatus.PASS,) * 4
    assert tuple((run.task_id, run.target, run.trial) for run in report.runs) == (
        ("edit", "first", 1),
        ("edit", "first", 2),
        ("edit", "second", 1),
        ("edit", "second", 2),
    )
    assert tuple(observed.get_nowait() for _ in range(4)) == (b"original",) * 4
    assert (original / "state.txt").read_text() == "original"
    assert len({box.workdir for box in factory.boxes}) == 4
    assert all(box.commands == (("prepare",), ("verify",)) for box in factory.boxes)
    assert all(box.close_count == 1 and not Path(box.workdir).exists() for box in factory.boxes)


@pytest.mark.parametrize("link_kind", ("file", "directory", "relative", "broken", "root", "cycle"))
async def test_should_reject_workspace_symbolic_links_before_creating_a_sandbox(tmp_path: Path, link_kind: str) -> None:
    host: Final = tmp_path / "host"
    host.mkdir()
    host_file: Final = host / "record.txt"
    host_file.write_text("host data")
    original: Final = tmp_path / "project"
    original.mkdir()
    (original / "notes.txt").write_text("task data")
    workspace: Final = tmp_path / "workspace" if link_kind == "root" else original

    if link_kind == "root":
        workspace.symlink_to(original, target_is_directory=True)
    elif link_kind == "directory":
        (original / "linked").symlink_to(host, target_is_directory=True)
    elif link_kind == "relative":
        (original / "linked").symlink_to("notes.txt")
    elif link_kind == "broken":
        (original / "linked").symlink_to(host / "missing.txt")
    elif link_kind == "cycle":
        (original / "linked").symlink_to(".", target_is_directory=True)
    else:
        (original / "linked").symlink_to(host_file)

    factory: Final = RecordingFactory()
    task: Final = EvalTask(
        "linked-workspace",
        "Complete the task",
        check=PredicateCheck(_passes),
        workspace=workspace,
        files={"linked/record.txt": "overlay"} if link_kind == "directory" else {},
    )
    report: Final = await aagent_eval((task,), (_target(),), sandbox=factory)

    assert report.runs[0].status is EvalStatus.ERROR
    assert report.runs[0].result is None
    assert "symbolic link" in report.runs[0].detail.lower()
    assert factory.boxes == ()
    assert host_file.read_text() == "host data"
    assert (original / "notes.txt").read_text() == "task data"


async def test_should_preserve_failed_and_error_rows_and_finish_the_other_tasks() -> None:
    async def command(box: RecordingSandbox, cmd: Sequence[str]) -> CompletedRun:
        return CompletedRun(stdout="", stderr="setup rejected", exit_code=2)

    factory: Final = RecordingFactory(command)
    tasks: Final = (
        EvalTask("pass", "Answer", check=PredicateCheck(_passes)),
        EvalTask("fail", "Answer", check=PredicateCheck(lambda result: False)),
        EvalTask("error", "Answer", check=PredicateCheck(_passes), setup=(("setup",),)),
    )
    report: Final = await aagent_eval(tasks, (_target(),), sandbox=factory)

    assert tuple(run.status for run in report.runs) == (EvalStatus.PASS, EvalStatus.FAIL, EvalStatus.ERROR)
    assert report.runs[0].result is not None
    assert report.runs[1].result is not None
    assert report.runs[2].result is None
    assert "setup" in report.runs[2].detail.lower()
    summary: Final = report.summaries[0]
    assert summary.total == 3 and summary.passed == 1 and summary.errors == 1
    assert summary.pass_rate == 1 / 3
    assert not summary.metrics_complete
    assert all(box.close_count == 1 for box in factory.boxes)


async def test_should_run_async_predicates_before_closing_the_sandbox() -> None:
    factory: Final = RecordingFactory()

    async def check(result: EvalOutput) -> bool:
        assert factory.boxes[0].close_count == 0
        return result.text == "done"

    report: Final = await aagent_eval(
        (EvalTask("async", "Answer", check=PredicateCheck(check)),), (_target(),), sandbox=factory
    )

    assert report.runs[0].status is EvalStatus.PASS
    assert factory.boxes[0].close_count == 1


async def test_should_not_pass_an_agent_that_reaches_its_turn_limit() -> None:
    response: Final = ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "ping", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    factory: Final = RecordingFactory()
    report: Final = await aagent_eval(
        (EvalTask("limited", "Keep calling tools", check=PredicateCheck(lambda result: True)),),
        (_target(response=response),),
        max_turns=1,
        sandbox=factory,
    )

    assert report.runs[0].stop_reason == "max_turns"
    assert report.runs[0].status is EvalStatus.FAIL


async def test_should_record_sandbox_startup_failure_without_aborting_other_targets() -> None:
    factory: Final = RecordingFactory()

    class StartupFactory:
        def __init__(self) -> None:
            self.attempts = 0

        def __call__(self, workspace: Path) -> RecordingSandbox:
            self.attempts += 1
            if self.attempts == 1:
                raise SandboxError("workspace unavailable")
            return factory(workspace)

    report: Final = await aagent_eval(
        (EvalTask("answer", "Answer", check=PredicateCheck(_passes)),),
        (_target("unavailable"), _target("working")),
        sandbox=StartupFactory(),
    )

    assert tuple(run.status for run in report.runs) == (EvalStatus.ERROR, EvalStatus.PASS)
    assert report.runs[0].result is None
    assert "workspace unavailable" in report.runs[0].detail
    assert all(box.close_count == 1 for box in factory.boxes)


async def test_should_record_checker_exceptions_and_preserve_the_agent_result() -> None:
    def broken_check(result: EvalOutput) -> bool:
        raise ValueError("invalid evaluation input")

    report: Final = await aagent_eval(
        (EvalTask("checker", "Answer", check=PredicateCheck(broken_check)),),
        (_target(),),
        sandbox=RecordingFactory(),
    )

    assert report.runs[0].status is EvalStatus.ERROR
    assert report.runs[0].result is not None
    assert report.runs[0].result.text == "done"
    assert "invalid evaluation input" in report.runs[0].detail


async def test_should_cleanup_on_command_timeout() -> None:
    async def timed_out(box: RecordingSandbox, cmd: Sequence[str]) -> CompletedRun:
        raise SandboxError("command timed out")

    factory: Final = RecordingFactory(timed_out)
    report: Final = await aagent_eval(
        (EvalTask("timeout", "Answer", check=CommandCheck(("verify",))),),
        (_target(),),
        sandbox=factory,
    )

    assert report.runs[0].status is EvalStatus.ERROR
    assert report.runs[0].result is not None
    assert "timed out" in report.runs[0].detail
    assert factory.boxes[0].close_count == 1


async def test_should_bound_concurrency_across_grading_and_propagate_suite_cancellation() -> None:
    entered: Final[asyncio.Queue[str]] = asyncio.Queue()
    release: Final = asyncio.Event()

    async def blocked(box: RecordingSandbox, cmd: Sequence[str]) -> CompletedRun:
        entered.put_nowait(box.workdir)
        await release.wait()
        return CompletedRun(stdout="ok", stderr="", exit_code=0)

    factory: Final = RecordingFactory(blocked)
    suite: Final = asyncio.create_task(
        aagent_eval(
            (EvalTask("blocked", "Answer", check=CommandCheck(("verify",))),),
            (_target(),),
            trials=3,
            max_concurrency=2,
            sandbox=factory,
        )
    )
    await entered.get()
    await entered.get()
    assert len(factory.boxes) == 2
    assert entered.empty()
    suite.cancel()

    with pytest.raises(asyncio.CancelledError):
        await suite

    assert len(factory.boxes) == 2
    assert all(box.close_count == 1 and not Path(box.workdir).exists() for box in factory.boxes)


@pytest.mark.parametrize(
    ("trials", "concurrency", "turns", "timeout", "setup_timeout"),
    ((0, 1, None, 1, 1), (1, 0, None, 1, 1), (1, 1, 0, 1, 1), (1, 1, None, 0, 1), (1, 1, None, 1, 0)),
)
async def test_should_reject_invalid_limits_before_creating_a_sandbox(
    trials: int, concurrency: int, turns: int | None, timeout: float, setup_timeout: float
) -> None:
    factory: Final = RecordingFactory()
    with pytest.raises(ValueError, match=r"positive|greater than zero"):
        await aagent_eval(
            (EvalTask("answer", "Answer", check=PredicateCheck(_passes)),),
            (_target(),),
            sandbox=factory,
            trials=trials,
            max_concurrency=concurrency,
            max_turns=turns,
            timeout=timeout,
            setup_timeout=setup_timeout,
        )
    assert factory.boxes == ()


@pytest.mark.parametrize("duplicate", ("task", "target"))
async def test_should_reject_duplicate_identifiers_before_creating_a_sandbox(duplicate: str) -> None:
    task: Final = EvalTask("answer", "Answer", check=PredicateCheck(_passes))
    target: Final = _target()
    factory: Final = RecordingFactory()

    with pytest.raises(ValueError, match="task ids" if duplicate == "task" else "target labels"):
        await aagent_eval(
            (task, task) if duplicate == "task" else (task,),
            (target, target) if duplicate == "target" else (target,),
            sandbox=factory,
        )

    assert factory.boxes == ()


@pytest.mark.parametrize("invalid", ("workspace", "parent-file", "absolute-file", "setup"))
async def test_should_reject_invalid_workspace_inputs_before_creating_a_sandbox(tmp_path: Path, invalid: str) -> None:
    factory: Final = RecordingFactory()
    workspace: Final = tmp_path / "missing" if invalid == "workspace" else None
    files: Final[Mapping[str, str]] = (
        {"../outside.txt": "invalid"}
        if invalid == "parent-file"
        else {str(tmp_path / "outside.txt"): "invalid"}
        if invalid == "absolute-file"
        else {}
    )
    setup: Final = ("not an argument vector",) if invalid == "setup" else ()
    with pytest.raises(ValueError, match=r"workspace|relative|command"):
        await aagent_eval(
            (
                EvalTask(
                    "answer", "Answer", check=PredicateCheck(_passes), workspace=workspace, files=files, setup=setup
                ),
            ),
            (_target(),),
            sandbox=factory,
        )

    assert factory.boxes == ()
    assert not (tmp_path / "outside.txt").exists()


def test_should_expose_sync_execution_with_the_same_results() -> None:
    report: Final = agent_eval(
        (EvalTask("answer", "Answer", check=PredicateCheck(_passes)),),
        (_target(),),
        trials=2,
        sandbox=RecordingFactory(),
    )

    assert tuple(run.status for run in report.runs) == (EvalStatus.PASS, EvalStatus.PASS)
    assert report.summaries[0].pass_rate == 1
    assert tuple(run.result.text for run in report.runs if run.result is not None) == ("done", "done")


async def test_should_direct_async_callers_to_the_async_entrypoint() -> None:
    with pytest.raises(RuntimeError, match="aagent_eval"):
        agent_eval(
            (EvalTask("answer", "Answer", check=PredicateCheck(_passes)),),
            (_target(),),
            sandbox=RecordingFactory(),
        )


async def test_should_compare_custom_agents_and_harness_targets_in_one_suite() -> None:
    observed: Final[asyncio.Queue[AgentInput]] = asyncio.Queue()

    class CustomAgent(EvalAgent):
        async def run(self, task: AgentInput) -> AgentOutput:
            observed.put_nowait(task)
            return AgentOutput(task.prompt)

    report: Final = await aagent_eval(
        (EvalTask("answer", "done", check=PredicateCheck(_passes)),),
        (_target("builtin"), CustomAgent("custom")),
        trials=2,
        max_concurrency=2,
        metadata={"experiment": "mixed"},
    )

    assert tuple((run.target, run.trial, run.status) for run in report.runs) == (
        ("builtin", 1, EvalStatus.PASS),
        ("builtin", 2, EvalStatus.PASS),
        ("custom", 1, EvalStatus.PASS),
        ("custom", 2, EvalStatus.PASS),
    )
    inputs: Final = tuple(observed.get_nowait() for _ in range(2))
    assert {task.trial for task in inputs} == {1, 2}
    assert all(task.task_id == "answer" and task.metadata["experiment"] == "mixed" for task in inputs)
    assert all(task.metadata["agent_eval_suite"] == report.suite_id for task in inputs)
    assert all(not Path(task.sandbox.workdir).exists() for task in inputs)
    summary: Final = report.summaries[1]
    assert summary.cost is None and summary.usage is None and summary.tool_calls is None
    assert not summary.metrics_complete
    assert report.runs[2].cost is None and report.runs[2].usage is None and report.runs[2].tool_calls is None
    assert '"cost": null' in report.to_json()
    assert "unknown" in report.to_markdown()


async def test_should_run_custom_agents_and_command_checks_in_the_prepared_sandbox(tmp_path: Path) -> None:
    original: Final = tmp_path / "project"
    original.mkdir()
    (original / "state.txt").write_text("original")

    async def command(box: RecordingSandbox, cmd: Sequence[str]) -> CompletedRun:
        if tuple(cmd) == ("prepare",):
            await box.write("state.txt", b"prepared")
        else:
            assert await box.read("answer.txt") == b"prepared"
            assert box.close_count == 0
        return CompletedRun(stdout="verified", stderr="", exit_code=0)

    class CustomAgent(EvalAgent):
        async def run(self, task: AgentInput) -> AgentOutput:
            assert task.trial == 3 and task.task_id == "workspace"
            assert task.metadata["agent_eval_suite"] == "custom-suite"
            assert task.metadata["agent_eval_trial"] == task.trial
            await task.sandbox.write("answer.txt", await task.sandbox.read("state.txt"))
            return AgentOutput(
                "done",
                cost=0.25,
                usage=Usage(3, 2, 1),
                events=(ToolCall("read", "read", "read", {}), ToolResult("read", "prepared")),
                files=(FileChange("answer.txt", "created"),),
            )

    factory: Final = RecordingFactory(command)
    run: Final = await aagent_eval_attempt(
        EvalTask("workspace", "Answer", check=CommandCheck(("verify",)), workspace=original, setup=(("prepare",),)),
        CustomAgent("custom", model="declared-model"),
        trial=3,
        suite_id="custom-suite",
        sandbox=factory,
    )
    assert run.status is EvalStatus.PASS
    assert run.cost == 0.25 and run.usage == Usage(3, 2, 1) and run.tool_calls == 1
    assert run.result is not None and run.result.files[0].path == "answer.txt"
    assert factory.boxes[0].commands == (("prepare",), ("verify",))
    assert factory.boxes[0].close_count == 1
    assert not Path(factory.boxes[0].workdir).exists()
    assert (original / "state.txt").read_text() == "original"
    assert not (original / "answer.txt").exists()


@pytest.mark.parametrize("failure", ("exception", "timeout", "turn-limit"))
async def test_should_preserve_custom_agent_failures_and_cleanup(failure: str) -> None:
    class CustomAgent(EvalAgent):
        async def run(self, task: AgentInput) -> AgentOutput:
            if failure == "exception":
                raise RuntimeError("custom agent failed")
            if failure == "timeout":
                await asyncio.Event().wait()
            return AgentOutput("partial", stop_reason="max_turns")

    factory: Final = RecordingFactory()
    run: Final = await aagent_eval_attempt(
        EvalTask("failure", "Answer", check=PredicateCheck(lambda result: True)),
        CustomAgent("custom"),
        timeout=0.001 if failure == "timeout" else 300,
        sandbox=factory,
    )
    assert run.status is (EvalStatus.FAIL if failure == "turn-limit" else EvalStatus.ERROR)
    assert run.check is None
    assert run.stop_reason == ("max_turns" if failure == "turn-limit" else None)
    assert run.cost is None and run.usage is None and run.tool_calls is None
    assert "TimeoutError" in run.detail if failure == "timeout" else "agent" in run.detail
    assert factory.boxes[0].close_count == 1
    assert not Path(factory.boxes[0].workdir).exists()


async def test_should_cleanup_custom_agents_when_the_caller_cancels() -> None:
    entered: Final = asyncio.Event()

    class CustomAgent(EvalAgent):
        async def run(self, task: AgentInput) -> AgentOutput:
            entered.set()
            await asyncio.Event().wait()
            return AgentOutput("unreachable")

    factory: Final = RecordingFactory()
    pending: Final = asyncio.create_task(
        aagent_eval_attempt(
            EvalTask("cancelled", "Answer", check=PredicateCheck(_passes)), CustomAgent("custom"), sandbox=factory
        )
    )
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert factory.boxes[0].close_count == 1
    assert not Path(factory.boxes[0].workdir).exists()
