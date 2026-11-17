from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import CommandCheck, EvalScorecard, EvalStatus, EvalTarget, EvalTask, JudgeCheck, PredicateCheck
from litellm.harness.evaluation.types import CheckResult, EvalRun
from litellm.harness.types import Harness, Result, ToolCall, Usage

_OBJECT: Final = TypeAdapter(dict[str, object])
_ROWS: Final = TypeAdapter(tuple[dict[str, object], ...])


def test_should_aggregate_known_usage_and_serialize_unknown_metrics_without_authentication_configuration() -> None:
    result: Final = Result(
        text="answer",
        output=None,
        files=[],
        events=[ToolCall(id="one", name="lookup", native_name="lookup", input={})],
        usage=Usage(input_tokens=10, output_tokens=4, calls=2),
        cost=0.25,
        stop_reason="done",
        session_id="session",
    )
    judged: Final = CheckResult(passed=True, detail="rubric passed", judge_usage=Usage(2, 1, 1), judge_cost=0.1)
    report: Final = EvalScorecard(
        suite_id="suite",
        runs=(
            EvalRun("first", "task", "target", 1, EvalStatus.PASS, result, judged, 2, "rubric passed"),
            EvalRun("second", "task", "target", 2, EvalStatus.FAIL, result, None, 4, "check failed"),
            EvalRun("third", "task", "target", 3, EvalStatus.ERROR, None, None, 0, "startup failed"),
        ),
    )

    summary: Final = report.summaries[0]
    assert summary.total == 3 and summary.passed == 1 and summary.errors == 1
    assert summary.pass_rate == 1 / 3
    assert summary.cost == 0.5
    assert summary.usage == Usage(input_tokens=20, output_tokens=8, calls=4)
    assert summary.tool_calls == 2
    assert summary.mean_duration == 2
    assert summary.judge_cost == 0.1
    assert summary.judge_usage == Usage(input_tokens=2, output_tokens=1, calls=1)
    assert not summary.metrics_complete

    payload: Final = _OBJECT.validate_json(report.to_json())
    runs: Final = _ROWS.validate_python(payload["runs"])
    summaries: Final = _ROWS.validate_python(payload["summaries"])
    assert tuple(run["status"] for run in runs) == ("pass", "fail", "error")
    assert runs[2]["cost"] is None and runs[2]["usage"] is None
    assert summaries[0]["cost"] == summary.cost
    assert summaries[0]["judge_cost"] == summary.judge_cost
    assert set(payload) == {"suite_id", "runs", "summaries"}
    assert "events" not in runs[0]
    assert "Agent cost (USD)" in report.to_markdown()
    assert "Judge cost (USD)" in report.to_markdown()
    assert "unknown" in report.to_markdown()


def test_should_escape_multiline_labels_and_check_output_in_markdown() -> None:
    report: Final = EvalScorecard(
        suite_id="suite",
        runs=(EvalRun("run", "task|name", "target\nname", 1, EvalStatus.ERROR, None, None, 0, "bad|output\nnext"),),
    )

    markdown: Final = report.to_markdown()
    assert "task\\|name" in markdown
    assert "target name" in markdown
    assert "bad\\|output next" in markdown


@pytest.mark.parametrize("command", ((), "pytest -q", ("",)))
def test_should_reject_commands_without_an_argument_vector(command: tuple[str, ...] | str) -> None:
    with pytest.raises(ValueError, match="command"):
        CommandCheck(command)


@pytest.mark.parametrize("timeout", (0, -1, float("inf"), float("nan")))
def test_should_reject_unbounded_or_nonpositive_check_deadlines(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        CommandCheck(("verify",), timeout=timeout)
    with pytest.raises(ValueError, match="timeout"):
        JudgeCheck("Evaluate the result", "openai/eval-judge", timeout=timeout)


def test_should_derive_distinct_default_target_labels_and_require_task_inputs() -> None:
    assert (
        EvalTarget(Harness.CODEX, "openai/eval-agent").name != EvalTarget(Harness.CLAUDE_CODE, "openai/eval-agent").name
    )
    with pytest.raises(ValueError, match="task id"):
        EvalTask("", "Answer", check=PredicateCheck(lambda result: True))
    with pytest.raises(ValueError, match="prompt"):
        EvalTask("task", "", check=PredicateCheck(lambda result: True))


def test_should_snapshot_caller_owned_inputs_before_later_trials_can_observe_mutations() -> None:
    def tool() -> str:
        return "done"

    files: Final = {"input.txt": "before"}  # mutable-ok: change caller inputs to verify construction snapshots
    command: Final = ["verify"]  # mutable-ok: change caller inputs to verify construction snapshots
    setup_command: Final = ["prepare"]  # mutable-ok: change caller inputs to verify construction snapshots
    setup: Final[list[Sequence[str]]] = [
        setup_command
    ]  # mutable-ok: change caller inputs to verify construction snapshots
    tools: Final[list[Callable[..., object]]] = [
        tool
    ]  # mutable-ok: change caller inputs to verify construction snapshots
    check: Final = CommandCheck(command)
    task: Final = EvalTask("task", "Answer", check=check, files=files, setup=setup)
    target: Final = EvalTarget(Harness.TOOL_LOOP, "openai/eval-agent", tools=tools)

    files["input.txt"] = "after"
    command.append("changed")
    setup_command.append("changed")
    setup.append(("another-command",))
    tools.clear()

    assert check.command == ("verify",)
    assert dict(task.files) == {"input.txt": "before"}
    assert task.setup == (("prepare",),)
    assert target.tools == (tool,)
