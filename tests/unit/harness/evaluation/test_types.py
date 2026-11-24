from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import (
    AgentInput,
    AgentOutput,
    CommandCheck,
    EvalAgent,
    EvalScorecard,
    EvalStatus,
    EvalTarget,
    EvalTask,
    JudgeCheck,
    PredicateCheck,
)
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
    assert "Known subtotals" in report.to_markdown()


def test_should_report_judge_token_counts_even_when_the_judge_price_is_unknown() -> None:
    judged: Final = CheckResult(passed=True, detail="rubric passed", judge_usage=Usage(20, 5, 1), judge_cost=None)
    output: Final = AgentOutput("answer", cost=0.0, usage=Usage(), events=())
    report: Final = EvalScorecard(
        "suite", (EvalRun("one", "task", "target", 1, EvalStatus.PASS, output, judged, 1, "rubric passed"),)
    )

    summary: Final = report.summaries[0]
    assert summary.judge_usage == Usage(20, 5, 1)
    assert summary.judge_cost is None
    assert not summary.metrics_complete


@pytest.mark.parametrize("cost", (None, 0.0, 0.25))
def test_native_result_requires_known_cost_for_complete_eval_metrics(cost: float | None) -> None:
    result: Final = Result("answer", None, [], [], Usage(10, 5, 1), cost, "done", "session")
    check: Final = CheckResult(True, "passed")
    report: Final = EvalScorecard(
        "suite", (EvalRun("one", "task", "target", 1, EvalStatus.PASS, result, check, 1, "passed"),)
    )

    summary: Final = report.summaries[0]
    assert summary.cost == cost
    assert summary.usage == Usage(10, 5, 1)
    assert summary.passed == 1
    assert summary.metrics_complete is (cost is not None)


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


@pytest.mark.parametrize("cost", (-1.0, float("inf"), float("nan"), True))
def test_should_reject_invalid_custom_agent_costs(cost: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        AgentOutput("answer", cost=cost)


def test_should_require_named_custom_agents_and_readonly_identity() -> None:
    class CustomAgent(EvalAgent):
        async def run(self, task: AgentInput) -> AgentOutput:
            return AgentOutput(task.prompt)

    with pytest.raises(ValueError, match="agent name"):
        CustomAgent(" ")
    with pytest.raises(ValueError, match="agent model"):
        CustomAgent("custom", model=" ")
    agent: Final = CustomAgent("custom")
    assert agent.name == "custom" and agent.model is None
    with pytest.raises(AttributeError, match="setter"):
        setattr(agent, "name", "changed")


def test_should_keep_unreported_custom_metrics_unknown_and_explicit_zero_known() -> None:
    unknown: Final = AgentOutput("answer")
    known: Final = AgentOutput("answer", cost=0.0, usage=Usage(), events=())
    report: Final = EvalScorecard(
        "suite",
        (
            EvalRun("one", "t", "unknown", 1, EvalStatus.PASS, unknown, None, 1, "pass"),
            EvalRun("two", "t", "known", 1, EvalStatus.PASS, known, None, 1, "pass"),
            EvalRun("three", "t", "unknown", 2, EvalStatus.ERROR, None, None, 1, "startup failed"),
        ),
    )
    incomplete, complete = report.summaries
    assert incomplete.cost is None and incomplete.usage is None and incomplete.tool_calls is None
    assert not incomplete.metrics_complete
    assert (incomplete.total, incomplete.passed, incomplete.errors) == (2, 1, 1)
    assert complete.cost == 0.0 and complete.usage == Usage() and complete.tool_calls == 0
    assert complete.metrics_complete
