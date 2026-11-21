from __future__ import annotations

import asyncio
from typing import Final

from litellm import EvalTarget, EvalTask, Harness, PredicateCheck, aagent_eval
from litellm.harness.evaluation import EvalProgress
from litellm.harness.types import ToolCall, ToolResult
from litellm.lens import AgentInput, AgentOutput, python_agent


async def test_should_call_python_agent_with_prompt_model_and_workspace_then_grade_output() -> None:
    inputs: Final[asyncio.Queue[AgentInput]] = asyncio.Queue()

    async def support_agent(input: AgentInput) -> AgentOutput:
        assert input.workspace.is_dir()
        inputs.put_nowait(input)
        return AgentOutput(text="billing", cost=0.04, model_calls=2)

    report: Final = await aagent_eval(
        (
            EvalTask(
                "ticket",
                "I was charged twice",
                check=PredicateCheck(lambda result: result.text == "billing"),
            ),
        ),
        (
            EvalTarget(
                Harness.TOOL_LOOP,
                "my-gateway/model",
                runner=python_agent(support_agent),
            ),
        ),
    )
    observed: Final = inputs.get_nowait()
    assert (observed.task_id, observed.prompt, observed.model) == (
        "ticket",
        "I was charged twice",
        "my-gateway/model",
    )
    assert report.runs[0].status.value == "pass"
    assert report.runs[0].cost == 0.04
    assert report.runs[0].usage is not None
    assert report.runs[0].usage.calls == 2


async def test_should_forward_live_events_once_when_output_also_contains_them() -> None:
    events: Final[asyncio.Queue[ToolCall | ToolResult]] = asyncio.Queue()
    call: Final = ToolCall("lookup-1", "lookup", "lookup", {"id": "ticket"})
    reply: Final = ToolResult(call.id, "found")

    async def agent(input: AgentInput) -> AgentOutput:
        input.emit(call)
        await asyncio.sleep(0)
        input.emit(reply)
        return AgentOutput(text="done", events=(call, reply))

    def progress(event: EvalProgress) -> None:
        if isinstance(event.event, (ToolCall, ToolResult)):
            events.put_nowait(event.event)

    report: Final = await aagent_eval(
        (
            EvalTask(
                "ticket",
                "Find ticket",
                check=PredicateCheck(lambda result: result.text == "done"),
            ),
        ),
        (EvalTarget(Harness.TOOL_LOOP, "custom/model", runner=python_agent(agent)),),
        on_progress=progress,
    )
    assert tuple(events.get_nowait() for _ in range(events.qsize())) == (call, reply)
    assert report.runs[0].result is not None
    assert tuple(report.runs[0].result.events) == (call, reply)


async def test_should_accept_simple_sync_output_and_replay_collected_events() -> None:
    call: Final = ToolCall("lookup", "lookup", "lookup", {})

    def agent(input: AgentInput) -> AgentOutput:
        return AgentOutput(text=input.prompt.upper(), events=(call,))

    report: Final = await aagent_eval(
        (
            EvalTask(
                "case",
                "done",
                check=PredicateCheck(lambda result: result.text == "DONE"),
            ),
        ),
        (EvalTarget(Harness.TOOL_LOOP, "custom/model", runner=python_agent(agent)),),
    )
    assert report.runs[0].status.value == "pass"
    assert report.runs[0].result is not None
    assert tuple(report.runs[0].result.events) == (call,)


async def test_should_record_python_harness_errors_as_case_errors() -> None:
    async def agent(input: AgentInput) -> AgentOutput:
        raise RuntimeError("lookup unavailable")

    report: Final = await aagent_eval(
        (EvalTask("case", "Find ticket", check=PredicateCheck(lambda result: True)),),
        (EvalTarget(Harness.TOOL_LOOP, "custom/model", runner=python_agent(agent)),),
    )
    assert report.runs[0].status.value == "error"
    assert "lookup unavailable" in report.runs[0].detail


async def test_should_preserve_reported_cost_when_agent_returns_runtime_failure() -> None:
    async def agent(input: AgentInput) -> AgentOutput:
        return AgentOutput(text="lookup failed", cost=0.02, model_calls=1, stop_reason="runtime_error")

    report: Final = await aagent_eval(
        (EvalTask("case", "Find ticket", check=PredicateCheck(lambda result: True)),),
        (EvalTarget(Harness.TOOL_LOOP, "custom/model", runner=python_agent(agent)),),
    )
    assert report.runs[0].status.value == "error"
    assert report.runs[0].cost == 0.02
    assert report.runs[0].check is None
    assert report.runs[0].result is not None
    assert report.runs[0].result.stop_reason == "runtime_error"
