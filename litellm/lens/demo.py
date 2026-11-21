from __future__ import annotations

import asyncio
import csv
from functools import partial
from pathlib import Path
from typing import Final

from litellm.harness.evaluation import EvalContext
from litellm.harness.types import Result, Text, ToolCall, ToolResult
from litellm.lens.adapter import AgentInput, AgentOutput, python_agent

_TICKETS: Final = (
    ("I was charged twice for my subscription.", "billing"),
    ("The app crashes every time I sign in.", "technical"),
    ("Can I get a quote for twenty seats?", "sales"),
)


async def fixture_agent(input: AgentInput, *, delay: float = 0.12) -> AgentOutput:
    index: Final = int(input.task_id.removeprefix("ticket-"))
    second_model: Final = input.model.endswith("/model-b")
    ticket: Final = _TICKETS[(index - 1) % len(_TICKETS)]
    call: Final = ToolCall(
        "lookup-1",
        "lookup_ticket",
        "lookup_ticket",
        {"ticket_id": input.task_id, "source": "support"},
    )
    reply: Final = ToolResult(call.id, "Fixture ticket found")
    duplicate: Final = ToolCall(
        "lookup-2",
        "lookup_ticket",
        "lookup_ticket",
        {"source": "support", "ticket_id": input.task_id},
    )
    duplicate_reply: Final = ToolResult(duplicate.id, "Fixture ticket found")
    input.emit(call)
    await asyncio.sleep(delay)
    input.emit(reply)
    if second_model and index % 3 == 1:
        input.emit(duplicate)
        await asyncio.sleep(delay)
        input.emit(duplicate_reply)
    if second_model and index % 11 == 0:
        error_call: Final = ToolCall(
            "lookup-error",
            "lookup_ticket",
            "lookup_ticket",
            {"ticket_id": input.task_id, "source": "support"},
        )
        error_reply: Final = ToolResult(error_call.id, "Fixture lookup timed out", is_error=True)
        input.emit(error_call)
        input.emit(error_reply)
        return AgentOutput(
            text="Fixture model endpoint timed out",
            events=(call, reply, error_call, error_reply),
            cost=0.0006,
            model_calls=0,
            stop_reason="runtime_error",
        )
    answer: Final = _TICKETS[index % len(_TICKETS)][1] if second_model and index % 4 == 1 else ticket[1]
    input.emit(Text(answer))
    await asyncio.sleep(delay)
    return AgentOutput(
        text=answer,
        events=((call, reply, duplicate, duplicate_reply) if second_model and index % 3 == 1 else (call, reply)),
        cost=0.0006 if second_model else 0.003,
        model_calls=0,
    )


async def fixture_runner(context: EvalContext, *, delay: float = 0.12) -> Result:
    runner: Final = python_agent(partial(fixture_agent, delay=delay))
    return await runner(context)


def create_demo(directory: Path, count: int = 12, delay: float = 0.12) -> tuple[Path, Path]:
    if count < 1 or delay < 0:
        raise ValueError("Demo count must be positive and delay must be non-negative")
    tasks: Final = directory / "tasks.csv"
    config: Final = directory / "evals.py"
    if tasks.exists() or config.exists():
        raise FileExistsError("Demo files already exist. Choose another directory.")
    directory.mkdir(parents=True, exist_ok=True)
    with tasks.open("x", newline="", encoding="utf-8") as destination:
        writer: Final = csv.writer(destination)
        writer.writerow(("id", "prompt", "expected"))
        writer.writerows(
            (
                f"ticket-{index:04d}",
                "Classify this ticket as billing, technical, or sales. Reply with only the category. Ticket: "
                + _TICKETS[(index - 1) % len(_TICKETS)][0],
                _TICKETS[(index - 1) % len(_TICKETS)][1],
            )
            for index in range(1, count + 1)
        )
    config.write_text(
        "from litellm import EvalTarget, EvalTask, Harness, PredicateCheck\n"
        "from litellm.lens import AgentInput, AgentOutput, python_agent\n"
        "from litellm.lens.core import TaskRow\n"
        "from litellm.lens.demo import fixture_agent\n"
        "\n"
        "fixture = True\n"
        'agent_name = "SupportAgent"\n'
        'telemetry_complete = frozenset({"model-a", "model-b"})\n'
        "\n"
        "\n"
        "def make_task(row: TaskRow) -> EvalTask:\n"
        "    return EvalTask(\n"
        "        id=row.id, prompt=row.prompt,\n"
        "        check=PredicateCheck(lambda result: result.text.strip().casefold() == row.expected.strip().casefold()),\n"
        "    )\n"
        "\n"
        "\n"
        "async def run_agent(input: AgentInput) -> AgentOutput:\n"
        f"    return await fixture_agent(input, delay={delay!r})\n"
        "\n"
        "\n"
        "targets = [\n"
        '    EvalTarget(Harness.TOOL_LOOP, "fixture/model-a", label="model-a", runner=python_agent(run_agent)),\n'
        '    EvalTarget(Harness.TOOL_LOOP, "fixture/model-b", label="model-b", runner=python_agent(run_agent)),\n'
        "]\n",
        encoding="utf-8",
    )
    return tasks, config
