from typing import Final

import litellm
from litellm import EvalTarget, EvalTask, Harness, PredicateCheck
from litellm.lens import AgentInput, AgentOutput, python_agent
from litellm.lens.core import TaskRow

agent_name = "SupportAgent"


async def run_agent(input: AgentInput) -> AgentOutput:
    response: Final = await litellm.acompletion(
        model=input.model,
        messages=[{"role": "user", "content": input.prompt}],
    )
    return AgentOutput(text=response.choices[0].message.content or "")


def make_task(row: TaskRow) -> EvalTask:
    return EvalTask(
        id=row.id,
        prompt=row.prompt,
        check=PredicateCheck(lambda result: result.text.strip().casefold() == row.expected.strip().casefold()),
    )


targets = [
    EvalTarget(
        Harness.TOOL_LOOP,
        "litellm_proxy/model-a",
        label="model-a",
        runner=python_agent(run_agent),
    ),
    EvalTarget(
        Harness.TOOL_LOOP,
        "litellm_proxy/model-b",
        label="model-b",
        runner=python_agent(run_agent),
    ),
]
