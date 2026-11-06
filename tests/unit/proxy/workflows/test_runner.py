from typing import Final

import pytest
from pydantic import JsonValue

from litellm.models.workflow_definition import ReferenceBinding, ValueBinding, WorkflowBundle, WorkflowRetry
from litellm.models.workflow_execution import ExecutionError, RunState, StepTrace
from litellm.proxy.workflows.runner import resolve_binding
from tests.unit.proxy.workflow_definition_rows import bundle
from tests.unit.repositories.test_workflow_execution_repository import NOW, pending_run


def two_nodes() -> WorkflowBundle:
    first: Final = bundle().nodes[0].model_copy(update={"retry": WorkflowRetry(max_attempts=2)})
    second: Final = first.model_copy(update={
        "id": "node_2", "messages": ReferenceBinding(source=first.id, output="result"),
    })
    return bundle().model_copy(update={
        "nodes": (first, second), "output": ReferenceBinding(source=second.id, output="result"),
    })


@pytest.mark.parametrize("value", [None, False, 0, [], {}, [{"role": "user", "content": "next"}]])
def test_bindings_preserve_falsy_and_structured_values(value: JsonValue) -> None:
    run: Final = pending_run()
    step: Final = StepTrace(step_id="step", run_id=run.run_id, node_id="node", state=RunState.completed,
                           input=None, output=value, attempts=1, started_at=NOW, deadline=NOW)
    assert resolve_binding(ValueBinding(value=value), run, ()) == value
    assert resolve_binding(ReferenceBinding(source="node", output="result"), run, (step,)) == value
    assert resolve_binding(ReferenceBinding(source="$input", output="messages"), run, ()) == list(run.input.messages)
    assert resolve_binding(ReferenceBinding(source="missing", output="result"), run, (step,)) == ExecutionError(
        "missing_input", "Node output is unavailable"
    )
