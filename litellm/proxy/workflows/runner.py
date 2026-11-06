import asyncio
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final, Protocol

from pydantic import ConfigDict, JsonValue, TypeAdapter

from litellm.models.workflow import (
    WorkflowCheckpoint,
    WorkflowConflict,
    WorkflowExecution,
    WorkflowRunState,
    WorkflowStep,
    WorkflowStore,
)
from litellm.models.workflow_bundle import (
    Binding,
    ChatCompletionNode,
    Constant,
    WorkflowBundle,
    WorkflowFailure,
    WorkflowNode,
)
from litellm.proxy.workflows.monty.compiler import compile_program
from litellm.proxy.workflows.monty.models import MontyFailure
from litellm.proxy.workflows.monty.runtime import MontyRuntime
from litellm.proxy.workflows.validation import validate_value
from litellm.types.llms.base import LiteLLMBaseModel


class ChildResult(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    output: JsonValue
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_cost: float | None = None
    request_id: str | None = None


class ChildCaller(Protocol):
    async def call(
        self, execution: WorkflowExecution, node: ChatCompletionNode, messages: JsonValue, remaining: float
    ) -> ChildResult | WorkflowFailure: ...


_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def resolve(binding: Binding, messages: JsonValue, outputs: Mapping[str, JsonValue]) -> JsonValue | WorkflowFailure:
    if isinstance(binding, Constant):
        return binding.value
    if binding.source != "$input" and binding.source not in outputs:
        return WorkflowFailure(
            code="missing_result", message="Referenced predecessor has no result", param=binding.source
        )
    value: Final = messages if binding.source == "$input" else outputs[binding.source]
    if binding.path in (None, ""):
        return value
    return project(value, tuple(part.replace("~1", "/").replace("~0", "~") for part in binding.path[1:].split("/")))


def project(value: JsonValue, parts: tuple[str, ...]) -> JsonValue | WorkflowFailure:
    if not parts:
        return value
    key: Final = parts[0]
    if isinstance(value, dict) and key in value:
        return project(value[key], parts[1:])
    if isinstance(value, list) and key.isdigit() and str(int(key)) == key and int(key) < len(value):
        return project(value[int(key)], parts[1:])
    return WorkflowFailure(code="missing_path", message="Referenced JSON Pointer has no value")


def node_input(
    node: WorkflowNode, messages: JsonValue, outputs: Mapping[str, JsonValue]
) -> JsonValue | WorkflowFailure:
    if isinstance(node, ChatCompletionNode):
        return resolve(node.messages, messages, outputs)
    arguments: Final = {name: resolve(binding, messages, outputs) for name, binding in node.inputs.items()}
    failures: Final = tuple(value for value in arguments.values() if isinstance(value, WorkflowFailure))
    if failures:
        return failures[0]
    return _JSON.validate_python(arguments)


async def execute_attempt(
    caller: ChildCaller,
    runtime: MontyRuntime,
    execution: WorkflowExecution,
    node: WorkflowNode,
    input: JsonValue,
    remaining: float,
) -> ChildResult | WorkflowFailure:
    if isinstance(node, ChatCompletionNode):
        return await caller.call(execution, node, input, remaining)
    compiled: Final = compile_program(node.program, runtime.limits)
    if isinstance(compiled, MontyFailure):
        return WorkflowFailure(code="invalid_program", message=compiled.message, param=node.id)
    result: Final = await runtime.run_program(compiled, input, deadline_monotonic=time.monotonic() + remaining)
    if isinstance(result, MontyFailure):
        return WorkflowFailure(code="python_execution_failed", message=result.message, param=node.id)
    return ChildResult(output=result.output)


async def execute_step(
    store: WorkflowStore,
    caller: ChildCaller,
    runtime: MontyRuntime,
    execution: WorkflowExecution,
    node: WorkflowNode,
    step: WorkflowStep,
) -> ChildResult | WorkflowFailure:
    remaining: Final = (min(step.deadline, execution.deadline) - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return WorkflowFailure(
            code="node_deadline", message="Node total deadline exceeded", param=node.id, status_code=504
        )
    if step.attempts >= node.retry.max_attempts:
        return WorkflowFailure(
            code="attempts_exhausted", message="Node attempt policy exhausted", param=node.id, status_code=502
        )
    attempt: Final = await store.attempt_step(step.step_id, execution.lease_owner or "")
    if isinstance(attempt, WorkflowConflict):
        return WorkflowFailure(code="lease_lost", message=attempt.message, status_code=503)
    attempt_remaining: Final = (min(step.deadline, execution.deadline) - datetime.now(timezone.utc)).total_seconds()
    try:
        async with asyncio.timeout(max(attempt_remaining, 0)):
            result: Final = await execute_attempt(caller, runtime, execution, node, step.input, attempt_remaining)
    except TimeoutError:
        return WorkflowFailure(
            code="node_deadline", message="Node total deadline exceeded", param=node.id, status_code=504
        )
    if not isinstance(result, WorkflowFailure) or attempt.attempts >= node.retry.max_attempts:
        return result
    if result.status_code not in {429, 500, 502, 503, 504}:
        return result
    backoff: Final = min(
        node.retry.backoff_seconds * 2.0 ** (attempt.attempts - 1),
        max((step.deadline - datetime.now(timezone.utc)).total_seconds(), 0),
    )
    await asyncio.sleep(backoff)
    return await execute_step(store, caller, runtime, execution, node, attempt)


async def execute_sequence(
    store: WorkflowStore,
    caller: ChildCaller,
    runtime: MontyRuntime,
    execution: WorkflowExecution,
    bundle: WorkflowBundle,
    messages: JsonValue,
    outputs: Mapping[str, JsonValue],
    index: int = 0,
) -> JsonValue | WorkflowFailure:
    if index == len(bundle.nodes):
        return resolve(bundle.output, messages, outputs)
    current: Final = await store.execution(execution.run_id)
    if current is None or current.cancel_requested:
        return WorkflowFailure(code="cancelled", message="Run cancellation requested", status_code=409)
    if current.lease_owner != execution.lease_owner:
        return WorkflowFailure(code="lease_lost", message="Run lease lost", status_code=503)
    node: Final = bundle.nodes[index]
    input: Final = node_input(node, messages, outputs)
    if isinstance(input, WorkflowFailure):
        return input
    deadline: Final = min(execution.deadline, datetime.now(timezone.utc) + timedelta(seconds=node.timeout_seconds))
    step: Final = await store.start_step(execution.run_id, node.id, input, deadline, execution.lease_owner or "")
    if isinstance(step, WorkflowConflict):
        return WorkflowFailure(code="lease_lost", message=step.message, status_code=503)
    if step.state is WorkflowRunState.COMPLETED:
        return await execute_sequence(
            store, caller, runtime, execution, bundle, messages, {**outputs, node.id: step.output}, index + 1
        )
    result: Final = await execute_step(store, caller, runtime, execution, node, step)
    steps: Final = await store.steps(execution.run_id)
    attempted: Final = next(item for item in steps if item.step_id == step.step_id)
    if isinstance(result, WorkflowFailure):
        await store.finish_step(
            attempted.model_copy(
                update={
                    "state": WorkflowRunState.FAILED,
                    "error": result.code,
                    "finished_at": datetime.now(timezone.utc),
                }
            ),
            execution.lease_owner or "",
        )
        return result
    completed: Final = attempted.model_copy(
        update={
            "state": WorkflowRunState.COMPLETED,
            "output": result.output,
            "finished_at": datetime.now(timezone.utc),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "response_cost": result.response_cost,
            "request_id": result.request_id,
        }
    )
    checkpointed: Final = await store.checkpoint(
        execution.run_id,
        WorkflowCheckpoint(
            node_path=node.id,
            input=step.input,
            output=result.output,
            attempts=attempted.attempts,
        ),
        completed,
        execution.lease_owner or "",
    )
    if not checkpointed:
        return WorkflowFailure(code="lease_lost", message="Unable to commit node checkpoint", status_code=503)
    return await execute_sequence(
        store, caller, runtime, execution, bundle, messages, {**outputs, node.id: result.output}, index + 1
    )


async def execute_run(
    store: WorkflowStore, caller: ChildCaller, runtime: MontyRuntime, execution: WorkflowExecution
) -> JsonValue | WorkflowFailure:
    snapshot: Final = await store.snapshot(execution.snapshot_id)
    if snapshot is None:
        return WorkflowFailure(code="snapshot_missing", message="Accepted snapshot is unavailable", status_code=503)
    bundle: Final = WorkflowBundle.model_validate_json(snapshot.bundle_json)
    request: Final = _OBJECT.validate_json(execution.request_json)
    messages: Final = request.get("messages")
    failure: Final = validate_value(messages, bundle.input_schema, "messages")
    if failure is not None:
        return failure
    return await execute_sequence(store, caller, runtime, execution, bundle, messages, {})
