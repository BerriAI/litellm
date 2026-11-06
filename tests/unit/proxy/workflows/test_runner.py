from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import pytest
from pydantic import JsonValue

from litellm.models.workflow_definition import (
    ChatCompletionNode,
    ReferenceBinding,
    ValueBinding,
    WorkflowBundle,
    WorkflowDraft,
    WorkflowRetry,
    WorkflowSnapshot,
)
from litellm.models.workflow_execution import (
    ChildCallResult,
    EngineRun,
    ExecutionError,
    RunState,
    StepTrace,
    workflow_call_id,
)
from litellm.proxy.workflows.runner import SchemaValidator, WorkflowRunner
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository
from litellm.repositories.workflow_execution_repository import WorkflowExecutionRepository
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, bundle
from tests.unit.repositories.test_workflow_execution_repository import NOW, ExecutionRows, pending_run


class FixedClock:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        return self.value


class ValidationReply:
    def __init__(self, error: ExecutionError | None = None) -> None:
        self.error: Final = error
        self.calls: tuple[tuple[JsonValue, Mapping[str, JsonValue]], ...] = ()

    async def __call__(self, messages: JsonValue, schema: Mapping[str, JsonValue]) -> ExecutionError | None:
        self.calls = (*self.calls, (messages, schema))
        return self.error


@dataclass(frozen=True, slots=True)
class ChatCall:
    key_hash: str
    team_id: str
    workflow_id: str
    node_id: str
    messages: JsonValue
    run_id: str
    session_id: str
    attempt: int
    deadline: datetime
    end_user_id: str | None


class ChatReplies:
    def __init__(
        self,
        replies: tuple[ChildCallResult | ExecutionError | Exception, ...],
        clock: FixedClock,
        elapsed: timedelta = timedelta(0),
        database: ExecutionRows | None = None,
    ) -> None:
        self.replies: Final = replies
        self.clock: Final = clock
        self.elapsed: Final = elapsed
        self.database: Final = database
        self.calls: tuple[ChatCall, ...] = ()
        self.before_dispatch: tuple[StepTrace, ...] = ()

    async def __call__(
        self,
        *,
        key_hash: str,
        team_id: str,
        workflow_id: str,
        node: ChatCompletionNode,
        messages: JsonValue,
        run_id: str,
        session_id: str,
        attempt: int,
        deadline: datetime,
        end_user_id: str | None,
    ) -> ChildCallResult | ExecutionError:
        reply: Final = self.replies[len(self.calls)]
        if self.database is not None:
            self.before_dispatch = (*self.before_dispatch, self.database.events[-1])
        self.calls = (
            *self.calls,
            ChatCall(
                key_hash, team_id, workflow_id, node.id, messages, run_id, session_id, attempt, deadline, end_user_id
            ),
        )
        self.clock.value += self.elapsed
        if isinstance(reply, Exception):
            raise reply
        return reply


def two_nodes(*, attempts: int = 1, timeout: float = 30) -> WorkflowBundle:
    first: Final = ChatCompletionNode.model_validate(bundle().nodes[0].model_dump()).model_copy(
        update={"retry": WorkflowRetry(max_attempts=attempts), "timeout_seconds": timeout}
    )
    second: Final = first.model_copy(
        update={"id": "node_2", "messages": ReferenceBinding.model_validate({"from": first.id, "output": "result"})}
    )
    return bundle().model_copy(
        update={
            "nodes": (first, second),
            "output": ReferenceBinding.model_validate({"from": second.id, "output": "result"}),
        }
    )


async def runner_inputs(
    definition: WorkflowBundle, *, deadline: datetime = NOW + timedelta(seconds=90)
) -> tuple[WorkflowDefinitionRepository, ExecutionRows, EngineRun]:
    definitions: Final = WorkflowDefinitionRepository(DefinitionRows().database)
    draft: Final = await definitions.create(definition, "team", "author")
    assert isinstance(draft, WorkflowDraft)
    snapshot: Final = await definitions.freeze_draft(draft.workflow_id, "team", draft.etag)
    assert isinstance(snapshot, WorkflowSnapshot)
    run: Final = pending_run(snapshot_id=snapshot.snapshot_id, workflow_id=snapshot.workflow_id, deadline=deadline)
    return definitions, ExecutionRows((run,)), run


async def run_once(
    repository: WorkflowExecutionRepository,
    definitions: WorkflowDefinitionRepository,
    chat: ChatReplies,
    clock: FixedClock,
    run_id: str,
    *,
    validate_input: SchemaValidator | None = None,
) -> None:
    runner: Final = WorkflowRunner(
        repository,
        definitions,
        chat,
        clock=clock,
        validate_input=ValidationReply() if validate_input is None else validate_input,
    )
    try:
        await runner._run(run_id)  # pyright: ignore[reportPrivateUsage]  # execute one claim without timed polling
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_sequential_nodes_preserve_json_values_and_record_usage_and_final_output() -> None:
    definitions, database, run = await runner_inputs(two_nodes())
    clock: Final = FixedClock()
    forwarded: Final[JsonValue] = [{"role": "user", "content": [{"type": "text", "text": "next"}]}]
    output: Final[JsonValue] = {"count": 3, "enabled": False, "missing": None, "ratios": [0.5, 1]}
    chat: Final = ChatReplies(
        (ChildCallResult(forwarded, 7, 2, "request-1", 0.01), ChildCallResult(output, 11, 3, "request-2", 0.02)),
        clock,
    )
    repository: Final = WorkflowExecutionRepository(database)
    await run_once(repository, definitions, chat, clock, run.run_id)
    finished: Final = await repository.get(run.run_id)
    assert finished is not None
    assert (finished.status, finished.output, finished.lease_token) == (RunState.COMPLETED, output, None)
    assert tuple((call.node_id, call.messages) for call in chat.calls) == (
        ("node_1", list(run.input.messages)),
        ("node_2", forwarded),
    )
    steps: Final = await repository.steps(run.run_id)
    assert tuple(
        (step.node_id, step.state, step.input, step.output, step.prompt_tokens, step.completion_tokens, step.request_id)
        for step in steps
    ) == (
        ("node_1", RunState.COMPLETED, list(run.input.messages), forwarded, 7, 2, "request-1"),
        ("node_2", RunState.COMPLETED, forwarded, output, 11, 3, "request-2"),
    )
    assert tuple(step.response_cost for step in steps) == (0.01, 0.02)
    assert tuple((step.node_id, step.state) for step in database.events) == (
        ("node_1", RunState.RUNNING),
        ("node_1", RunState.COMPLETED),
        ("node_2", RunState.RUNNING),
        ("node_2", RunState.COMPLETED),
    )


@pytest.mark.asyncio
async def test_first_nonretryable_error_stops_downstream_and_keeps_failure_trace() -> None:
    original: Final = two_nodes(attempts=3)
    independent: Final = original.nodes[1].model_copy(
        update={"messages": ValueBinding(value=[{"role": "user", "content": "independent downstream"}])}
    )
    definitions, database, run = await runner_inputs(
        original.model_copy(update={"nodes": (original.nodes[0], independent)})
    )
    clock: Final = FixedClock()
    chat: Final = ChatReplies(
        (ExecutionError("invalid_input", "Expected messages"), ChildCallResult("must not execute", 0, 0, None, None)),
        clock,
    )
    repository: Final = WorkflowExecutionRepository(database)
    await run_once(repository, definitions, chat, clock, run.run_id)
    finished: Final = await repository.get(run.run_id)
    assert finished is not None
    assert (finished.status, finished.output, finished.metadata) == (
        RunState.FAILED,
        None,
        {"error": "Expected messages"},
    )
    assert tuple((call.node_id, call.attempt) for call in chat.calls) == (("node_1", 1),)
    steps: Final = await repository.steps(run.run_id)
    assert tuple((step.node_id, step.state, step.attempts, step.error) for step in steps) == (
        ("node_1", RunState.FAILED, 1, "Expected messages"),
    )


@pytest.mark.asyncio
async def test_retry_preserves_run_session_step_and_absolute_deadline() -> None:
    definitions, database, run = await runner_inputs(
        two_nodes(attempts=2, timeout=60), deadline=NOW + timedelta(seconds=20)
    )
    clock: Final = FixedClock()
    chat: Final = ChatReplies(
        (
            ExecutionError(
                "provider_error", "Retry later", receipt=ChildCallResult(None, 4, 2, "failed-receipt", 0.125)
            ),
            ChildCallResult([{"role": "user", "content": "continue"}], 2, 1, "retried", 0.25),
            ChildCallResult("done", 3, 1, "last", None),
        ),
        clock,
        elapsed=timedelta(seconds=3),
        database=database,
    )
    repository: Final = WorkflowExecutionRepository(database)
    await run_once(repository, definitions, chat, clock, run.run_id)
    assert chat.calls[:2] == tuple(
        ChatCall(
            run.created_by,
            run.team_id,
            run.workflow_id,
            "node_1",
            [dict(message) for message in run.input.messages],
            run.run_id,
            run.session_id,
            attempt,
            run.input.deadline,
            run.input.end_user_id,
        )
        for attempt in (1, 2)
    )
    first_steps: Final = tuple(step for step in database.events if step.node_id == "node_1")
    assert len(first_steps) == 4
    assert len({step.step_id for step in first_steps}) == 1
    assert tuple(step.attempts for step in first_steps) == (1, 1, 2, 2)
    assert {step.deadline for step in first_steps} == {run.input.deadline}
    assert {step.started_at for step in first_steps} == {NOW}
    assert tuple((step.state, step.attempts, step.call_id) for step in chat.before_dispatch[:2]) == tuple(
        (RunState.RUNNING, attempt, workflow_call_id(run.run_id, "node_1", attempt)) for attempt in (1, 2)
    )
    assert len({step.call_id for step in first_steps}) == 2
    traces: Final = await repository.steps(run.run_id)
    retried: Final = next(step for step in traces if step.node_id == "node_1")
    assert (retried.prompt_tokens, retried.completion_tokens, retried.response_cost, retried.request_ids) == (
        6,
        3,
        0.375,
        ("failed-receipt", "retried"),
    )
    finished: Final = await repository.get(run.run_id)
    assert finished is not None and finished.status is RunState.COMPLETED and finished.output == "done"


@pytest.mark.asyncio
async def test_retry_budget_cannot_extend_node_deadline_or_dispatch_downstream() -> None:
    definitions, database, run = await runner_inputs(two_nodes(attempts=3, timeout=5))
    clock: Final = FixedClock()
    chat: Final = ChatReplies(
        (ExecutionError("provider_error", "Unavailable"), ChildCallResult("must not execute", 0, 0, None, None)),
        clock,
        elapsed=timedelta(seconds=5),
    )
    repository: Final = WorkflowExecutionRepository(database)
    await run_once(repository, definitions, chat, clock, run.run_id)
    assert tuple((call.node_id, call.attempt, call.deadline) for call in chat.calls) == (
        ("node_1", 1, NOW + timedelta(seconds=5)),
    )
    finished: Final = await repository.get(run.run_id)
    assert finished is not None and finished.status is RunState.FAILED and finished.output is None


@pytest.mark.asyncio
async def test_expired_workflow_never_dispatches_or_records_a_step() -> None:
    definitions, database, run = await runner_inputs(two_nodes(), deadline=NOW)
    clock: Final = FixedClock()
    chat: Final = ChatReplies((ChildCallResult("must not execute", 0, 0, None, None),), clock)
    repository: Final = WorkflowExecutionRepository(database)
    await run_once(repository, definitions, chat, clock, run.run_id)
    assert chat.calls == () and database.events == ()
    finished: Final = await repository.get(run.run_id)
    assert finished is not None
    assert (finished.status, finished.metadata) == (RunState.FAILED, {"error": "Workflow deadline expired"})


@pytest.mark.asyncio
async def test_unexpected_executor_exception_records_failed_node_and_stops_downstream() -> None:
    definitions, database, run = await runner_inputs(two_nodes())
    clock: Final = FixedClock()
    failure: Final = RuntimeError("injected executor failure")
    chat: Final = ChatReplies((failure,), clock)
    repository: Final = WorkflowExecutionRepository(database)
    await run_once(repository, definitions, chat, clock, run.run_id)
    finished: Final = await repository.get(run.run_id)
    assert finished is not None and finished.status is RunState.FAILED
    assert tuple(call.node_id for call in chat.calls) == ("node_1",)
    traces: Final = await repository.steps(run.run_id)
    assert tuple((step.node_id, step.state, step.output, step.error) for step in traces) == (
        ("node_1", RunState.FAILED, None, "The node could not complete"),
    )
    assert finished.metadata == {"error": traces[0].error}


@pytest.mark.asyncio
async def test_pinned_input_schema_rejection_fails_run_before_dispatch_or_checkpoint() -> None:
    definition: Final = two_nodes().model_copy(update={"input_schema": {"type": "array", "maxItems": 0}})
    definitions, database, run = await runner_inputs(definition)
    clock: Final = FixedClock()
    chat: Final = ChatReplies((ChildCallResult("must not execute", 0, 0, None, None),), clock)
    validation: Final = ValidationReply(ExecutionError("invalid_input", "Pinned input schema rejected messages"))
    repository: Final = WorkflowExecutionRepository(database)
    await run_once(repository, definitions, chat, clock, run.run_id, validate_input=validation)
    assert validation.calls == ((list(run.input.messages), definition.input_schema),)
    assert chat.calls == () and database.events == ()
    finished: Final = await repository.get(run.run_id)
    assert finished is not None
    assert (finished.status, finished.output, finished.metadata) == (
        RunState.FAILED,
        None,
        {"error": "Pinned input schema rejected messages"},
    )
