import asyncio
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, cast

import psycopg
import pytest
import pytest_asyncio
from fastapi import HTTPException, Request
from openai.types.chat import ChatCompletion
from prisma import Json, Prisma
from prisma.types import DatasourceOverride
from psycopg import sql

from litellm.models.workflow_definition import ChatCompletionNode, WorkflowDraft, WorkflowSnapshot
from litellm.models.workflow_execution import (
    ChildCallResult,
    EngineRun,
    ExecutionError,
    RunAdmission,
    RunInput,
    RunPage,
    RunState,
    StepTrace,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.workflow_execution_endpoints import (
    TestRunRequest as PaidRunRequest,
)
from litellm.proxy.management_endpoints.workflow_execution_endpoints import (
    cancel_workflow_run,
    get_run_definition,
    get_workflow_steps,
    list_definition_runs,
)
from litellm.proxy.management_endpoints.workflow_execution_endpoints import (
    test_workflow as run_draft_test,
)
from litellm.proxy.workflows.runner import WorkflowRunner
from litellm.repositories.workflow_definition_repository import DefinitionDatabase, WorkflowDefinitionRepository
from litellm.repositories.workflow_execution_repository import ExecutionDatabase, WorkflowExecutionRepository
from tests.integration._support.database import scratch_database
from tests.unit.proxy.workflows.test_runner import two_nodes

pytestmark: Final = pytest.mark.asyncio(loop_scope="function")
_ROOT: Final = Path(__file__).resolve().parents[3]
_MESSAGES: Final = ({"role": "user", "content": "plan"},)
_CALLER: Final = UserAPIKeyAuth(token="owner", team_id="team").model_copy(update={"via_virtual_key": True})
_REQUEST: Final = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})


@dataclass(frozen=True, slots=True)
class Storage:
    db: Prisma
    runs: WorkflowExecutionRepository
    definitions: WorkflowDefinitionRepository

    async def admit(self, *, key: str | None = None, digest: str = "input", owner: str = "owner",
                    team: str = "team", snapshot: WorkflowSnapshot | None = None,
                    deadline: datetime | None = None) -> RunAdmission | ExecutionError:
        return await self.runs.admit(
            workflow_id=snapshot.workflow_id if snapshot else "workflow",
            snapshot_id=snapshot.snapshot_id if snapshot else "snapshot",
            team_id=team, key_hash=owner, input=RunInput(messages=_MESSAGES, draft=False,
            deadline=deadline or datetime.now(timezone.utc) + timedelta(minutes=5)), digest=digest, idempotency_key=key,
        )

    async def snapshot(self, team: str = "team") -> WorkflowSnapshot:
        draft: Final = await self.definitions.create(two_nodes(), team, "author")
        assert isinstance(draft, WorkflowDraft)
        snapshot: Final = await self.definitions.publish(draft.workflow_id, team, draft.etag, draft.digest, "admin")
        assert isinstance(snapshot, WorkflowSnapshot)
        return snapshot


@pytest.fixture(scope="module")
def schema_sql() -> bytes:
    return subprocess.run(
        [sys.executable, "-I", "-m", "prisma", "migrate", "diff", "--from-empty", "--to-schema-datamodel",
         str(_ROOT / "schema.prisma"), "--script"], check=True, capture_output=True, cwd=_ROOT,
    ).stdout


@pytest_asyncio.fixture(loop_scope="function")
async def store(schema_sql: bytes) -> AsyncIterator[Storage]:
    with scratch_database("workflow_execution") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute(schema_sql)
        connection.execute(sql.SQL("ALTER DATABASE {} SET timezone TO 'Asia/Singapore'").format(
            sql.Identifier(connection.info.dbname)
        ))
        async with Prisma(datasource=DatasourceOverride(url=url)) as client:
            assert await client.query_raw("SHOW TIME ZONE") == [{"TimeZone": "Asia/Singapore"}]
            for team in ("team", "other"):
                await client.litellm_teamtable.create(data={"team_id": team, "members_with_roles": Json([
                    {"user_id": "admin", "role": "admin"}, {"user_id": "member", "role": "user"},
                ])})
            yield Storage(client, WorkflowExecutionRepository(cast(ExecutionDatabase, client)),
                          WorkflowDefinitionRepository(cast(DefinitionDatabase, client)))


class ChatReplies:
    def __init__(self, store: Storage, replies: tuple[ChildCallResult | ExecutionError | Exception, ...],
                 cancel_after: int | None = None) -> None:
        self.store, self.replies, self.cancel_after = store, replies, cancel_after
        self.calls: tuple[StepTrace, ...] = ()
        self.models: tuple[str, ...] = ()
        self.persisted: tuple[StepTrace, ...] = ()

    async def __call__(self, request: Request, run: EngineRun, node: ChatCompletionNode,
                       step: StepTrace) -> ChildCallResult | ExecutionError:
        self.calls = (*self.calls, step)
        self.models = (*self.models, node.model)
        self.persisted = (*self.persisted, (await self.store.runs.steps(run.run_id))[-1])
        if len(self.calls) == self.cancel_after:
            await self.store.runs.cancel(run.run_id)
        reply: Final = self.replies[len(self.calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


async def test_concurrent_idempotency_pins_first_snapshot_and_conflicts_on_input(store: Storage) -> None:
    original: Final = await store.snapshot()
    different: Final = original.model_copy(update={"snapshot_id": "later-snapshot"})
    results: Final = await asyncio.gather(*(
        store.admit(key="same", snapshot=snapshot) for snapshot in (original, different) * 4
    ))
    assert all(isinstance(result, RunAdmission) for result in results)
    admitted: Final = tuple(result for result in results if isinstance(result, RunAdmission))
    assert sum(result.created for result in admitted) == 1
    assert len({result.run.run_id for result in admitted}) == 1
    assert len({result.run.snapshot_id for result in admitted}) == 1
    assert all(result.run == admitted[0].run for result in admitted)
    assert await store.admit(key="same", snapshot=different, digest="changed") == ExecutionError(
        "idempotency_conflict", "Idempotency key was used with different input"
    )
    assert isinstance(await store.admit(key="same", snapshot=original, owner="another"), RunAdmission)
    assert isinstance(await store.admit(key="same", snapshot=original, team="other"), RunAdmission)
    assert await store.db.litellm_workflowrun.count() == 3


async def test_draft_test_retry_after_edit_recovers_only_the_same_scoped_request(store: Storage) -> None:
    draft: Final = await store.definitions.create(two_nodes(), "team", "member")
    assert isinstance(draft, WorkflowDraft)
    caller: Final = _CALLER.model_copy(update={"user_id": "member", "user_role": LitellmUserRoles.INTERNAL_USER})
    data: Final = PaidRunRequest(messages=_MESSAGES, authorize_billing=True)
    chat: Final = ChatReplies(store, (
        ChildCallResult(list(_MESSAGES), 1, 1, "first", 0.125), ChildCallResult("done", 1, 1, "last", 0.125),
    ))
    runner: Final = WorkflowRunner(store.runs, store.definitions, chat)
    request: Final = Request({**_REQUEST.scope, "state": {"workflow_runner": runner}})
    try:
        accepted: Final = await run_draft_test(
            request, draft.workflow_id, data, caller, store.definitions, store.runs,
            if_match=draft.etag, idempotency_key="retry",
        )
    finally:
        await runner.close()
    changed: Final = await store.definitions.replace(
        draft.workflow_id, "team", draft.etag, two_nodes(), "member",
    )
    assert isinstance(changed, WorkflowDraft) and changed.etag != draft.etag
    replay: Final = await run_draft_test(
        _REQUEST, draft.workflow_id, data, caller, store.definitions, store.runs,
        if_match=draft.etag, idempotency_key="retry",
    )
    assert replay.status_code == accepted.status_code == 202
    assert replay.body == accepted.body and replay.headers == accepted.headers
    for body, identity, etag, key, status in (
        (data.model_copy(update={"messages": ({"role": "user", "content": "changed"},)}), caller, draft.etag, "retry", 409),
        (data, caller, changed.etag, "retry", 409),
        (data, caller, None, "retry", 412),
        (data, caller, draft.etag, "new-key", 412),
        (data, caller.model_copy(update={"token": "peer"}), draft.etag, "retry", 412),
        (data, caller.model_copy(update={"team_id": "other"}), draft.etag, "retry", 404),
        (data, caller.model_copy(update={"user_id": "removed-member"}), draft.etag, "retry", 404),
        (data, caller.model_copy(update={"via_virtual_key": False}), draft.etag, "retry", 403),
        (data.model_copy(update={"authorize_billing": False}), caller, draft.etag, "retry", 400),
    ):
        with pytest.raises(HTTPException) as rejected:
            await run_draft_test(
                _REQUEST, draft.workflow_id, body, identity, store.definitions, store.runs,
                if_match=etag, idempotency_key=key,
            )
        assert rejected.value.status_code == status
    assert await store.db.litellm_workflowrun.count() == await store.db.litellm_workflowsnapshot.count() == 1
    assert len(chat.calls) == 2


async def test_nullable_history_filters_cursor_ties_and_utc_in_non_utc_database(store: Storage) -> None:
    before: Final = datetime.now(timezone.utc).replace(microsecond=0)
    admissions: Final = tuple([await store.admit(owner=owner) for owner in ("owner", "foreign", "owner")])
    assert all(isinstance(item, RunAdmission) for item in admissions)
    runs: Final = tuple(item.run for item in admissions if isinstance(item, RunAdmission))
    await store.db.litellm_workflowrun.update_many(where={}, data={"created_at": runs[0].created_at})
    ordered: Final = tuple(sorted((run.run_id for run in runs), reverse=True))
    for owner, state in ((None, None), ("owner", None), (None, RunState.pending), ("owner", RunState.pending)):
        page: Final = await store.runs.list_runs("workflow", "team", owner, state, 1, None)
        assert isinstance(page, RunPage) and page.next_cursor is not None
        tail: Final = await store.runs.list_runs("workflow", "team", owner, state, 10, page.next_cursor)
        assert isinstance(tail, RunPage) and tail.next_cursor is None
        expected: Final = tuple(run_id for run_id in ordered if owner is None or run_id != runs[1].run_id)
        assert tuple(run.run_id for run in (*page.runs, *tail.runs)) == expected
    await store.runs.finish(runs[0].run_id, RunState.failed, None, "failed")
    finished: Final = await store.runs.get(runs[0].run_id)
    assert finished is not None and finished.output is None and finished.metadata == {"error": "failed"}
    filtered: Final = await store.runs.list_runs("workflow", "team", "owner", RunState.failed, 10, None)
    assert isinstance(filtered, RunPage) and tuple(run.run_id for run in filtered.runs) == (finished.run_id,)
    for timestamp in (runs[0].created_at, finished.updated_at):
        assert before <= timestamp <= datetime.now(timezone.utc)
    assert isinstance(await store.runs.list_runs("workflow", "team", None, None, 10, "bad-cursor"), ExecutionError)


async def test_attempt_receipts_deduplicate_checkpoints_and_exclude_foreign_spend(store: Storage) -> None:
    admitted: Final = await store.admit()
    assert isinstance(admitted, RunAdmission)
    run: Final = admitted.run
    first: Final = StepTrace(step_id="step", run_id=run.run_id, node_id="node", state=RunState.failed, input=None,
                            attempts=1, started_at=run.created_at, deadline=run.input.deadline, call_id="first",
                            prompt_tokens=4, completion_tokens=2, response_cost=0.125, request_id="failed")
    second: Final = first.model_copy(update={
        "attempts": 2, "state": RunState.completed, "call_id": "second", "output": False,
    })
    for step in (first, first, second, second):
        assert await store.runs.checkpoint(run.run_id, step)
    for request_id, key, team, call_id, session in (
        ("a", "owner", "team", "second", run.session_id),
        ("b", "owner", "team", "second", "parent-session"),
        ("wrong-key", "foreign", "team", "second", run.session_id),
        ("wrong-team", "owner", "other", "second", run.session_id),
        ("wrong-call", "owner", "team", "unrelated", run.session_id),
    ):
        await store.db.litellm_spendlogs.create(data={
            "request_id": request_id, "call_type": "acompletion", "api_key": key, "team_id": team,
            "session_id": session, "litellm_call_id": call_id, "startTime": run.created_at, "endTime": run.created_at,
            "prompt_tokens": 3, "completion_tokens": 1, "spend": 0.25,
        })
    assert await store.runs.steps(run.run_id) == (second.model_copy(update={
        "prompt_tokens": 10, "completion_tokens": 4, "response_cost": 0.625,
        "request_id": None, "request_ids": ("failed", "a", "b"),
    }),)
    events: Final = await store.db.litellm_workflowevent.find_many(where={"run_id": run.run_id})
    assert len(events) == 4
    assert all(run.created_at <= event.created_at <= datetime.now(timezone.utc) for event in events)


@pytest.mark.parametrize("phase", ["before", "retry", "downstream"])
async def test_cancellation_fences_every_dispatch_but_keeps_completed_receipts(store: Storage, phase: str) -> None:
    snapshot: Final = await store.snapshot()
    admitted: Final = await store.admit(snapshot=snapshot)
    assert isinstance(admitted, RunAdmission)
    run: Final = admitted.run
    receipt: Final = ChildCallResult(list(_MESSAGES), 4, 2, "receipt", 0.125)
    reply: Final = ExecutionError("admission_rate_limit", "retry") if phase == "retry" else receipt
    chat: Final = ChatReplies(store, (reply,), cancel_after=1)
    if phase == "before":
        await store.runs.cancel(run.run_id)
    runner: Final = WorkflowRunner(store.runs, store.definitions, chat)
    try:
        result: Final = await runner._sequence(run, _REQUEST, snapshot.bundle, ())
        assert isinstance(result, ExecutionError) and result.code == "cancelled"
        assert len(chat.calls) == (0 if phase == "before" else 1)
        traces: Final = await store.runs.steps(run.run_id)
        expected_ids: Final = () if phase == "before" else ((),) if phase == "retry" else (("receipt",),)
        assert tuple(trace.request_ids for trace in traces) == expected_ids
        await store.runs.finish(run.run_id, RunState.completed, {"must": "not win cancellation"}, None)
        settled: Final = await store.runs.get(run.run_id)
        assert settled is not None and settled.status is RunState.cancelled and settled.output is None
    finally:
        await runner.close()


@pytest.mark.parametrize("error", [
    ExecutionError("invalid_input", "rejected"), ExecutionError("provider_error", "unavailable"),
    ExecutionError("rate_limit", "upstream 429"), ExecutionError("runtime_error", "runtime failed"),
    ExecutionError("admission_rate_limit", "retry budget exhausted"),
    ExecutionError("provider_error", "billed failure", ChildCallResult(None, 4, 2, "billed", 0.125)),
    ExecutionError("admission_rate_limit", "known receipt", ChildCallResult(None, 4, 2, "billed", 0.125)),
    RuntimeError("private details"),
])
async def test_failure_stops_downstream_and_persists_sanitized_terminal_trace(
    store: Storage, error: ExecutionError | Exception,
) -> None:
    snapshot: Final = await store.snapshot()
    admitted: Final = await store.admit(snapshot=snapshot)
    assert isinstance(admitted, RunAdmission)
    chat: Final = ChatReplies(store, (error, error))
    runner: Final = WorkflowRunner(store.runs, store.definitions, chat)
    try:
        result: Final = await runner._sequence(admitted.run, _REQUEST, snapshot.bundle, ())
        expected: Final = error if isinstance(error, ExecutionError) else ExecutionError(
            "execution_failed", "The node could not complete"
        )
        assert result == expected
        attempts: Final = 2 if expected.code == "admission_rate_limit" and expected.receipt is None else 1
        traces: Final = await store.runs.steps(admitted.run.run_id)
        assert tuple((step.node_id, step.state, step.attempts, step.error) for step in traces) == (
            ("node_1", RunState.failed, attempts, expected.message),
        )
        assert len(chat.calls) == attempts
        assert (traces[0].prompt_tokens, traces[0].completion_tokens, traces[0].response_cost, traces[0].request_ids) == (
            (4, 2, 0.125, ("billed",)) if expected.receipt is not None else (0, 0, None, ())
        )
    finally:
        await runner.close()


@pytest.mark.parametrize("trigger", ["get", "admit", "list"])
async def test_deadline_expiry_is_terminal_and_fences_dispatch(store: Storage, trigger: str) -> None:
    admitted: Final = await store.admit(deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert isinstance(admitted, RunAdmission)
    if trigger == "admit":
        await store.admit()
    elif trigger == "list":
        await store.runs.list_runs("workflow", "team", None, None, 10, None)
    else:
        await store.runs.get(admitted.run.run_id)
    stored: Final = await store.db.litellm_workflowrun.find_unique(where={"run_id": admitted.run.run_id})
    assert stored is not None and stored.status == "failed"
    step: Final = StepTrace(
        step_id="step", run_id=stored.run_id, node_id="node", state=RunState.running,
        input=None, attempts=1, started_at=admitted.run.created_at, deadline=admitted.run.input.deadline,
    )
    assert not await store.runs.checkpoint(stored.run_id, step)
    await store.runs.finish(stored.run_id, RunState.completed, "late output", None)
    expired: Final = await store.runs.get(stored.run_id)
    assert expired is not None and expired.status is RunState.failed and expired.output is None
    assert await store.runs.steps(stored.run_id) == ()


@pytest.mark.parametrize("role,user", [
    (LitellmUserRoles.PROXY_ADMIN, "admin"), (LitellmUserRoles.INTERNAL_USER, "admin"),
    (LitellmUserRoles.INTERNAL_USER, "member"), (None, None),
])
async def test_key_scope_intersects_user_scope_and_own_application_key_can_cancel(
    store: Storage, role: LitellmUserRoles | None, user: str | None,
) -> None:
    local, foreign = await store.snapshot(), await store.snapshot("other")
    own: Final = await store.admit(snapshot=local)
    peer: Final = await store.admit(snapshot=local, owner="peer")
    other: Final = await store.admit(snapshot=foreign, team="other")
    assert isinstance(own, RunAdmission) and isinstance(peer, RunAdmission) and isinstance(other, RunAdmission)
    caller: Final = _CALLER.model_copy(update={"user_role": role, "user_id": user})
    for endpoint in (get_workflow_steps, get_run_definition, cancel_workflow_run):
        with pytest.raises(HTTPException) as denied:
            await endpoint(other.run.run_id, caller, store.definitions, store.runs)
        assert denied.value.status_code == 404
    with pytest.raises(HTTPException) as denied_history:
        await list_definition_runs(
            foreign.workflow_id, caller, store.definitions, store.runs, team_id="other", limit=10, after=None,
        )
    assert denied_history.value.status_code == 404
    history: Final = await list_definition_runs(
        local.workflow_id, caller, store.definitions, store.runs, limit=10, after=None,
    )
    expected: Final = {own.run.run_id, peer.run.run_id} if user == "admin" else {own.run.run_id}
    assert {run.run_id for run in history.runs} == expected
    for endpoint in (get_workflow_steps, get_run_definition, cancel_workflow_run):
        if user == "admin":
            assert await endpoint(peer.run.run_id, caller, store.definitions, store.runs) is not None
        else:
            with pytest.raises(HTTPException) as denied_peer:
                await endpoint(peer.run.run_id, caller, store.definitions, store.runs)
            assert denied_peer.value.status_code == 404
    pinned: Final = await get_run_definition(own.run.run_id, caller, store.definitions, store.runs)
    assert pinned.snapshot_id == local.snapshot_id
    assert (await cancel_workflow_run(own.run.run_id, caller, store.definitions, store.runs)).status_code == 202
    assert (await get_workflow_steps(own.run.run_id, caller, store.definitions, store.runs)).cancel_requested
    foreign_trace: Final = await get_workflow_steps(
        other.run.run_id, _CALLER.model_copy(update={"team_id": "other"}), store.definitions, store.runs,
    )
    assert not foreign_trace.cancel_requested


async def test_submit_drains_pinned_snapshot_serializes_chat_and_never_cold_replays(store: Storage) -> None:
    snapshot: Final = await store.snapshot()
    admitted: Final = await store.admit(snapshot=snapshot, key="original")
    orphan: Final = await store.admit(snapshot=snapshot)
    assert isinstance(admitted, RunAdmission) and isinstance(orphan, RunAdmission)
    draft: Final = await store.definitions.get_draft(snapshot.workflow_id, "team")
    assert draft is not None
    changed: Final = snapshot.bundle.model_copy(update={
        "nodes": tuple(node.model_copy(update={"model": "edited"}) for node in snapshot.bundle.nodes),
    })
    replaced: Final = await store.definitions.replace(snapshot.workflow_id, "team", draft.draft.etag, changed, "author")
    assert isinstance(replaced, WorkflowDraft)
    republished: Final = await store.definitions.publish(
        snapshot.workflow_id, "team", replaced.etag, replaced.digest, "admin",
    )
    assert isinstance(republished, WorkflowSnapshot)
    chat: Final = ChatReplies(store, (
        ExecutionError("admission_rate_limit", "retry"),
        ChildCallResult(list(_MESSAGES), 4, 2, "first", 0.125),
        ChildCallResult({"answer": False}, 6, 3, "last", 0.25),
    ))
    runner: Final = WorkflowRunner(store.runs, store.definitions, chat)
    try:
        assert await runner.submit(admitted.run, _REQUEST, _CALLER) is None
    finally:
        await runner.close()
    completed: Final = await store.runs.get(admitted.run.run_id)
    assert completed is not None and completed.status is RunState.completed, completed
    completion: Final = ChatCompletion.model_validate(completed.output)
    assert json.loads(completion.choices[0].message.content or "null") == {"answer": False}
    assert completion.usage is not None
    usage: Final = completion.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 5, 15)
    assert chat.models == ("gateway-model",) * 3
    assert tuple((step.node_id, step.attempts) for step in chat.calls) == (("node_1", 1), ("node_1", 2), ("node_2", 1))
    assert chat.calls[0].step_id == chat.calls[1].step_id and chat.calls[0].deadline == chat.calls[1].deadline
    assert len({step.call_id for step in chat.calls}) == 3
    assert tuple((step.state, step.call_id) for step in chat.persisted) == tuple(
        (RunState.running, step.call_id) for step in chat.calls
    )
    assert chat.calls[1].input == chat.calls[0].input == chat.calls[2].input == list(_MESSAGES)
    pinned: Final = await get_run_definition(completed.run_id, _CALLER, store.definitions, store.runs)
    assert pinned.bundle == snapshot.bundle
    restarted: Final = WorkflowRunner(store.runs, store.definitions, chat)
    await restarted.close()
    pending: Final = await store.runs.get(orphan.run.run_id)
    assert pending is not None and pending.status is RunState.pending and len(chat.calls) == 3
    assert await store.runs.steps(orphan.run.run_id) == ()
