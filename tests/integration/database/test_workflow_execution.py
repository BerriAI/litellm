import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, cast

import psycopg
import pytest
import pytest_asyncio
from prisma import Prisma
from prisma.types import DatasourceOverride
from psycopg import sql
from pydantic import TypeAdapter

from litellm.models.workflow_execution import RunAdmission, RunInput, RunPage, RunState, StepTrace
from litellm.repositories.workflow_execution_repository import ExecutionDatabase, WorkflowExecutionRepository
from tests.integration._support.database import scratch_database

_ROOT: Final = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def schema_sql() -> bytes:
    return subprocess.run(
        [sys.executable, "-I", "-m", "prisma", "migrate", "diff", "--from-empty", "--to-schema-datamodel",
         str(_ROOT / "schema.prisma"), "--script"],
        check=True, capture_output=True, cwd=_ROOT,
    ).stdout


@pytest.fixture
def database(schema_sql: bytes) -> Iterator[str]:
    with scratch_database("workflow_execution") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute(schema_sql)
        connection.execute(
            sql.SQL("ALTER DATABASE {} SET timezone TO 'Asia/Singapore'").format(sql.Identifier(connection.info.dbname))
        )
        yield url


@pytest_asyncio.fixture(loop_scope="function")
async def ledger(database: str) -> AsyncIterator[Prisma]:
    client: Final = Prisma(datasource=DatasourceOverride(url=database))
    await client.connect()
    try:
        assert TypeAdapter(tuple[dict[str, str], ...]).validate_python(await client.query_raw("SHOW TIME ZONE")) == (
            {"TimeZone": "Asia/Singapore"},
        )
        yield client
    finally:
        await client.disconnect()


def _repository(ledger: Prisma) -> WorkflowExecutionRepository:
    return WorkflowExecutionRepository(
        cast(ExecutionDatabase, ledger)  # cast-ok: generated Prisma transaction delegates satisfy the protocol
    )


def _input(now: datetime) -> RunInput:
    return RunInput(messages=({"role": "user", "content": "plan"},), deadline=now + timedelta(minutes=5), draft=True)


async def _admit(
    repository: WorkflowExecutionRepository, data: RunInput, key: str | None,
    *, snapshot: str = "snapshot", owner: str = "owner",
) -> RunAdmission:
    result: Final = await repository.admit(
        workflow_id="workflow", snapshot_id=snapshot, team_id="team", key_hash=owner,
        input=data, digest="same-input", idempotency_key=key,
    )
    assert isinstance(result, RunAdmission)
    return result


@pytest.mark.asyncio(loop_scope="function")
async def test_concurrent_admission_pins_snapshot_and_execution_timestamps_are_utc(ledger: Prisma) -> None:
    repository: Final = _repository(ledger)
    before: Final = datetime.now(timezone.utc).replace(microsecond=0)
    data: Final = _input(before)
    first, second = await asyncio.gather(
        _admit(repository, data, "idempotency-key", snapshot="first"),
        _admit(repository, data, "idempotency-key", snapshot="edited"),
    )
    assert sorted((first.created, second.created)) == [False, True]
    assert first.run == second.run
    assert await repository.find_idempotent("workflow", "team", "owner", "idempotency-key") == first.run
    assert await ledger.litellm_workflowrun.count() == 1
    run_id: Final = first.run.run_id
    claimed: Final = await repository.claim(run_id, "worker", data.deadline)
    assert claimed is not None and claimed.status is RunState.RUNNING and claimed.lease_expires_at == data.deadline
    assert await repository.claim(run_id, "other-worker", data.deadline) is None
    extended: Final = data.deadline + timedelta(minutes=1)
    assert await repository.renew(run_id, "worker", extended) is True
    renewed: Final = await repository.get(run_id)
    assert renewed is not None and renewed.lease_expires_at == extended
    step: Final = StepTrace(
        step_id="step", run_id=run_id, node_id="node", state=RunState.COMPLETED, input={"n": 1}, output={"n": 2},
        attempts=1, deadline=data.deadline, started_at=before, finished_at=before,
    )
    assert await repository.checkpoint(run_id, "other-worker", step) is False
    assert await repository.checkpoint(run_id, "worker", step) is True
    assert await repository.steps(run_id) == (step,)
    assert await repository.finish(run_id, "worker", RunState.COMPLETED, {"n": 2}, None) is True
    finished: Final = await repository.get(run_id)
    assert finished is not None and finished.status is RunState.COMPLETED
    assert (finished.output, finished.metadata, finished.lease_token, finished.lease_expires_at) == (
        {"n": 2}, {"error": None}, None, None,
    )
    stored: Final = await ledger.litellm_workflowrun.find_unique(where={"run_id": run_id})
    events: Final = await ledger.litellm_workflowevent.find_many(where={"run_id": run_id})
    assert stored is not None and stored.created_at == finished.created_at and stored.updated_at == finished.updated_at
    assert len(events) == 1
    for timestamp in (first.run.created_at, claimed.updated_at, finished.updated_at, events[0].created_at):
        assert before <= timestamp <= datetime.now(timezone.utc)


@pytest.mark.asyncio(loop_scope="function")
async def test_nullable_admission_and_history_filters_round_trip_through_prisma(ledger: Prisma) -> None:
    repository: Final = _repository(ledger)
    data: Final = _input(datetime.now(timezone.utc).replace(microsecond=0))
    first: Final = await _admit(repository, data, None)
    foreign: Final = await _admit(repository, data, None, owner="other")
    second: Final = await _admit(repository, data, None)
    assert len({first.run.run_id, foreign.run.run_id, second.run.run_id}) == 3
    await ledger.litellm_workflowrun.update_many(where={}, data={"created_at": first.run.created_at})
    ordered: Final = tuple(sorted((first.run.run_id, foreign.run.run_id, second.run.run_id), reverse=True))
    owned: Final = tuple(run_id for run_id in ordered if run_id != foreign.run.run_id)
    page: Final = await repository.list_runs("workflow", "team", None, None, 2, None)
    assert isinstance(page, RunPage) and page.count == 2 and page.next_cursor is not None
    assert tuple(run.run_id for run in page.runs) == ordered[:2]
    tail: Final = await repository.list_runs("workflow", "team", None, None, 2, page.next_cursor)
    assert isinstance(tail, RunPage) and tail.count == 1 and tail.next_cursor is None
    assert tuple(run.run_id for run in tail.runs) == ordered[2:]
    scoped: Final = await repository.list_runs("workflow", "team", "owner", RunState.PENDING, 1, None)
    assert isinstance(scoped, RunPage) and scoped.next_cursor is not None
    assert tuple(run.run_id for run in scoped.runs) == owned[:1]
    continuation: Final = await repository.list_runs("workflow", "team", "owner", RunState.PENDING, 1, scoped.next_cursor)
    assert isinstance(continuation, RunPage) and continuation.count == 1 and continuation.next_cursor is None
    assert tuple(run.run_id for run in continuation.runs) == owned[1:]
    assert await repository.claim(second.run.run_id, "worker", data.deadline) is not None
    assert await repository.finish(second.run.run_id, "worker", RunState.FAILED, None, "execution_failed") is True
    failed: Final = await repository.get(second.run.run_id)
    assert failed is not None and failed.output is None and failed.metadata == {"error": "execution_failed"}
    filtered: Final = await repository.list_runs("workflow", "team", "owner", RunState.FAILED, 2, None)
    assert isinstance(filtered, RunPage) and filtered.count == 1 and filtered.next_cursor is None
    assert tuple(run.run_id for run in filtered.runs) == (second.run.run_id,)
