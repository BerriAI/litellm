from dataclasses import dataclass
from typing import Final

import pytest
from prisma.errors import PrismaError
from typing_extensions import LiteralString

from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.runtime import TelemetryRuntime, deployment_hash_secret
from litellm.proxy.telemetry.settings import TelemetrySettings
from litellm.proxy.telemetry.store import HASH_SECRET_PARAM, LocalTableExporter, TelemetryStore
from litellm.telemetry.records import AttemptRecord, InstanceInfo, StatusClass
from litellm.telemetry.report import Report
from litellm.telemetry.sink import ExportOutcome


@dataclass
class _FakeDatabase:
    instance_id: str = "persisted-install"
    hash_secret: str = "persisted-secret"
    fail_reads: bool = False
    fail_writes: bool = False
    fail_prune: bool = False
    executed: tuple[tuple[str, tuple[object, ...]], ...] = ()

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        if self.fail_reads:
            raise PrismaError("database is down")
        return ({"value": self.instance_id if args[0] == "telemetry_instance_id" else self.hash_secret},)

    failed_ids: tuple[object, ...] = ()

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        if self.fail_writes and query.lstrip().startswith("INSERT"):
            self.failed_ids = (*self.failed_ids, args[0])
        if self.fail_writes or (self.fail_prune and query.lstrip().startswith("DELETE")):
            raise PrismaError("database is down")
        self.executed = (*self.executed, (query, args))
        return 1


def _report(*, dropped_records: int = 3) -> Report:
    return Report(
        instance=InstanceInfo(instance_id="i", litellm_version="1.0.0"),
        window_start=10.0,
        window_end=70.0,
        dropped_records=dropped_records,
    )


@pytest.mark.asyncio
async def test_a_stored_report_keeps_its_window_and_prunes_past_the_retention() -> None:
    db: Final = _FakeDatabase()
    assert await LocalTableExporter(TelemetryStore(db, retention_days=7)).export(_report()) is ExportOutcome.SENT
    (insert, insert_args), (prune, prune_args) = db.executed
    assert 'INSERT INTO "LiteLLM_TelemetryReport"' in insert
    assert insert_args[1:3] == (10.0, 70.0)
    assert '"instance_id": "i"' in str(insert_args[3])
    assert 'DELETE FROM "LiteLLM_TelemetryReport"' in prune
    assert prune_args == (7,)


@pytest.mark.asyncio
async def test_an_idle_window_is_not_stored() -> None:
    db: Final = _FakeDatabase()
    exporter: Final = LocalTableExporter(TelemetryStore(db, retention_days=7))
    assert await exporter.export(_report(dropped_records=0)) is ExportOutcome.SENT
    assert db.executed == ()


@pytest.mark.asyncio
async def test_a_database_failure_keeps_the_window_for_the_next_flush() -> None:
    exporter: Final = LocalTableExporter(TelemetryStore(_FakeDatabase(fail_writes=True), retention_days=7))
    assert await exporter.export(_report()) is ExportOutcome.RETRY


@pytest.mark.asyncio
async def test_retrying_a_window_whose_insert_may_have_committed_overwrites_it_instead_of_adding_a_copy() -> None:
    db: Final = _FakeDatabase(fail_writes=True)
    exporter: Final = LocalTableExporter(TelemetryStore(db, retention_days=7))
    assert await exporter.export(_report()) is ExportOutcome.RETRY
    db.fail_writes = False
    assert await exporter.export(_report()) is ExportOutcome.SENT
    assert await exporter.export(_report(dropped_records=5)) is ExportOutcome.SENT
    insert_ids: Final = [args[0] for query, args in db.executed if 'LiteLLM_TelemetryReport" (id' in query]
    retried, next_window = insert_ids
    assert retried == db.failed_ids[0] and next_window != retried
    assert "ON CONFLICT (id) DO UPDATE" in db.executed[0][0]


@pytest.mark.asyncio
async def test_a_failed_prune_does_not_resend_a_window_that_was_already_stored() -> None:
    db: Final = _FakeDatabase(fail_prune=True)
    assert await LocalTableExporter(TelemetryStore(db, retention_days=7)).export(_report()) is ExportOutcome.SENT
    assert len(db.executed) == 1


@pytest.mark.asyncio
async def test_without_an_endpoint_the_runtime_keeps_reports_locally_under_the_persisted_instance_id() -> None:
    db: Final = _FakeDatabase()
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="heartbeat,request_success,token_info,request_taxonomy"),
        db=lambda: db,
        register=registered.append,
    )
    assert runtime.store is not None
    assert runtime.sink is not None
    assert len(registered) == 1
    runtime.sink.record_attempt(AttemptRecord(provider="anthropic", provider_status=StatusClass.SUCCESS, stream=False))
    await runtime.stop()
    stored: Final = tuple(args for query, args in db.executed if 'INSERT INTO "LiteLLM_TelemetryReport"' in query)
    assert len(stored) == 1
    assert '"instance_id": "persisted-install"' in str(stored[0][3])


@pytest.mark.asyncio
async def test_deployments_are_hashed_with_the_secret_stored_in_the_database_not_the_salt_key() -> None:
    db: Final = _FakeDatabase()
    secret: Final = await deployment_hash_secret(TelemetryStore(db, retention_days=7), lambda: b"salt-key")
    ((_, (param, candidate)),) = db.executed
    assert secret == b"persisted-secret"
    assert param == HASH_SECRET_PARAM
    assert len(str(candidate)) == 64, "a fresh install stores a 32 byte random secret"


@pytest.mark.parametrize(
    "store",
    [None, TelemetryStore(_FakeDatabase(fail_reads=True), retention_days=7)],
    ids=["no_database", "unreadable_database"],
)
@pytest.mark.asyncio
async def test_without_a_readable_stored_secret_deployments_are_hashed_with_the_salt_key(
    store: TelemetryStore | None,
) -> None:
    assert await deployment_hash_secret(store, lambda: b"salt-key") == b"salt-key"
