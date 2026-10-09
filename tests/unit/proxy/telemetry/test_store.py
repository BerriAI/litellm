from dataclasses import dataclass
from typing import Final

from typing_extensions import LiteralString

import pytest
from prisma.errors import PrismaError

from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.runtime import TelemetryRuntime
from litellm.proxy.telemetry.settings import TelemetrySettings
from litellm.proxy.telemetry.store import LocalTableExporter, TelemetryStore
from litellm.telemetry.records import AttemptRecord, InstanceInfo, StatusClass
from litellm.telemetry.report import Report
from litellm.telemetry.sink import ExportOutcome


@dataclass
class _FakeDatabase:
    instance_id: str = "persisted-install"
    fail_writes: bool = False
    fail_prune: bool = False
    executed: tuple[tuple[str, tuple[object, ...]], ...] = ()

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        return ({"instance_id": self.instance_id},)

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
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
