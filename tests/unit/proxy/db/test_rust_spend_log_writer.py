import asyncio
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.proxy.db.rust_spend_log_writer import (
    _request_flush_from_any_thread,
    flush_in_rust,
    install_rust_spend_log_writer,
    rust_spend_log_writer,
    uninstall_rust_spend_log_writer,
)
from litellm.proxy.utils import (
    PrismaClient,
    _park_remaining_spend_logs,
    _total_queued_spend_transactions,
    accounting_only_spend_log,
    dequeue_spend_logs,
    enqueue_spend_logs,
    update_spend,
)
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase, NativeDatabaseFailure, RustSpendLogWriter

_UNREACHABLE: Final = "postgresql://litellm:secret@db.invalid:5432/proxy?pool_timeout=1"
_REFUSED: Final = "postgresql://litellm:secret@127.0.0.1:1/proxy?pool_timeout=1"
_LEDGER_ON: Final[Mapping[str, object]] = {"use_spend_logs_ledger": True}


class FakePrismaClient:
    def __init__(self) -> None:
        self.spend_log_transactions: list[Mapping[str, object]] = []
        self.tool_usage_transactions: list[object] = []
        self.model_usage_transactions: list[object] = []
        self.autorouter_turn_transactions: list[object] = []
        self.baseline_accounting_transactions: list[object] = []
        self._spend_log_transactions_lock = asyncio.Lock()
        self._tool_usage_transactions_lock = asyncio.Lock()
        self._model_usage_transactions_lock = asyncio.Lock()
        self._autorouter_turn_transactions_lock = asyncio.Lock()
        self.baseline_accounting_lock = asyncio.Lock()
        self.spend_log_write_lock = asyncio.Lock()
        self.spend_log_flush_requested: asyncio.Event | None = None

    def jsonify_object(self, obj: Mapping[str, object]) -> Mapping[str, object]:
        return obj


class FailingWriter:
    def __init__(self, rows: Sequence[str]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    async def flush(self, limit: int, retention: object) -> list[dict[str, object]]:
        raise NativeDatabaseFailure("connection refused")

    def dequeue(self, limit: int) -> list[str]:
        taken: Final = self.rows[:limit]
        self.rows[:] = self.rows[limit:]
        return taken

    async def wait_idle(self) -> None:
        return None


class WritingWriter:
    def __init__(self, written: list[dict[str, object]]) -> None:
        self.written = written

    def __len__(self) -> int:
        return len(self.written)

    async def flush(self, limit: int, retention: object) -> list[dict[str, object]]:
        return self.written

    async def wait_idle(self) -> None:
        return None


async def _unreachable_database(url: str = _UNREACHABLE) -> NativeDatabase:
    return await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(url, 1),
        None,
    )


def _proxy_logging(*, parked: bool = False) -> MagicMock:
    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock()
    buffer: Final = proxy_logging_obj.db_spend_update_writer.redis_update_buffer
    buffer.redis_cache = MagicMock() if parked else None
    buffer.get_spend_logs_from_redis_buffer = AsyncMock(return_value=[])
    buffer.store_spend_logs_in_redis = AsyncMock(return_value=parked)
    return proxy_logging_obj


def _row(request_id: str, **fields: object) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "acompletion",
        "api_key": "hashed",
        "spend": 1.5,
        "total_tokens": 3,
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "startTime": "2026-11-02T12:00:00+00:00",
        "endTime": "2026-11-02T12:00:01+00:00",
        "completionStartTime": None,
        "model": "gpt",
        "user": "user",
        "team_id": "team",
        "organization_id": "",
        "metadata": '{"status": "success"}',
        "cache_key": "Cache OFF",
        "cache_hit": "False",
        "request_tags": "[]",
        "end_user": "",
        "api_base": "https://example.test",
        "model_group": "gpt",
        "model_id": "deployment",
        "mcp_namespaced_tool_name": None,
        "agent_id": None,
        "billing_agent_id": None,
        "requester_ip_address": None,
        "custom_llm_provider": "openai",
        "messages": "[]",
        "response": "{}",
        "proxy_server_request": "{}",
        "session_id": None,
        "request_duration_ms": 1000,
        "status": "success",
        "litellm_call_id": "call",
        **fields,
    }


def _charged(request_id: str, **fields: object) -> Mapping[str, object]:
    return _row(
        request_id,
        accounting_delivery_id=f"{request_id}-delivery",
        accounting_payload={"spend": 1.5},
        accounting_log_disabled=False,
        **fields,
    )


@pytest.fixture(autouse=True)
def _python_queue_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PrismaClient, "spend_log_queue_bytes", 0)


@pytest_asyncio.fixture
async def installed() -> AsyncIterator[tuple[FakePrismaClient, RustSpendLogWriter]]:
    client: Final = FakePrismaClient()
    writer: Final = await install_rust_spend_log_writer(client, await _unreachable_database(), _LEDGER_ON)
    assert writer is not None
    yield client, writer
    await uninstall_rust_spend_log_writer(client)


def _request_ids(rows: Sequence[Mapping[str, object]]) -> Sequence[object]:
    return [row.get("request_id") for row in rows]


@pytest.mark.asyncio
async def test_rust_queue_keeps_the_python_queue_order(
    installed: tuple[FakePrismaClient, RustSpendLogWriter],
) -> None:
    client, writer = installed

    await enqueue_spend_logs(client, [_row("a"), _charged("b")])
    await enqueue_spend_logs(client, [_row("c")], at_head=True)

    assert client.spend_log_transactions == []
    assert len(writer) == 3
    assert await _total_queued_spend_transactions(client) == 3
    assert _request_ids(await dequeue_spend_logs(client, 2)) == ["c", "a"]
    assert await dequeue_spend_logs(client, 10) == [_charged("b")]


@pytest.mark.asyncio
async def test_a_datetime_crosses_the_boundary_as_its_isoformat(
    installed: tuple[FakePrismaClient, RustSpendLogWriter],
) -> None:
    client, _ = installed
    started: Final = datetime(2026, 11, 2, 12, 0, 0, 500000, tzinfo=timezone.utc)

    await enqueue_spend_logs(client, [_row("a", startTime=started)])

    assert await dequeue_spend_logs(client, 1) == [_row("a", startTime=started.isoformat())]


@pytest.mark.asyncio
async def test_a_row_the_rust_writer_cannot_hold_is_dropped_alone_and_keeps_its_charge(
    installed: tuple[FakePrismaClient, RustSpendLogWriter],
) -> None:
    client, _ = installed
    sparse: Final = {
        "request_id": "sparse",
        "call_type": "acompletion",
        "spend": 2.0,
        "accounting_delivery_id": "sparse-delivery",
        "accounting_payload": {"spend": 2.0},
    }

    await enqueue_spend_logs(client, [_row("kept"), sparse, {"spend": 1.0}])

    assert await dequeue_spend_logs(client, 10) == [_row("kept"), accounting_only_spend_log(sparse)]


@pytest.mark.asyncio
async def test_the_python_queue_stays_when_the_ledger_is_off() -> None:
    client: Final = FakePrismaClient()
    await enqueue_spend_logs(client, [_row("a")])

    writer: Final = await install_rust_spend_log_writer(
        client, await _unreachable_database(), {"use_spend_logs_ledger": False}
    )

    assert writer is None
    assert rust_spend_log_writer(client) is None
    assert client.spend_log_transactions == [_row("a")]


@pytest.mark.asyncio
async def test_installing_moves_python_rows_ahead_and_uninstalling_moves_them_back() -> None:
    client: Final = FakePrismaClient()
    await enqueue_spend_logs(client, [_row("early"), _charged("earlier")])

    writer: Final = await install_rust_spend_log_writer(client, await _unreachable_database(), {})
    await enqueue_spend_logs(client, [_row("late")])

    assert writer is not None
    assert client.spend_log_transactions == []
    assert PrismaClient.spend_log_queue_bytes == 0
    assert len(writer) == 3
    await uninstall_rust_spend_log_writer(client)
    assert rust_spend_log_writer(client) is None
    assert len(writer) == 0
    assert client.spend_log_transactions == [_row("early"), _charged("earlier"), _row("late")]


@pytest.mark.asyncio
async def test_an_outage_raises_a_native_failure_and_requeues_the_pass_at_the_head() -> None:
    client: Final = FakePrismaClient()
    writer: Final = await install_rust_spend_log_writer(client, await _unreachable_database(_REFUSED), _LEDGER_ON)
    proxy_logging_obj: Final = _proxy_logging()
    rows: Final = [_charged("a"), _row("b")]
    await enqueue_spend_logs(client, rows)

    with pytest.raises(NativeDatabaseFailure) as failure:
        await update_spend(client, None, proxy_logging_obj)
    await asyncio.sleep(0)
    requeued: Final = await dequeue_spend_logs(client, 10)
    await uninstall_rust_spend_log_writer(client)

    assert writer is not None
    assert failure.value.unavailable is True
    assert failure.value.kind == "unreachable"
    assert requeued == rows
    assert proxy_logging_obj.failure_handler.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("parked", "left"), [(True, 1), (False, 3)], ids=["redis_buffer_on", "redis_buffer_off"])
async def test_a_failed_pass_is_parked_in_redis_when_the_buffer_is_on(parked: bool, left: int) -> None:
    writer: Final = FailingWriter([f'{{"request_id": "{index}"}}' for index in range(3)])
    proxy_logging_obj: Final = _proxy_logging(parked=parked)

    with (
        patch.object(RedisUpdateBuffer, "should_commit_spend_updates_to_redis", return_value=True),
        pytest.raises(NativeDatabaseFailure),
    ):
        await flush_in_rust(writer, FakePrismaClient(), proxy_logging_obj, 2)  # type: ignore[arg-type]
    await asyncio.sleep(0)

    store: Final = proxy_logging_obj.db_spend_update_writer.redis_update_buffer.store_spend_logs_in_redis
    assert len(writer) == left
    assert store.await_count == (1 if parked else 0)
    assert proxy_logging_obj.failure_handler.await_count == 1


@pytest.mark.asyncio
async def test_a_written_pass_feeds_the_guardrail_usage_rollup() -> None:  # test-quality-ok: no other effect
    written: Final[list[dict[str, object]]] = [{"request_id": "a", "metadata": {"guardrail_information": []}}]
    client: Final = FakePrismaClient()

    with patch("litellm.proxy.guardrails.usage_tracking.process_spend_logs_guardrail_usage", new=AsyncMock()) as rollup:
        await flush_in_rust(WritingWriter(written), client, _proxy_logging(), 10)  # type: ignore[arg-type]

    rollup.assert_awaited_once_with(prisma_client=client, logs_to_process=written)


class UnguardedWriter(WritingWriter):
    def __len__(self) -> int:
        return 1


@pytest.mark.parametrize(("writer", "retried"), [(UnguardedWriter([]), 1), (WritingWriter([]), 0)])
@pytest.mark.asyncio
async def test_a_pass_with_no_guardrail_results_retries_the_pending_guardrail_rollups(  # test-quality-ok: the retry is the only effect
    writer: WritingWriter, retried: int
) -> None:
    client: Final = FakePrismaClient()

    with patch("litellm.proxy.guardrails.usage_tracking.retry_pending_guardrail_usage", new=AsyncMock()) as retry:
        await flush_in_rust(writer, client, _proxy_logging(), 10)  # type: ignore[arg-type]

    assert retry.await_count == retried


@pytest.mark.asyncio
async def test_a_flush_request_from_another_thread_wakes_the_monitor() -> None:
    client: Final = FakePrismaClient()
    client.spend_log_flush_requested = asyncio.Event()
    loop: Final = asyncio.get_running_loop()

    thread: Final = threading.Thread(target=_request_flush_from_any_thread, args=(loop, client))
    thread.start()
    thread.join()

    await asyncio.wait_for(client.spend_log_flush_requested.wait(), timeout=1)

    assert client.spend_log_flush_requested.is_set()


@pytest.mark.asyncio
async def test_rows_left_at_shutdown_are_parked_from_the_rust_queue(
    installed: tuple[FakePrismaClient, RustSpendLogWriter],
) -> None:
    client, writer = installed
    proxy_logging_obj: Final = _proxy_logging(parked=True)
    await enqueue_spend_logs(client, [_row("a"), _charged("b")])

    await _park_remaining_spend_logs(client, proxy_logging_obj)  # type: ignore[arg-type]

    store: Final = proxy_logging_obj.db_spend_update_writer.redis_update_buffer.store_spend_logs_in_redis
    store.assert_awaited_once_with([_row("a"), _charged("b")])
    assert len(writer) == 0
