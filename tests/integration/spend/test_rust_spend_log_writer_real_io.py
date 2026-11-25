import asyncio
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy import proxy_server
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.proxy.db.rust_spend_log_writer import (
    flush_in_rust,
    install_rust_spend_log_writer,
    uninstall_rust_spend_log_writer,
)
from litellm.proxy.utils import PrismaClient, dequeue_spend_logs, enqueue_spend_logs
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase, NativeDatabaseFailure

_REFUSED: Final = "postgresql://litellm:secret@127.0.0.1:1/proxy?pool_timeout=1"
_LEDGER_ON: Final[Mapping[str, object]] = {"use_spend_logs_ledger": True}


class FakePrismaClient:
    def __init__(self) -> None:
        self.spend_log_transactions: list[Mapping[str, object]] = []
        self._spend_log_transactions_lock = asyncio.Lock()
        self.spend_log_flush_requested: asyncio.Event | None = None


async def _unreachable_database() -> NativeDatabase:
    return await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(_REFUSED, 1),
        None,
    )


def _proxy_logging() -> MagicMock:
    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    buffer: Final = proxy_logging_obj.db_spend_update_writer.redis_update_buffer
    buffer.redis_cache = MagicMock()
    buffer.store_spend_logs_in_redis = AsyncMock(return_value=True)
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


@pytest.mark.asyncio
async def test_a_failed_filtered_pass_only_parks_its_own_remaining_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PrismaClient, "spend_log_queue_bytes", 0)
    client: Final = FakePrismaClient()
    writer: Final = await install_rust_spend_log_writer(client, await _unreachable_database(), _LEDGER_ON)
    assert writer is not None
    proxy_logging_obj: Final = _proxy_logging()
    monkeypatch.setattr(proxy_server, "disable_spend_logs", True)
    await enqueue_spend_logs(client, [_row("drop"), _charged("charge"), _row("later")])

    try:
        with (
            patch.object(RedisUpdateBuffer, "should_commit_spend_updates_to_redis", return_value=True),
            pytest.raises(NativeDatabaseFailure),
        ):
            await flush_in_rust(writer, client, proxy_logging_obj, 2)

        store: Final = proxy_logging_obj.db_spend_update_writer.redis_update_buffer.store_spend_logs_in_redis
        (parked,) = store.await_args.args
        assert tuple(parked) == (
            {
                "request_id": "charge",
                "call_type": "acompletion",
                "accounting_delivery_id": "charge-delivery",
                "accounting_payload": {"spend": 1.5},
                "accounting_log_disabled": True,
            },
        )
        assert await dequeue_spend_logs(client, 10) == [_row("later")]
    finally:
        await uninstall_rust_spend_log_writer(client)
