"""Record what the Python spend log writer does, as goldens the Rust writer's tests replay.

`queue.json` holds the byte count `spend_log_row_bytes` gives each row and its accounting-only row, and
the queue `enqueue_spend_logs` leaves after an overflow. `stored.json` holds the `LiteLLM_SpendLogs` and
`LiteLLM_SpendLogDelivery` rows `ProxyUpdateSpend.update_spend_logs` stores for a flush that is replayed
once, which needs a Prisma-migrated database in DATABASE_URL.

Run from the repository root:

    PYTHONPATH=.:litellm/scripts:litellm/runtime DATABASE_URL=postgresql://... \
        python litellm/rust/crates/spend-log-writer/scripts/record_goldens.py

The recorder runs the Python implementation that the ledger-off path still uses; once that path is
deleted, the committed goldens become fixed fixtures and the recorder goes with it.
"""

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import psycopg
from psycopg.rows import dict_row

from litellm.proxy.db.spend_log_batching import spend_log_row_bytes
from litellm.proxy.utils import (
    PrismaClient,
    ProxyUpdateSpend,
    accounting_only_spend_log,
    enqueue_spend_logs,
    jsonify_object,
)

GOLDENS: Final = Path(__file__).resolve().parents[1] / "tests" / "goldens"
PREFIX: Final = "golden"
_PLACEHOLDER: Final = re.compile(r"\$(\d+)")
_STARTED: Final = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)


def _iso(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _text(row: Mapping[str, object]) -> str:
    return json.dumps(row, default=_iso)


def _row(name: str, **fields: object) -> dict[str, object]:
    request_id: Final = f"{PREFIX}-{name}"
    return {
        "request_id": request_id,
        "call_type": "acompletion",
        "api_key": "hashed",
        "spend": 0.25,
        "total_tokens": 3,
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "startTime": _STARTED.isoformat(),
        "endTime": (_STARTED + timedelta(seconds=1)).isoformat(),
        "completionStartTime": (_STARTED + timedelta(milliseconds=500)).isoformat(),
        "model": "gpt",
        "user": "user",
        "team_id": "team",
        "organization_id": "",
        "metadata": '{"status": "success", "tenant": "a", "nested": {"x": [1, 2.5, null]}}',
        "cache_key": "Cache OFF",
        "cache_hit": "False",
        "request_tags": '["tag", "other"]',
        "end_user": "",
        "api_base": "https://example.test",
        "model_group": "gpt",
        "model_id": "deployment",
        "mcp_namespaced_tool_name": None,
        "agent_id": None,
        "billing_agent_id": None,
        "requester_ip_address": None,
        "custom_llm_provider": "openai",
        "messages": '[{"role": "user", "content": "h\\u00e9llo \\u2603"}]',
        "response": '{"id": "resp", "choices": [{"text": "a \\"quoted\\" \\\\ value"}]}',
        "proxy_server_request": "{}",
        "session_id": None,
        "request_duration_ms": 1000,
        "status": "success",
        "litellm_call_id": "call",
        **fields,
    }


def _ledger(name: str, spend: float = 0.25, **fields: object) -> dict[str, object]:
    return {
        **_row(name, spend=spend),
        "accounting_delivery_id": f"{PREFIX}-{name}-delivery",
        "accounting_payload": {"request_id": f"{PREFIX}-{name}", "spend": spend, "nested": {"a": [1, {"b": "c"}]}},
        "accounting_log_disabled": False,
        **fields,
    }


class _FakePrismaClient:
    def __init__(self) -> None:
        self.spend_log_transactions: list[Mapping[str, object]] = []
        self._spend_log_transactions_lock = asyncio.Lock()
        self.db = MagicMock()

    def jsonify_object(self, data: Mapping[str, object]) -> Mapping[str, object]:
        return jsonify_object(dict(data))


async def _queue_golden() -> Mapping[str, object]:
    measured: Final = (
        _ledger("ascii"),
        _ledger("unicode", messages="héllo ☃ \U0001d11e", metadata='{"tags": ["é", 1, 2.5, null, true]}'),
        _ledger("escapes", response='{"text": "quote \\" backslash \\\\ newline \\n tab \\t"}'),
        _ledger("numbers", spend=0.1 + 0.2, total_tokens=12345678901234, prompt_tokens=-3),
        _ledger("small_float", spend=3.5e-05),
        _row("uncharged"),
        {key: value for key, value in _ledger("settle_once").items() if key != "accounting_delivery_id"},
    )
    rows: Final = tuple(
        {**_ledger(str(index), messages="x" * 200)} if index % 2 else _row(str(index), messages="x" * 200)
        for index in range(10)
    )
    budget: Final = spend_log_row_bytes(rows[0]) * 4
    client: Final = _FakePrismaClient()
    PrismaClient.spend_log_queue_bytes = 0
    for row in rows:
        await enqueue_spend_logs(client, [row], max_bytes=budget)  # type: ignore[arg-type]
    await enqueue_spend_logs(client, [_ledger("replayed", messages="y" * 600)], at_head=True, max_bytes=budget)  # type: ignore[arg-type]
    return {
        "measured": [
            {
                "row": _text(row),
                "bytes": spend_log_row_bytes(row),
                "accounting_only_bytes": spend_log_row_bytes(accounting_only_spend_log(row))
                if "accounting_payload" in row
                else None,
            }
            for row in measured
        ],
        "overflow": {
            "budget": budget,
            "enqueued": [_text(row) for row in rows],
            "replayed_at_head": _text(_ledger("replayed", messages="y" * 600)),
            "queue": [
                {"request_id": row["request_id"], "accounting_log_disabled": row.get("accounting_log_disabled")}
                for row in client.spend_log_transactions
            ],
        },
    }


class _Transaction:
    def __init__(self, connection: psycopg.AsyncConnection[dict[str, object]]) -> None:
        self._connection = connection

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        cursor: Final = await self._connection.execute(_PLACEHOLDER.sub(r"%(p\1)s", query), _named(args))
        return await cursor.fetchall() if cursor.description is not None else []

    async def execute_raw(self, query: str, *args: object) -> int:
        cursor: Final = await self._connection.execute(_PLACEHOLDER.sub(r"%(p\1)s", query), _named(args))
        return cursor.rowcount


def _named(args: Sequence[object]) -> Mapping[str, object]:
    return {f"p{index}": value for index, value in enumerate(args, start=1)}


class _Database:
    def __init__(self, url: str) -> None:
        self._url = url

    @asynccontextmanager
    async def tx(self, timeout: timedelta) -> AsyncIterator[_Transaction]:
        async with (
            await psycopg.AsyncConnection.connect(self._url, row_factory=dict_row) as connection,
            connection.transaction(),
        ):
            yield _Transaction(connection)


async def _stored(url: str) -> Mapping[str, object]:
    async with await psycopg.AsyncConnection.connect(url, row_factory=dict_row) as connection:
        logs: Final = await (
            await connection.execute(
                """SELECT to_jsonb(s) - 'created_at' - 'updated_at' AS row FROM "LiteLLM_SpendLogs" s """
                "WHERE request_id LIKE %s ORDER BY request_id",
                (f"{PREFIX}-%",),
            )
        ).fetchall()
        deliveries: Final = await (
            await connection.execute(
                """SELECT to_jsonb(d) - 'created_at' - 'next_attempt_at' AS row FROM "LiteLLM_SpendLogDelivery" d """
                "WHERE request_id LIKE %s ORDER BY delivery_id",
                (f"{PREFIX}-%",),
            )
        ).fetchall()
        await connection.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (f"{PREFIX}-%",))
        await connection.execute('DELETE FROM "LiteLLM_SpendLogDelivery" WHERE request_id LIKE %s', (f"{PREFIX}-%",))
    return {"logs": [row["row"] for row in logs], "deliveries": [row["row"] for row in deliveries]}


async def _stored_golden(url: str) -> Mapping[str, object]:
    rows: Final = (
        _ledger("full"),
        _ledger("full", spend=9.0),
        _ledger(
            "batch",
            spend=4.0,
            call_type="aretrieve_batch",
            accounting_delivery_id=None,
            accounting_payload='{"spend": 4.0}',
        ),
        {
            "request_id": f"{PREFIX}-disabled",
            "call_type": "acompletion",
            "accounting_delivery_id": f"{PREFIX}-disabled-delivery",
            "accounting_payload": {"spend": 0.5},
            "accounting_log_disabled": True,
        },
        _ledger(
            "odd_values",
            metadata='{"status": "success", "spend": NaN, "nul": "a\\u0000b"}',
            response='{"text": "x\\u0000y", "big": 1e300}',
        ),
    )
    batch: Final = [{k: v for k, v in row.items() if not (k == "accounting_delivery_id" and v is None)} for row in rows]
    client: Final = _FakePrismaClient()
    client.writer_db = _Database(url)  # type: ignore[attr-defined]
    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    for _ in range(2):
        PrismaClient.spend_log_queue_bytes = 0
        await enqueue_spend_logs(client, batch)  # type: ignore[arg-type]
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=3,
            prisma_client=client,  # type: ignore[arg-type]
            db_writer_client=None,
            proxy_logging_obj=proxy_logging_obj,
        )
    return {"rows": [_text(row) for row in batch], **(await _stored(url))}


async def main() -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    (GOLDENS / "queue.json").write_text(json.dumps(await _queue_golden(), indent=2) + "\n")
    (GOLDENS / "stored.json").write_text(
        json.dumps(await _stored_golden(os.environ["DATABASE_URL"]), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
