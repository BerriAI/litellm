import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import pytest

from litellm.repositories.table_repositories import SpendLogsRepository

RAW_BODY: Final = "the full raw prompt body"


@dataclass
class FakeTransaction:
    accepted: frozenset[str]
    executed: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)  # mutable-ok: records statements
    queried: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)  # mutable-ok: records statements

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        self.queried.append((query, args))
        if 'INSERT INTO "LiteLLM_SpendLogDelivery"' in query:
            offered = json.loads(str(args[0]))
            return [{"delivery_id": row["delivery_id"]} for row in offered if row["delivery_id"] in self.accepted]
        return []

    async def execute_raw(self, query: str, *args: object) -> int:
        self.executed.append((query, args))
        return 1

    def inserted(self, table: str) -> list[Mapping[str, object]]:
        return [
            row
            for query, args in self.executed
            if f'INSERT INTO "{table}"' in query
            for row in json.loads(str(args[0]))  # comprehension-ok: flattens each recorded batch for assertions
        ]


def _row(delivery_id: str, *, with_body: bool = True) -> Mapping[str, object]:
    return {
        "request_id": f"request-{delivery_id}",
        "call_type": "acompletion",
        "spend": 0.5,
        "startTime": "2026-11-18T12:00:00+00:00",
        "messages": "[]",
        "accounting_delivery_id": delivery_id,
        "accounting_payload": {"request_id": f"request-{delivery_id}"},
        "lens_export": {
            "record": {"request_id": f"request-{delivery_id}", "spend": 0.5, "messages": "preview"},
            "payloads": (
                [
                    {
                        "field": "messages",
                        "team_id": "team",
                        "api_key": "hash",
                        "event_time_ms": 1,
                        "content": RAW_BODY,
                        "bytes": len(RAW_BODY),
                        "sha256": hashlib.sha256(RAW_BODY.encode()).hexdigest(),
                    }
                ]
                if with_body
                else []
            ),
        },
    }


@pytest.mark.asyncio
async def test_only_accepted_deliveries_reach_the_lens_outbox() -> None:
    transaction: Final = FakeTransaction(accepted=frozenset({"accepted"}))

    accepted: Final = await SpendLogsRepository._insert_ledger(  # pyright: ignore[reportPrivateUsage]  # the transaction body is the unit under test
        transaction,  # pyright: ignore[reportArgumentType]  # minimal transaction fake
        (_row("accepted"), _row("duplicate")),
        write_spend_logs=True,
    )

    assert accepted == 1
    assert [row["event_id"] for row in transaction.inserted("LiteLLM_LensLedgerExport")] == ["accepted"]
    assert transaction.inserted("LiteLLM_LensLedgerExport")[0]["event"] == {
        "request_id": "request-accepted",
        "spend": 0.5,
        "messages": "preview",
    }
    assert [(row["event_id"], row["field"]) for row in transaction.inserted("LiteLLM_LensLedgerPayload")] == [
        ("accepted", "messages")
    ]


@pytest.mark.asyncio
async def test_raw_bodies_never_reach_the_local_spend_log_row() -> None:
    transaction: Final = FakeTransaction(accepted=frozenset({"accepted"}))

    await SpendLogsRepository._insert_ledger(  # pyright: ignore[reportPrivateUsage]  # the transaction body is the unit under test
        transaction,  # pyright: ignore[reportArgumentType]  # minimal transaction fake
        (_row("accepted"),),
        write_spend_logs=True,
    )

    local_inserts: Final = [
        args for query, args in transaction.queried if "LiteLLM_SpendLogs" in query and "Delivery" not in query
    ]
    assert local_inserts
    assert RAW_BODY not in repr(local_inserts)
    assert "lens_export" not in repr(local_inserts)


@pytest.mark.asyncio
async def test_rows_without_bodies_write_no_payload_rows() -> None:
    transaction: Final = FakeTransaction(accepted=frozenset({"accepted"}))

    await SpendLogsRepository._insert_ledger(  # pyright: ignore[reportPrivateUsage]  # the transaction body is the unit under test
        transaction,  # pyright: ignore[reportArgumentType]  # minimal transaction fake
        (_row("accepted", with_body=False),),
        write_spend_logs=False,
    )

    assert len(transaction.inserted("LiteLLM_LensLedgerExport")) == 1
    assert transaction.inserted("LiteLLM_LensLedgerPayload") == []


@pytest.mark.parametrize(
    ("delivery_id", "event_id"),
    [
        ("0f1e2d3c4b5a69788796a5b4c3d2e1f0", "0f1e2d3c4b5a69788796a5b4c3d2e1f0"),
        ("batch:abc/cost", hashlib.sha256(b"batch:abc/cost").hexdigest()),
    ],
    ids=["uuid_hex", "job_cost_request_id"],
)
@pytest.mark.asyncio
async def test_event_identity_is_the_delivery_id_or_its_stable_digest(delivery_id: str, event_id: str) -> None:
    transaction: Final = FakeTransaction(accepted=frozenset({delivery_id}))

    await SpendLogsRepository._insert_ledger(  # pyright: ignore[reportPrivateUsage]  # the transaction body is the unit under test
        transaction,  # pyright: ignore[reportArgumentType]  # minimal transaction fake
        (_row(delivery_id),),
        write_spend_logs=False,
    )

    assert [row["event_id"] for row in transaction.inserted("LiteLLM_LensLedgerExport")] == [event_id]
    assert [row["event_id"] for row in transaction.inserted("LiteLLM_LensLedgerPayload")] == [event_id]
