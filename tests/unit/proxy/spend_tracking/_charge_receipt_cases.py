import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, TypeAlias

import pytest
from pydantic import SecretStr

from litellm.proxy.spend_tracking.charge_receipts import ChargeReceipt, ChargeReceiptSettings
from litellm.proxy.spend_tracking.spend_event import SpendEvent, project_spend_event

Rows: TypeAlias = tuple[Mapping[str, object], ...]
Reply: TypeAlias = Rows | Exception

RECEIPT: Final = ChargeReceipt(
    event_id="owned-event",
    request_id="owned-request",
    gateway_scope="a" * 64,
    key_hash="b" * 64,
    amount_usd="0.25",
    model="owned-billing-model",
    input_tokens=4,
    output_tokens=2,
    total_tokens=6,
)
SETTINGS: Final = ChargeReceiptSettings(
    enabled=True,
    gateway_scope=RECEIPT.gateway_scope,
    key_hashes=frozenset((RECEIPT.key_hash,)),
    destination_url="https://receiver.example.test/billing/receipts",
    destination_secret=SecretStr("unit-receipt-secret-no-production-credentials"),
)
STAMP: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class QueryCall:
    query: str
    parameters: tuple[object, ...]


class ScriptedQuery:
    def __init__(self, replies: tuple[Reply, ...] = (), *, hold: bool = False) -> None:
        self.replies = replies
        self.hold: Final = hold
        self.calls: tuple[QueryCall, ...] = ()
        self.entered: Final = asyncio.Event()
        self.release: Final = asyncio.Event()

    async def __call__(self, query: str, *parameters: object) -> Sequence[Mapping[str, object]]:
        index: Final = len(self.calls)
        self.calls = (*self.calls, QueryCall(query, parameters))
        if index >= len(self.replies):
            pytest.fail(f"Unexpected database call {index + 1}")
        self.entered.set()
        if self.hold:
            await self.release.wait()
        reply: Final = self.replies[index]
        if isinstance(reply, Exception):
            raise reply
        return reply


def acknowledged(receipt: ChargeReceipt = RECEIPT) -> Rows:
    return (
        {
            "event_id": receipt.event_id,
            "payload_hash": receipt.fingerprint(),
            "key_counter_updated": True,
        },
    )


def priced_event(*, amount: float | None = 0.25) -> SpendEvent:
    value: Final = project_spend_event(
        {
            "model": RECEIPT.model,
            "litellm_params": {},
            "response_cost": amount,
            "messages": ({"role": "user", "content": "private prompt must not be journaled"},),
            "standard_logging_object": {
                "response_cost": amount,
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        },
        {"id": "owned-response", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
        STAMP,
        STAMP,
        store_bodies=False,
    )
    assert isinstance(value, SpendEvent)
    return value
