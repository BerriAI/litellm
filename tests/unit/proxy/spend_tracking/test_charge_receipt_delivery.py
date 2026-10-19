import json
from typing import Final

import httpx
import pytest

from litellm.proxy.spend_tracking.charge_receipt_delivery import ChargeReceiptPublisher
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptStore
from tests.unit.proxy.spend_tracking._charge_receipt_cases import RECEIPT, SETTINGS, ScriptedQuery


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,reason",
    (
        ("offline", "receiver_unavailable"),
        ("status", "receiver_unavailable"),
        ("redirect", "receiver_unavailable"),
        ("compressed", "invalid_acknowledgement"),
        ("oversized", "invalid_acknowledgement"),
        ("invalid", "invalid_acknowledgement"),
        ("unknown", "invalid_acknowledgement"),
        ("empty", "receipt_rejected"),
    ),
)
async def test_unacknowledged_delivery_is_retained_for_retry(case: str, reason: str) -> None:
    query: Final = ScriptedQuery((({"payload": RECEIPT.model_dump()},), ()))

    def receiver(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == SETTINGS.destination_url
        assert json.loads(request.content) == {"events": [RECEIPT.model_dump()]}
        assert request.headers["authorization"] == "Bearer " + SETTINGS.destination_secret.get_secret_value()
        if case == "offline":
            raise httpx.ReadError("receiver offline", request=request)
        if case == "status":
            return httpx.Response(503)
        if case == "redirect":
            return httpx.Response(307, headers={"location": "https://another.example.test/receipts"})
        if case == "compressed":
            return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=httpx.ByteStream(b"unused"))
        if case == "oversized":
            return httpx.Response(200, stream=httpx.ByteStream(b" " * (256 * 1024 + 1)))
        if case == "invalid":
            return httpx.Response(200, content=b"invalid JSON")
        return httpx.Response(200, json={"acknowledged": ["unknown"] if case == "unknown" else []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
        publisher: Final = ChargeReceiptPublisher(ChargeReceiptStore(query), SETTINGS, client)
        assert await publisher.deliver_once() == 0
    assert len(query.calls) == 2
    assert (RECEIPT.event_id,) in query.calls[1]
    assert publisher.owner in query.calls[1] and reason in query.calls[1]


@pytest.mark.asyncio
async def test_partial_acknowledgement_never_discards_the_unconfirmed_receipt() -> None:
    second: Final = RECEIPT.model_copy(update={"event_id": "second-event"})
    query: Final = ScriptedQuery(
        (
            (
                {"payload": RECEIPT.model_dump()},
                {"payload": second.model_dump()},
            ),
            (),
            (),
        )
    )

    def receiver(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"events": [RECEIPT.model_dump(), second.model_dump()]}
        return httpx.Response(200, json={"acknowledged": [RECEIPT.event_id]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
        publisher: Final = ChargeReceiptPublisher(ChargeReceiptStore(query), SETTINGS, client)
        assert await publisher.deliver_once() == 1
    assert (RECEIPT.event_id,) in query.calls[1] and publisher.owner in query.calls[1]
    assert (second.event_id,) in query.calls[2]
    assert publisher.owner in query.calls[2] and "receipt_rejected" in query.calls[2]
