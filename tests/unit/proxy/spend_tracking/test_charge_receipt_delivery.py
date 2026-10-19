import asyncio
import json
from typing import Final

import httpx
import pytest

from litellm.proxy.spend_tracking.charge_receipt_delivery import ChargeReceiptPublisher, run_charge_receipt_publisher
from litellm.proxy.spend_tracking.charge_receipt_request import configured_charge_receipts
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptStore
from tests.unit.proxy.spend_tracking._charge_receipt_cases import RECEIPT, SETTINGS, ScriptedQuery


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("disabled", "invalid_scope", "no_keys", "no_database"))
async def test_startup_rejects_incomplete_opt_in_but_leaves_disabled_gateways_unchanged(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    values: Final = {
        "ENABLED": str(mode != "disabled"),
        "GATEWAY_SCOPE": "" if mode == "invalid_scope" else RECEIPT.gateway_scope,
        "KEY_HASHES": json.dumps([] if mode == "no_keys" else [RECEIPT.key_hash]),
        "DESTINATION_URL": SETTINGS.destination_url,
        "DESTINATION_SECRET": SETTINGS.destination_secret.get_secret_value(),
    }
    for name, value in values.items():
        monkeypatch.setenv("LITELLM_CHARGE_RECEIPTS_" + name, value)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    configured: Final = await configured_charge_receipts(None)
    assert configured is None if mode == "disabled" else isinstance(configured, str)
    with pytest.raises(SystemExit) as error:
        await run_charge_receipt_publisher()
    assert str(error.value) and SETTINGS.destination_secret.get_secret_value() not in str(error.value)


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
async def test_publisher_recovers_from_database_failure_and_retains_unacknowledged_receipts() -> None:
    stop: Final = asyncio.Event()
    second: Final = RECEIPT.model_copy(update={"event_id": "second-event"})
    query: Final = ScriptedQuery(
        (
            ConnectionError("database temporarily unavailable"),
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
        stop.set()
        return httpx.Response(200, json={"acknowledged": [RECEIPT.event_id]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
        publisher: Final = ChargeReceiptPublisher(
            ChargeReceiptStore(query), SETTINGS.model_copy(update={"poll_seconds": 0}), client
        )
        await publisher.run(stop)
    assert len(query.calls) == 4
    assert (RECEIPT.event_id,) in query.calls[2] and publisher.owner in query.calls[2]
    assert (second.event_id,) in query.calls[3]
    assert publisher.owner in query.calls[3] and "receipt_rejected" in query.calls[3]
