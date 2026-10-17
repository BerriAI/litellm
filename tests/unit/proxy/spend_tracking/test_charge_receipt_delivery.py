import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Final

import httpx
import pytest

from litellm.proxy.spend_tracking.charge_receipt_delivery import ChargeReceiptPublisher, run_charge_receipt_publisher
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptStore
from tests.unit.proxy.spend_tracking._charge_receipt_cases import RECEIPT, SETTINGS, ScriptedQuery


class Body(httpx.AsyncByteStream):
    def __init__(self, value: bytes) -> None:
        self.value: Final = value

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.value


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
            return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=Body(b"unused"))
        if case == "oversized":
            return httpx.Response(200, stream=Body(b" " * (256 * 1024 + 1)))
        if case == "invalid":
            return httpx.Response(200, content=b"invalid JSON")
        return httpx.Response(200, json={"acknowledged": ["unknown"] if case == "unknown" else []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
        publisher: Final = ChargeReceiptPublisher(ChargeReceiptStore(query), SETTINGS, client)
        assert await publisher.deliver_once() == 0
    assert len(query.calls) == 2
    assert (RECEIPT.event_id,) in query.calls[1].parameters
    assert publisher.owner in query.calls[1].parameters and reason in query.calls[1].parameters


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
    assert (RECEIPT.event_id,) in query.calls[1].parameters and publisher.owner in query.calls[1].parameters
    assert (second.event_id,) in query.calls[2].parameters
    assert publisher.owner in query.calls[2].parameters and "receipt_rejected" in query.calls[2].parameters


@pytest.mark.asyncio
async def test_empty_outbox_never_contacts_the_receiver() -> None:
    query: Final = ScriptedQuery(((),))

    def receiver(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Empty outbox contacted {request.url.host}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
        assert await ChargeReceiptPublisher(ChargeReceiptStore(query), SETTINGS, client).deliver_once() == 0
    assert len(query.calls) == 1


@pytest.mark.asyncio
async def test_fully_acknowledged_delivery_does_not_schedule_a_retry() -> None:
    query: Final = ScriptedQuery((({"payload": RECEIPT.model_dump()},), ()))

    def receiver(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"events": [RECEIPT.model_dump()]}
        return httpx.Response(200, json={"acknowledged": [RECEIPT.event_id]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
        publisher: Final = ChargeReceiptPublisher(ChargeReceiptStore(query), SETTINGS, client)
        assert await publisher.deliver_once() == 1
    assert len(query.calls) == 2 and query.calls[1].parameters == ((RECEIPT.event_id,), publisher.owner)


@pytest.mark.asyncio
async def test_database_outage_does_not_prevent_the_publisher_from_stopping() -> None:
    stop: Final = asyncio.Event()

    async def offline(query: str, *parameters: object) -> Sequence[Mapping[str, object]]:
        stop.set()
        raise ConnectionError("database offline")

    async with httpx.AsyncClient() as client:
        await ChargeReceiptPublisher(ChargeReceiptStore(offline), SETTINGS, client).run(stop)
    assert stop.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("disabled", "invalid", "no-database"))
async def test_publisher_rejects_incomplete_configuration_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    for name, value in (
        ("ENABLED", "false" if mode == "disabled" else "true"),
        ("GATEWAY_SCOPE", "invalid" if mode == "invalid" else SETTINGS.gateway_scope),
        ("KEY_HASHES", '["' + RECEIPT.key_hash + '"]'),
        ("DESTINATION_URL", SETTINGS.destination_url),
        ("DESTINATION_SECRET", SETTINGS.destination_secret.get_secret_value()),
    ):
        monkeypatch.setenv("LITELLM_CHARGE_RECEIPTS_" + name, value)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit) as error:
        await run_charge_receipt_publisher()
    assert error.value.code and SETTINGS.destination_secret.get_secret_value() not in str(error.value)
