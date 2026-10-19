from typing import Final

import pytest
from pydantic import SecretStr

from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptStore
from litellm.types.charge_receipts import CommittedCharge, FailedCharge
from tests.unit.proxy.spend_tracking._charge_receipt_cases import (
    RECEIPT,
    SETTINGS,
    Rows,
    ScriptedQuery,
    acknowledged,
)


@pytest.mark.parametrize(
    "url",
    (
        "https://receiver.example.test/billing/receipts",
        "http://localhost/billing/receipts",
        "http://127.0.0.1:8788/billing/receipts",
        "http://[::1]:8788/billing/receipts",
    ),
)
def test_destination_accepts_tls_or_loopback_only(url: str) -> None:
    assert SETTINGS.model_copy(update={"destination_url": url}).configuration_error() is None


@pytest.mark.parametrize(
    "url",
    (
        "",
        "https://receiver.example.test",
        "https://user:password@receiver.example.test/receipts",
        "https://receiver.example.test/receipts?secret=anything",
        "https://receiver.example.test/receipts#fragment",
        "https://receiver.example.test:999999/receipts",
        "http://receiver.example.test/receipts",
        "http://localhost.example.test/receipts",
        "http://192.0.2.1/receipts",
        "ftp://127.0.0.1/receipts",
    ),
)
def test_destination_rejects_unqualified_or_credential_bearing_urls(url: str) -> None:
    error: Final = SETTINGS.model_copy(update={"destination_url": url}).configuration_error()
    assert error is not None and "password" not in error and "anything" not in error


@pytest.mark.parametrize("secret", ("unit-secret-too-small", "a" * 32 + "\n", "é" * 32))
def test_configuration_rejects_unsafe_shared_secrets_without_echoing_them(secret: str) -> None:
    error: Final = SETTINGS.model_copy(update={"destination_secret": SecretStr(secret)}).configuration_error()
    assert error is not None and secret not in error


def test_configuration_requires_scope_and_keys_only_when_enabled() -> None:
    assert SETTINGS.model_copy(update={"gateway_scope": "invalid"}).configuration_error() is not None
    assert SETTINGS.model_copy(update={"key_hashes": frozenset()}).configuration_error() is not None
    assert SETTINGS.model_copy(update={"enabled": False, "gateway_scope": ""}).configuration_error() is None


@pytest.mark.parametrize("amount", ("0.250", "0.250000000000000000000000"))
def test_receipt_identity_compares_numeric_amount_without_ignoring_other_fields(amount: str) -> None:
    assert RECEIPT.fingerprint() == RECEIPT.model_copy(update={"amount_usd": amount}).fingerprint()
    assert RECEIPT.fingerprint() != RECEIPT.model_copy(update={"amount_usd": "0.26"}).fingerprint()
    assert RECEIPT.fingerprint() != RECEIPT.model_copy(update={"request_id": "other"}).fingerprint()


@pytest.mark.asyncio
@pytest.mark.parametrize("installed,durable", ((True, True), (False, True), (True, False)))
async def test_readiness_requires_installed_schema_and_durable_storage(installed: bool, durable: bool) -> None:
    store: Final = ChargeReceiptStore(ScriptedQuery((({"installed": installed, "durable": durable},),)))
    assert await store.ready() is (installed and durable)


@pytest.mark.asyncio
async def test_database_failure_never_claims_readiness_or_success() -> None:
    query: Final = ScriptedQuery((ConnectionError("offline"), ConnectionError("offline")))
    store: Final = ChargeReceiptStore(query)
    assert not await store.ready()
    assert await store.append(RECEIPT) == FailedCharge(RECEIPT.key_hash, "database_unavailable")


@pytest.mark.asyncio
async def test_duplicate_commit_requires_the_original_immutable_payload() -> None:
    store: Final = ChargeReceiptStore(ScriptedQuery(((), acknowledged())))
    assert await store.append(RECEIPT) == CommittedCharge(
        RECEIPT.event_id, RECEIPT.key_hash, RECEIPT.amount_usd, True, True
    )
    conflict: Final = ChargeReceiptStore(ScriptedQuery(((), acknowledged())))
    assert await conflict.append(RECEIPT.model_copy(update={"amount_usd": "0.26"})) == FailedCharge(
        RECEIPT.key_hash, "receipt_conflict"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    (
        ({"event_id": "bad"},),
        ({"event_id": "other", "payload_hash": "wrong", "key_counter_updated": True},),
        acknowledged() * 2,
    ),
)
async def test_unverifiable_database_ack_is_never_reported_as_committed(rows: Rows) -> None:
    store: Final = ChargeReceiptStore(ScriptedQuery((rows,)))
    assert await store.append(RECEIPT) == FailedCharge(RECEIPT.key_hash, "invalid_database_result")


@pytest.mark.asyncio
async def test_a_suppressed_insert_without_an_existing_receipt_remains_unconfirmed() -> None:
    store: Final = ChargeReceiptStore(ScriptedQuery(((), ())))
    assert await store.append(RECEIPT) == FailedCharge(RECEIPT.key_hash, "unsafe_durability")
