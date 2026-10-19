from typing import Final

import pytest
from pydantic import SecretStr

from tests.unit.proxy.spend_tracking._charge_receipt_cases import (
    RECEIPT,
    SETTINGS,
)


@pytest.mark.parametrize(
    "url,allowed",
    (
        ("https://receiver.example.test/billing/receipts", True),
        ("http://localhost/billing/receipts", True),
        ("http://127.0.0.1:8788/billing/receipts", True),
        ("http://[::1]:8788/billing/receipts", True),
        ("", False),
        ("https://receiver.example.test", False),
        ("https://user:password@receiver.example.test/receipts", False),
        ("https://receiver.example.test/receipts?secret=anything", False),
        ("https://receiver.example.test/receipts#fragment", False),
        ("https://receiver.example.test:999999/receipts", False),
        ("http://receiver.example.test/receipts", False),
        ("http://localhost.example.test/receipts", False),
        ("http://192.0.2.1/receipts", False),
        ("ftp://127.0.0.1/receipts", False),
    ),
)
def test_destination_accepts_only_tls_or_loopback_without_embedded_credentials(url: str, allowed: bool) -> None:
    error: Final = SETTINGS.model_copy(update={"destination_url": url}).configuration_error()
    assert (error is None) == allowed
    assert error is None or "password" not in error and "anything" not in error


@pytest.mark.parametrize("secret", ("unit-secret-too-small", "a" * 32 + "\n", "é" * 32))
def test_configuration_rejects_unsafe_shared_secrets_without_echoing_them(secret: str) -> None:
    error: Final = SETTINGS.model_copy(update={"destination_secret": SecretStr(secret)}).configuration_error()
    assert error is not None and secret not in error


@pytest.mark.parametrize("amount", ("0.250", "0.250000000000000000000000"))
def test_receipt_identity_compares_numeric_amount_without_ignoring_other_fields(amount: str) -> None:
    assert RECEIPT.fingerprint() == RECEIPT.model_copy(update={"amount_usd": amount}).fingerprint()
    assert RECEIPT.fingerprint() != RECEIPT.model_copy(update={"amount_usd": "0.26"}).fingerprint()
    assert RECEIPT.fingerprint() != RECEIPT.model_copy(update={"request_id": "other"}).fingerprint()
