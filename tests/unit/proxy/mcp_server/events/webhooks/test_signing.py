import base64
import hashlib
import hmac
from typing import Final

import pytest

from litellm.proxy.mcp_server.events.webhooks.signing import (
    new_secret,
    new_webhook_id,
    parse_secret,
    sign,
    signed_headers,
    verify,
)

KEY: Final = bytes(range(32))
SECRET: Final = "whsec_" + base64.b64encode(KEY).decode()
NOW: Final = 1_800_000_000


def test_the_signature_is_the_standard_webhooks_hmac_over_id_timestamp_and_raw_body() -> None:
    body: Final = b'{"eventId":"evt_1"}'
    expected: Final = base64.b64encode(hmac.new(KEY, b"msg_1.1800000000." + body, hashlib.sha256).digest()).decode()
    assert sign(KEY, "msg_1", NOW, body) == "v1," + expected


@pytest.mark.parametrize(
    ("value", "valid"),
    (
        (SECRET, True),
        ("whsec_" + base64.b64encode(bytes(24)).decode(), True),
        ("whsec_" + base64.b64encode(bytes(64)).decode(), True),
        ("whsec_" + base64.b64encode(bytes(23)).decode(), False),
        ("whsec_" + base64.b64encode(bytes(65)).decode(), False),
        (base64.b64encode(KEY).decode(), False),
        ("whsec_not base64!", False),
    ),
)
def test_secrets_must_be_whsec_base64_of_24_to_64_bytes(value: str, valid: bool) -> None:
    assert (parse_secret(value) is not None) is valid


def test_generated_secrets_parse_and_ids_are_header_safe_and_unique() -> None:
    assert parse_secret(new_secret()) is not None
    ids: Final = {new_webhook_id() for _ in range(50)} | {new_webhook_id("gap")}
    assert len(ids) == 51
    assert all(item.startswith("msg_") and item.replace("_", "").isalnum() for item in ids)
    assert new_webhook_id("terminated").startswith("msg_terminated_")


def _headers(body: bytes, *, timestamp: int = NOW, signature: str | None = None) -> dict[str, str]:
    signed: Final = dict(signed_headers(KEY, "msg_9", timestamp, body, "sub_1"))
    return {**signed, "webhook-signature": signature} if signature is not None else signed


def test_a_signed_delivery_verifies() -> None:
    body: Final = b'{"type":"gap"}'
    assert verify((KEY,), _headers(body), body, NOW)


@pytest.mark.parametrize(
    ("headers_body", "presented_body", "now"),
    (
        (b"original", b"tampered", NOW),
        (b"same", b"same", NOW + 301),
        (b"same", b"same", NOW - 301),
    ),
)
def test_tampered_or_stale_deliveries_fail(headers_body: bytes, presented_body: bytes, now: int) -> None:
    assert not verify((KEY,), _headers(headers_body), presented_body, now)


def test_any_of_several_space_delimited_signatures_verifies() -> None:
    body: Final = b"{}"
    good: Final = sign(KEY, "msg_9", NOW, body)
    assert verify((KEY,), _headers(body, signature=f"v1,bogus {good}"), body, NOW)
    assert not verify((bytes(32),), _headers(body), body, NOW)


@pytest.mark.parametrize("missing", ("webhook-id", "webhook-timestamp", "webhook-signature"))
def test_a_delivery_missing_a_required_header_fails(missing: str) -> None:
    body: Final = b"{}"
    headers: Final = {key: value for key, value in _headers(body).items() if key != missing}
    assert not verify((KEY,), headers, body, NOW)


@pytest.mark.parametrize(
    "headers",
    (
        {"webhook-id": "msg_1", "webhook-timestamp": "²³", "webhook-signature": "v1,AAAA"},
        {"webhook-id": "msg_1", "webhook-timestamp": "9" * 400, "webhook-signature": "v1,AAAA"},
        {"webhook-id": "msg_1", "webhook-timestamp": "1800000000", "webhook-signature": "v1,é"},
    ),
)
def test_malformed_signature_headers_are_rejected_not_raised(headers: dict[str, str]) -> None:
    assert verify((b"k" * 32,), headers, b"{}", 1_800_000_000) is False
