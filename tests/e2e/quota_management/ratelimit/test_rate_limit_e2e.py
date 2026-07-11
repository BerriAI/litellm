"""Live e2e: key-level rpm/tpm rate limits on the gateway.

Covers quota_management.ratelimit.*: a key generated with rpm_limit/tpm_limit gets a
429 once the limit is crossed inside one window (blocks_over_limit), serves
again once the window rolls (resets_after_window), and successful responses
report x-ratelimit-* limit/remaining headers so clients can pace
(headers_report_remaining). Each test asserts both halves of the contract: the
recorded state (/key/info echoes the configured limit) and the enforced
behavior (the 429, the recovery, or the headers on live traffic).

The v3 limiter counts a request against the rpm budget at the pre-call hook,
before model routing, so every call that clears auth consumes budget whether or
not it ultimately succeeds. All calls of one test must land inside a single
window (LITELLM_RATE_LIMIT_WINDOW_SIZE, 60s default), which real chat latency
comfortably allows.
"""

from __future__ import annotations

import time

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from models import KeyGenerateBody
from quota_client import QuotaClient

pytestmark = pytest.mark.e2e

MODEL = "claude-haiku-4-5"


def _limited_key(
    client: QuotaClient,
    resources: ResourceManager,
    *,
    rpm_limit: int | None = None,
    tpm_limit: int | None = None,
) -> str:
    key = client.gateway.generate_key(KeyGenerateBody(models=[MODEL], rpm_limit=rpm_limit, tpm_limit=tpm_limit))
    resources.defer(lambda: client.gateway.delete_key(key))
    return key


def _chat(client: QuotaClient, key: str) -> StreamingResponse:
    return client.chat(key, MODEL, f"reply with one word {unique_marker()}")


def _first_ok(client: QuotaClient, key: str) -> StreamingResponse:
    """First successful call on a fresh key, which opens the rate-limit window.
    A fresh key may briefly 401 until the data plane's auth cache picks it up, so
    retry on 401 to a deadline; a 401 never reaches the rate limiter, so only the
    successful call consumes budget. Any other failure is behavior under test and
    fails hard."""
    deadline = time.monotonic() + client.gateway.poll_timeout
    while True:
        outcome = _chat(client, key)
        if outcome.ok:
            return outcome
        if outcome.status_code != 401 or time.monotonic() >= deadline:
            require_successful_call(outcome)
        time.sleep(client.gateway.poll_interval)


def _assert_rate_limited(outcome: StreamingResponse, limit_type: str) -> None:
    assert outcome.status_code == 429, (
        f"expected a 429 {limit_type} rate-limit block, got {outcome.status_code}: {outcome.body[:300]}"
    )
    assert "Rate limit exceeded for api_key" in outcome.body, (
        f"429 body must name the api_key scope, got: {outcome.body[:300]}"
    )
    assert f"Limit type: {limit_type}" in outcome.body, (
        f"429 body must carry 'Limit type: {limit_type}', got: {outcome.body[:300]}"
    )
    retry_after = outcome.headers.get("retry-after")
    assert retry_after is not None and retry_after.isdigit() and int(retry_after) > 0, (
        f"429 must carry a positive integer retry-after header, got {retry_after!r}"
    )


class TestKeyRateLimits:
    @pytest.mark.covers("quota_management.ratelimit.rpm.blocks_over_limit")
    def test_rpm_limit_blocks_over_limit(self, client: QuotaClient, resources: ResourceManager) -> None:
        key = _limited_key(client, resources, rpm_limit=3)
        info = client.gateway.key_info(key)
        assert info.rpm_limit == 3, f"/key/info reports rpm_limit {info.rpm_limit}, configured 3"

        _ = _first_ok(client, key)
        for _ in range(2):
            require_successful_call(_chat(client, key))

        _assert_rate_limited(_chat(client, key), "requests")

    @pytest.mark.covers("quota_management.ratelimit.tpm.blocks_over_limit")
    def test_tpm_limit_blocks_over_limit(self, client: QuotaClient, resources: ResourceManager) -> None:
        key = _limited_key(client, resources, tpm_limit=60)
        info = client.gateway.key_info(key)
        assert info.tpm_limit == 60, f"/key/info reports tpm_limit {info.tpm_limit}, configured 60"

        _ = _first_ok(client, key)
        for _ in range(8):
            outcome = _chat(client, key)
            if outcome.status_code == 429:
                _assert_rate_limited(outcome, "tokens")
                return
            require_successful_call(outcome)
        pytest.fail("tpm_limit=60 was never enforced with a 429 within 8 calls of ~25 tokens each")

    @pytest.mark.covers("quota_management.ratelimit.rpm.resets_after_window")
    def test_rpm_limit_resets_after_window(self, client: QuotaClient, resources: ResourceManager) -> None:
        key = _limited_key(client, resources, rpm_limit=1)

        _ = _first_ok(client, key)
        _assert_rate_limited(_chat(client, key), "requests")

        deadline = time.monotonic() + client.gateway.poll_timeout
        while time.monotonic() < deadline:
            outcome = _chat(client, key)
            if outcome.ok:
                return
            assert outcome.status_code == 429, (
                f"while the window drains only 429s are acceptable, got {outcome.status_code}: {outcome.body[:300]}"
            )
            time.sleep(client.gateway.poll_interval)
        pytest.fail("a blocked key never recovered after the rate-limit window elapsed")

    @pytest.mark.covers("quota_management.ratelimit.rpm.headers_report_remaining")
    def test_headers_report_limit_and_remaining(self, client: QuotaClient, resources: ResourceManager) -> None:
        key = _limited_key(client, resources, rpm_limit=5, tpm_limit=100000)

        first = _first_ok(client, key)
        assert first.headers.get("x-ratelimit-api_key-limit-requests") == "5", (
            f"success response must report the key's request limit, headers: "
            f"{ {k: v for k, v in first.headers.items() if 'ratelimit' in k} }"
        )
        assert first.headers.get("x-ratelimit-api_key-remaining-requests") == "4", (
            f"first call against rpm_limit=5 must leave 4 remaining, got "
            f"{first.headers.get('x-ratelimit-api_key-remaining-requests')!r}"
        )
        assert first.headers.get("x-ratelimit-api_key-limit-tokens") == "100000", (
            f"success response must report the key's token limit, got "
            f"{first.headers.get('x-ratelimit-api_key-limit-tokens')!r}"
        )
        remaining_tokens = first.headers.get("x-ratelimit-api_key-remaining-tokens")
        assert remaining_tokens is not None and remaining_tokens.isdigit() and int(remaining_tokens) < 100000, (
            f"one call must leave remaining tokens reported and below the limit, got {remaining_tokens!r}"
        )
