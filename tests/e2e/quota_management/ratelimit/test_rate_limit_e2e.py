"""Live e2e: key-level rpm/tpm rate limits on the gateway.

Covers quota_management.ratelimit.*: a key generated with rpm_limit/tpm_limit gets a
429 once the limit is crossed inside one window (blocks_over_limit), serves
again once the window rolls and no sooner (resets_after_window), and successful responses
report x-ratelimit-* limit/remaining headers so clients can pace
(headers_report_remaining). Each test asserts both halves of the contract: the
recorded state (/key/info echoes the configured limit) and the enforced
behavior (the 429, the recovery, or the headers on live traffic).

The v3 limiter counts a request against the rpm budget at the pre-call hook,
before model routing, so every call that clears auth consumes budget whether or
not it ultimately succeeds. The tpm budget is reserved pre-call from an estimate
(message chars // 4 + max_tokens) and reconciled to the body's actual
usage.total_tokens after the call, so a block may legitimately fire before the
actual spend crosses the limit; the tpm test asserts the exact contract on both
sides (a 429 only once the blocked call's reservation exceeds the remaining
budget, and no later than the first call after actual spend reaches the limit).
All calls of one test must land inside a single window
(LITELLM_RATE_LIMIT_WINDOW_SIZE, 60s default), which real chat latency
comfortably allows.

The window opens at the pre-call hook of the first counted call, which happens
after that call is sent, so the send timestamp of the winning first call is a
lower bound on the window start. The reset test uses it to reject an early
reset: recovery must not arrive before the full window has elapsed from that
send, less a small tolerance for the limiter's integer-second window
arithmetic.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from models import KeyGenerateBody
from quota_client import QuotaClient

pytestmark = pytest.mark.e2e

MODEL = "claude-haiku-4-5"
TPM_LIMIT = 60
CHAT_MAX_TOKENS = 16
RESERVATION_CHARS_PER_TOKEN = 4
WINDOW_SECONDS = 60
RESET_TOLERANCE_SECONDS = 5
LAST_CALL_LATENCY_MARGIN_SECONDS = 10


@dataclass(frozen=True, slots=True)
class _FirstOk:
    sent_at: float
    response: StreamingResponse


class _ChatUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_tokens: int


class _ChatBodyWithUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    usage: _ChatUsage


def _total_tokens(outcome: StreamingResponse) -> int:
    try:
        return _ChatBodyWithUsage.model_validate_json(outcome.body).usage.total_tokens
    except ValidationError:
        pytest.fail(f"successful chat body must report usage.total_tokens, got: {outcome.body[:300]}")


def _reserved_tokens(content: str) -> int:
    return max(1, len(content) // RESERVATION_CHARS_PER_TOKEN) + CHAT_MAX_TOKENS


def _remaining_from_429(body: str) -> int:
    found = re.search(r"Remaining: (\d+)", body)
    if found is None:
        pytest.fail(f"429 body must report the remaining budget, got: {body[:300]}")
    return int(found.group(1))


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


def _first_ok(client: QuotaClient, key: str) -> _FirstOk:
    """First successful call on a fresh key, which opens the rate-limit window;
    `sent_at` is captured just before the winning send, so the window opened no
    earlier than it. A fresh key may briefly 401 until the data plane's auth
    cache picks it up, so retry on 401 to a deadline; a 401 never reaches the
    rate limiter, so only the successful call consumes budget. Any other failure
    is behavior under test and fails hard."""
    deadline = time.monotonic() + client.gateway.poll_timeout
    while True:
        sent_at = time.monotonic()
        outcome = _chat(client, key)
        if outcome.ok:
            return _FirstOk(sent_at=sent_at, response=outcome)
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
        key = _limited_key(client, resources, tpm_limit=TPM_LIMIT)
        info = client.gateway.key_info(key)
        assert info.tpm_limit == TPM_LIMIT, f"/key/info reports tpm_limit {info.tpm_limit}, configured {TPM_LIMIT}"

        first = _first_ok(client, key)
        window_deadline = first.sent_at + WINDOW_SECONDS - LAST_CALL_LATENCY_MARGIN_SECONDS
        spent = _total_tokens(first.response)

        while spent < TPM_LIMIT:
            assert time.monotonic() < window_deadline, (
                f"spent only {spent} of {TPM_LIMIT} tokens before the {WINDOW_SECONDS}s window could roll; "
                "the exact-crossing assertion needs every call inside one window"
            )
            content = f"reply with one word {unique_marker()}"
            outcome = client.chat(key, MODEL, content, max_tokens=CHAT_MAX_TOKENS)
            if outcome.status_code == 429:
                _assert_rate_limited(outcome, "tokens")
                remaining = _remaining_from_429(outcome.body)
                reserved = _reserved_tokens(content)
                assert reserved > remaining, (
                    f"blocked while the call still fit: {remaining} of {TPM_LIMIT} tokens remained but the call "
                    f"reserved only {reserved} ({spent} actual tokens spent so far)"
                )
                return
            require_successful_call(outcome)
            spent += _total_tokens(outcome)

        _assert_rate_limited(_chat(client, key), "tokens")

    @pytest.mark.covers("quota_management.ratelimit.rpm.resets_after_window")
    def test_rpm_limit_resets_after_window(self, client: QuotaClient, resources: ResourceManager) -> None:
        key = _limited_key(client, resources, rpm_limit=1)

        first = _first_ok(client, key)
        _assert_rate_limited(_chat(client, key), "requests")

        deadline = time.monotonic() + client.gateway.poll_timeout
        while time.monotonic() < deadline:
            attempt_sent_at = time.monotonic()
            outcome = _chat(client, key)
            if outcome.ok:
                window_age = attempt_sent_at - first.sent_at
                assert window_age >= WINDOW_SECONDS - RESET_TOLERANCE_SECONDS, (
                    f"the key recovered {window_age:.1f}s after the window opened, before the "
                    f"{WINDOW_SECONDS}s window (less {RESET_TOLERANCE_SECONDS}s tolerance) elapsed; "
                    "the limiter reset early instead of after the window"
                )
                return
            assert outcome.status_code == 429, (
                f"while the window drains only 429s are acceptable, got {outcome.status_code}: {outcome.body[:300]}"
            )
            time.sleep(client.gateway.poll_interval)
        pytest.fail("a blocked key never recovered after the rate-limit window elapsed")

    @pytest.mark.covers("quota_management.ratelimit.rpm.headers_report_remaining")
    def test_headers_report_limit_and_remaining(self, client: QuotaClient, resources: ResourceManager) -> None:
        key = _limited_key(client, resources, rpm_limit=5, tpm_limit=100000)

        first = _first_ok(client, key).response
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
