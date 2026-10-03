"""Live e2e: a model group alias must share its per-key deployment rate-limit
bucket with the model group it resolves to.

Covers quota_management.ratelimit.model_group_alias.shares_bucket: the proxy
config declares `e2e-alias-rl-target` (a cheap Anthropic deployment) with
`default_api_key_rpm_limit: 3` and `router_settings.model_group_alias` mapping
`e2e-alias-rl-alias` -> `e2e-alias-rl-target`. Both spellings must draw on
one per-key rpm bucket, so a key that exhausts the limit on one spelling is
blocked on the other spelling inside the same window; each test in this file
exhausts the budget on one name and asserts the other name 429s.

All calls of one test must land inside a single window
(LITELLM_RATE_LIMIT_WINDOW_SIZE, 60s default), which real chat latency
comfortably allows.
"""

from __future__ import annotations

import time

import pytest
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from quota_client import QuotaClient

pytestmark = pytest.mark.e2e

MODEL_GROUP = "e2e-alias-rl-target"
MODEL_ALIAS = "e2e-alias-rl-alias"
RPM_LIMIT = 3
WINDOW_SECONDS = 60
LAST_CALL_LATENCY_MARGIN_SECONDS = 10


def _chat(client: QuotaClient, key: str, model: str) -> StreamingResponse:
    return client.chat(key, model, f"reply with one word {unique_marker()}")


def _exhaust_rpm(client: QuotaClient, key: str, model: str) -> float:
    """Send RPM_LIMIT successful calls on `model`, opening the rate-limit
    window; returns the send timestamp of the first call as a lower bound on
    the window start. A fresh key may briefly 401 until the data plane's auth
    cache picks it up, so retry on 401 to a deadline; a 401 never reaches the
    rate limiter."""
    deadline = time.monotonic() + client.proxy.poll_timeout
    first_sent_at: float | None = None
    sent = 0
    while sent < RPM_LIMIT:
        if first_sent_at is None:
            first_sent_at = time.monotonic()
        outcome = _chat(client, key, model)
        if outcome.status_code == 401 and time.monotonic() < deadline:
            time.sleep(client.proxy.poll_interval)
            continue
        require_successful_call(outcome)
        sent += 1
    assert first_sent_at is not None
    return first_sent_at


def _assert_blocked_inside_window(
    client: QuotaClient, key: str, model: str, window_opened_at: float
) -> StreamingResponse:
    assert time.monotonic() < window_opened_at + WINDOW_SECONDS - LAST_CALL_LATENCY_MARGIN_SECONDS, (
        f"the {RPM_LIMIT} exhaust calls took too long; the follow-up call could land in the "
        "next window and mask a shared-bucket regression"
    )
    outcome = _chat(client, key, model)
    assert outcome.status_code == 429, (
        f"{model} must share its rpm bucket with the model group/alias that was already "
        f"exhausted, expected a 429 but got {outcome.status_code}: {outcome.body[:300]}"
    )
    return outcome


class TestModelGroupAliasRateLimit:
    @pytest.mark.covers("quota_management.ratelimit.model_group_alias.shares_bucket")
    def test_alias_shares_rpm_bucket_with_model_group(self, client: QuotaClient, scoped_key: str) -> None:
        opened_at = _exhaust_rpm(client, scoped_key, MODEL_GROUP)
        _assert_blocked_inside_window(client, scoped_key, MODEL_ALIAS, opened_at)

    @pytest.mark.covers("quota_management.ratelimit.model_group_alias.shares_bucket")
    def test_model_group_shares_rpm_bucket_with_alias(self, client: QuotaClient, scoped_key: str) -> None:
        opened_at = _exhaust_rpm(client, scoped_key, MODEL_ALIAS)
        _assert_blocked_inside_window(client, scoped_key, MODEL_GROUP, opened_at)
