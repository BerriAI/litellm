"""Live e2e coverage for virtual-key TPM enforcement."""

from __future__ import annotations

import time

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from models import KeyGenerateBody
from rate_limiting_client import RateLimitingClient

pytestmark = pytest.mark.e2e


RATE_LIMIT_MARKERS = ("rate limit", "tpm limit", "current tpm")


def _generate_tpm_key(
    client: RateLimitingClient, resources: ResourceManager, tpm_limit: int
) -> str:
    key = client.gateway.generate_key(
        KeyGenerateBody(
            models=["gemini-2.5-flash"],
            key_alias=f"e2e-rate-limit-{unique_marker()}",
            tpm_limit=tpm_limit,
        )
    )
    resources.defer(lambda: client.gateway.delete_key(key))
    return key


def _assert_tpm_limit_persisted(
    client: RateLimitingClient, key: str, tpm_limit: int
) -> None:
    info = client.gateway.key_info(key)
    assert (
        info.tpm_limit == tpm_limit
    ), f"/key/info reports tpm_limit {info.tpm_limit}, configured {tpm_limit}"


def _chat(client: RateLimitingClient, key: str, marker: str) -> StreamingResponse:
    return client.chat_status(
        key, "gemini-2.5-flash", f"reply with one short word {marker}"
    )


class TestKeyTpmRateLimiting:
    @pytest.mark.covers("rate_limiting.key.tpm.under_limit_allows")
    def test_tpm_key_under_limit_allows_request(
        self, client: RateLimitingClient, resources: ResourceManager
    ) -> None:
        key = _generate_tpm_key(client, resources, tpm_limit=100_000)
        _assert_tpm_limit_persisted(client, key, 100_000)

        outcome = _chat(client, key, unique_marker())

        require_successful_call(outcome)
        assert (
            outcome.call_id is not None
        ), "successful chat response should include x-litellm-call-id for spend-log correlation"

    @pytest.mark.covers("rate_limiting.key.tpm.over_limit_blocks")
    def test_tpm_key_over_limit_blocks_next_request(
        self, client: RateLimitingClient, resources: ResourceManager
    ) -> None:
        key = _generate_tpm_key(client, resources, tpm_limit=1)
        _assert_tpm_limit_persisted(client, key, 1)

        first = _chat(client, key, unique_marker())
        require_successful_call(first)
        assert (
            first.call_id is not None
        ), "first request should return a call id so the TPM-consuming success can be observed"

        rows = client.gateway.poll_logs_for_request_id(
            first.call_id,
            predicate=lambda found: any((row.total_tokens or 0) > 1 for row in found),
        )
        assert any(
            (row.total_tokens or 0) > 1 for row in rows
        ), f"expected first request to record more than the 1 TPM limit, got rows={rows}"

        deadline = time.monotonic() + client.gateway.poll_timeout
        last = StreamingResponse(status_code=-1, body="not attempted")
        while time.monotonic() < deadline:
            last = _chat(client, key, unique_marker())
            if last.status_code == 429:
                break
            time.sleep(client.gateway.poll_interval)

        assert last.status_code == 429, (
            f"request after exhausting a 1 TPM key should be blocked with 429, got "
            f"{last.status_code}: {last.body[:300]}"
        )
        assert any(
            marker in last.body.lower() for marker in RATE_LIMIT_MARKERS
        ), f"429 body should identify TPM/rate limiting, got: {last.body[:300]}"
