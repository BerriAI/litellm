"""Live e2e: cached prompt tokens must not burn TPM budget (LIT-1930).

Customer expectation: after a cacheable prefix is warmed, the remaining TPM
budget decreases by non-cached tokens only. If cached tokens still counted,
remaining would drop by the full prompt size.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import require_successful_call, unwrap
from lifecycle import ResourceManager
from models import (
    CacheControl,
    ChatResponse,
    KeyGenerateBody,
    LiteLLMParamsBody,
    RichMessage,
    TextBlock,
    Usage,
)
from quota_client import QuotaClient

pytestmark = pytest.mark.e2e

# Anthropic prompt caching (host has ANTHROPIC_API_KEY; Bedrock was "Operation not allowed").
ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5-20251001"
# High enough that pre-call reservation of a cacheable prefix still clears.
TPM_LIMIT = 100_000


class CacheChatBody(BaseModel):
    model: str
    messages: list[RichMessage]
    max_tokens: int = 16
    cache: dict[str, bool] = {"no-cache": True}


def _prefix() -> str:
    marker = unique_marker()
    body = " ".join(f"TPM cache paragraph {i} run {marker}." for i in range(600))
    return f"{body}\nEnd {marker}."


def _cached_tokens(usage: Usage | None) -> int:
    if usage is None:
        return 0
    if usage.cache_read_input_tokens:
        return usage.cache_read_input_tokens
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
        return usage.prompt_tokens_details.cached_tokens
    return 0


def _chat_raw(client: QuotaClient, key: str, model: str, prefix: str):
    body = CacheChatBody(
        model=model,
        messages=[
            RichMessage(
                role="system",
                content=[TextBlock(text=prefix, cache_control=CacheControl())],
            ),
            RichMessage(role="user", content=[TextBlock(text="Reply with one word.")]),
        ],
    )
    return client.proxy.transport.send(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=body,
    )


def _chat(client: QuotaClient, key: str, model: str, prefix: str) -> ChatResponse:
    body = CacheChatBody(
        model=model,
        messages=[
            RichMessage(
                role="system",
                content=[TextBlock(text=prefix, cache_control=CacheControl())],
            ),
            RichMessage(role="user", content=[TextBlock(text="Reply with one word.")]),
        ],
    )
    return unwrap(
        client.proxy.transport.post(
            "/chat/completions",
            headers=client.proxy.transport.bearer(key),
            json=body,
            response_type=ChatResponse,
        )
    )


class TestTpmExcludesCachedTokens:
    @pytest.mark.covers(
        "quota_management.ratelimit.tpm.excludes_cached_tokens",
        exercised_on=["chat_completions"],
    )
    def test_cache_hit_reduces_tpm_by_non_cached_only(
        self, client: QuotaClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-tpm-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=ANTHROPIC_MODEL, api_key="os.environ/ANTHROPIC_API_KEY"
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = client.proxy.generate_key(
            KeyGenerateBody(models=[model], tpm_limit=TPM_LIMIT)
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        prefix = _prefix()
        first = _chat(client, key, model, prefix)
        assert first.choices, f"cache prime returned no choices: {first}"
        first_total = (first.usage.total_tokens or 0) if first.usage else 0
        assert first_total > 0, f"prime call must report usage: {first.usage}"

        deadline = time.monotonic() + 45.0
        second_usage: Usage | None = None
        remaining_after: str | None = None
        while time.monotonic() < deadline:
            outcome = _chat_raw(client, key, model, prefix)
            require_successful_call(outcome)
            parsed = ChatResponse.model_validate_json(outcome.body)
            if _cached_tokens(parsed.usage) > 0:
                second_usage = parsed.usage
                remaining_after = outcome.headers.get(
                    "x-ratelimit-api_key-remaining-tokens"
                )
                break
            time.sleep(2.0)

        assert second_usage is not None, "second call never reported cache-read tokens"
        cached = _cached_tokens(second_usage)
        assert cached > 0
        second_total = second_usage.total_tokens or 0
        assert second_total > cached, (
            f"need total > cached so non-cached slice is measurable: {second_usage}"
        )

        assert remaining_after is not None and remaining_after.isdigit(), (
            f"cache-hit response must expose remaining TPM headers, got {remaining_after!r}"
        )
        remaining = int(remaining_after)
        # If cached tokens were counted, remaining would be limit - first - second_total.
        # With exclusion, remaining is closer to limit - first - (second_total - cached).
        counted_full = TPM_LIMIT - first_total - second_total
        counted_excluding_cache = TPM_LIMIT - first_total - (second_total - cached)
        assert remaining > counted_full, (
            f"remaining TPM {remaining} looks like cached tokens still counted "
            f"(would be ~{counted_full} if full second_total={second_total} counted; "
            f"expected closer to ~{counted_excluding_cache} after excluding "
            f"cache_read={cached}; LIT-1930)"
        )
