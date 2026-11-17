"""
Unit tests for TPM rate limit for concurrent requests

Verifies token-reservation pattern:
- Concurrent requests cannot all observe "under limit" before any of them
  has incremented the counter (atomic reservation via
  ``atomic_check_and_increment_by_n``).
- After a successful request, the counter is reconciled to actual usage.
- After a failed request, the full reservation is released.

The reservation path delegates atomicity to ``atomic_check_and_increment_by_n``,
which uses Redis Lua when available and an asyncio-locked in-memory check
otherwise. These tests exercise the in-memory fallback so they run without
Redis.
"""

import json
from datetime import datetime
from typing import Any, Final

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks import parallel_request_limiter_v3
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _AUDIO_BYTES_PER_TOKEN,
    PROJECT_ITPM_DESCRIPTOR_KEY,
    PROJECT_OTPM_DESCRIPTOR_KEY,
    _apply_implicit_output_cap,
    _call_id_from_callback_kwargs,
    _estimate_audio_block_tokens,
    _get_explicit_output_cap,
    _is_embedding_request,
    _request_stash,
    _rerank_input_to_text,
    _responses_input_to_chat_messages,
    _strip_audio_content_blocks,
    estimate_input_and_output_tokens,
    estimate_precise_input_tokens,
    get_output_candidate_count,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.types.llms.openai import (
    InputTokensDetails,
    ResponseAPIUsage,
    ResponsesAPIResponse,
)
from litellm.types.rerank import RerankResponse
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage
from tests._support.rust_parallel_request_limiter import (
    PROXY_MaxParallelRequestsHandler_v3 as RateLimitHandler,
)
from tests._support.rust_parallel_request_limiter import get_request_stash

_PROJECT_MODEL: Final = "gpt-4o-mini"


@pytest.fixture
def rate_limiter():
    cache = DualCache()
    handler = RateLimitHandler(internal_usage_cache=InternalUsageCache(cache))
    return handler, cache


@pytest.fixture(autouse=True)
def _isolated_request_stash():
    token = _request_stash.set(None)
    yield
    _request_stash.reset(token)


def _tokens(handler: RateLimitHandler, key: str, value: str) -> int:
    return int(handler.rate_limits.local((RateLimitScope(key, value).counter("tokens"),))[0] or 0)


def _project_io_tokens(handler: RateLimitHandler, project_id: str) -> tuple[int, int]:
    value: Final = f"{project_id}:{_PROJECT_MODEL}"
    return _tokens(handler, PROJECT_ITPM_DESCRIPTOR_KEY, value), _tokens(handler, PROJECT_OTPM_DESCRIPTOR_KEY, value)


def _chat_data() -> dict[str, Any]:
    return {"model": _PROJECT_MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}


async def _reserve(handler: RateLimitHandler, cache: DualCache, auth: UserAPIKeyAuth) -> int:
    await handler.async_pre_call_hook(auth, cache, _chat_data(), "completion")
    stash: Final = get_request_stash()
    assert stash is not None
    return stash.reserved_tokens


async def _reserve_project_io(handler: RateLimitHandler, cache: DualCache, project_id: str) -> tuple[int, int]:
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token(f"sk-{project_id}"),
        project_id=project_id,
        project_metadata={
            "model_itpm_limit": {_PROJECT_MODEL: 100_000},
            "model_otpm_limit": {_PROJECT_MODEL: 100_000},
        },
    )
    await handler.async_pre_call_hook(auth, cache, _chat_data(), "completion")
    stash: Final = get_request_stash()
    assert stash is not None
    assert stash.itpm_reserved_tokens > 0 and stash.otpm_reserved_tokens > 0
    assert _project_io_tokens(handler, project_id) == (stash.itpm_reserved_tokens, stash.otpm_reserved_tokens)
    return stash.itpm_reserved_tokens, stash.otpm_reserved_tokens


def _project_kwargs(project_id: str, **extra: object) -> dict[str, object]:
    return {
        "model": _PROJECT_MODEL,
        "standard_logging_object": {
            "metadata": {"user_api_key_hash": hash_token(f"sk-{project_id}"), "user_api_key_project_id": project_id}
        },
        **extra,
    }


async def _settle_project_io(
    handler: RateLimitHandler, cache: DualCache, project_id: str, response_obj: object, **extra: object
) -> tuple[tuple[int, int], tuple[int, int]]:
    reserved: Final = await _reserve_project_io(handler, cache, project_id)
    await handler.async_log_success_event(
        kwargs=_project_kwargs(project_id, **extra),
        response_obj=response_obj,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    return reserved, _project_io_tokens(handler, project_id)


def _chat_response(usage: Usage) -> ModelResponse:
    return ModelResponse(
        id="test",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=_PROJECT_MODEL,
        usage=usage,
        choices=[],
    )


@pytest.mark.asyncio
async def test_model_scope_refund_targets_reserved_model(rate_limiter):
    """
    The pre-call reservation is charged against ``data["model"]`` but the
    router later writes ``model_group`` into ``litellm_params.metadata``,
    which can be ``None`` or a different value. Reconciliation MUST refund the
    same model-scoped counter that was incremented; otherwise model-level
    counters (model_per_team / model_per_key / etc.) drift up forever.
    """
    handler, cache = rate_limiter
    api_key: Final = hash_token("sk-test-model-mismatch")
    team_id: Final = "team-abc"
    reserved: Final = await _reserve(
        handler,
        cache,
        UserAPIKeyAuth(api_key=api_key, team_id=team_id, team_metadata={"model_tpm_limit": {_PROJECT_MODEL: 10_000}}),
    )
    model_per_team: Final = f"{team_id}:{_PROJECT_MODEL}"
    assert reserved > 0
    assert _tokens(handler, "model_per_team", model_per_team) == reserved

    await handler.async_log_failure_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": api_key, "user_api_key_team_id": team_id}}
        },
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert _tokens(handler, "model_per_team", model_per_team) == 0


@pytest.mark.asyncio
async def test_org_scope_refund_on_failure(rate_limiter):
    """
    The plain `organization` scope is reserved upfront (it carries
    tokens_per_unit) — so on failure, the full reservation must be released
    against {organization:org_id}:tokens.
    """
    handler, cache = rate_limiter
    api_key: Final = hash_token("sk-org-refund")
    org_id: Final = "org-acme"
    reserved: Final = await _reserve(
        handler, cache, UserAPIKeyAuth(api_key=api_key, org_id=org_id, organization_tpm_limit=10_000)
    )
    assert reserved > 0
    assert _tokens(handler, "organization", org_id) == reserved

    await handler.async_log_failure_event(
        kwargs={"standard_logging_object": {"metadata": {"user_api_key_hash": api_key, "user_api_key_org_id": org_id}}},
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert _tokens(handler, "organization", org_id) == 0


@pytest.mark.asyncio
async def test_org_scope_reconciled_on_success(rate_limiter):
    """On success the org tokens counter must be reconciled to actual usage."""
    handler, cache = rate_limiter
    api_key: Final = hash_token("sk-org-success")
    org_id: Final = "org-acme"
    reserved: Final = await _reserve(
        handler, cache, UserAPIKeyAuth(api_key=api_key, org_id=org_id, organization_tpm_limit=10_000)
    )
    assert reserved > 50

    await handler.async_log_success_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": api_key, "user_api_key_org_id": org_id}},
            "model": _PROJECT_MODEL,
        },
        response_obj=_chat_response(Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50)),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert _tokens(handler, "organization", org_id) == 50


@pytest.mark.asyncio
async def test_reservation_release_idempotent(rate_limiter):
    """
    If both `async_post_call_failure_hook` and `async_log_failure_event` end
    up firing for the same request, only the first refund applies — the
    second sees the released marker and no-ops.
    """
    handler, cache = rate_limiter
    api_key: Final = hash_token("sk-idempotent")
    auth: Final = UserAPIKeyAuth(api_key=api_key, tpm_limit=10_000)
    reserved: Final = await _reserve(handler, cache, auth)
    assert _tokens(handler, "api_key", api_key) == reserved > 0

    await handler.async_post_call_failure_hook(
        request_data=_chat_data(),
        original_exception=Exception("rejected"),
        user_api_key_dict=auth,
    )
    assert _tokens(handler, "api_key", api_key) == 0

    await handler.async_log_failure_event(
        kwargs={"standard_logging_object": {"metadata": {"user_api_key_hash": api_key}}},
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert _tokens(handler, "api_key", api_key) == 0


@pytest.mark.asyncio
async def test_unreserved_scopes_charged_actual_not_delta_on_success(rate_limiter):
    """
    Counter-drift fix: a scope present in metadata but NOT reserved at
    pre-call (no configured TPM limit for it) must be charged the full
    `actual_tokens` on success — never the `delta = actual - reserved`.
    Otherwise that scope's counter goes negative whenever `actual < reserved`.
    """
    handler, cache = rate_limiter
    api_key: Final = hash_token("sk-mixed-scopes")
    team_id: Final = "team-no-tpm-limit"
    reserved: Final = await _reserve(handler, cache, UserAPIKeyAuth(api_key=api_key, team_id=team_id, tpm_limit=10_000))
    assert reserved > 50
    assert _tokens(handler, "team", team_id) == 0

    await handler.async_log_success_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": api_key, "user_api_key_team_id": team_id}},
            "model": _PROJECT_MODEL,
        },
        response_obj=_chat_response(Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50)),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert _tokens(handler, "api_key", api_key) == 50
    assert _tokens(handler, "team", team_id) == 50


@pytest.mark.asyncio
async def test_unreserved_scopes_not_refunded_on_failure(rate_limiter):
    """
    Failure refund must only emit ops against scopes the reservation
    actually charged. Refunding an unreserved scope (which was never
    incremented at pre-call) would drive its counter to -reserved.
    """
    handler, cache = rate_limiter
    api_key: Final = hash_token("sk-mixed-fail")
    team_id: Final = "team-no-tpm"
    reserved: Final = await _reserve(handler, cache, UserAPIKeyAuth(api_key=api_key, team_id=team_id, tpm_limit=10_000))
    assert reserved > 0

    await handler.async_log_failure_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": api_key, "user_api_key_team_id": team_id}}
        },
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert _tokens(handler, "api_key", api_key) == 0
    assert _tokens(handler, "team", team_id) == 0


@pytest.mark.asyncio
async def test_project_itpm_reconciled_on_success_excludes_cached_tokens(rate_limiter):
    """
    On success, ITPM reconciles to billable input tokens (prompt_tokens
    minus cached_tokens) -- not raw prompt_tokens. Cached prompt-read tokens
    are free under Bedrock Mantle and must not count against the ITPM quota,
    even though they still appear in usage/cost logging elsewhere.
    """
    handler, cache = rate_limiter

    _, settled = await _settle_project_io(
        handler,
        cache,
        "proj-mantle",
        _chat_response(
            Usage(
                prompt_tokens=80,
                completion_tokens=40,
                total_tokens=120,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=30),
            )
        ),
    )

    assert settled == (50, 40)


@pytest.mark.asyncio
async def test_project_itpm_otpm_released_on_failure(rate_limiter):
    """On failure, the full ITPM and OTPM reservations must be refunded."""
    handler, cache = rate_limiter
    await _reserve_project_io(handler, cache, "proj-mantle-fail")

    await handler.async_log_failure_event(
        kwargs=_project_kwargs("proj-mantle-fail"),
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert _project_io_tokens(handler, "proj-mantle-fail") == (0, 0)


@pytest.mark.asyncio
async def test_responses_api_usage_reconciles_using_input_output_tokens_fields(rate_limiter):
    """
    ResponseAPIUsage exposes input_tokens/output_tokens, not
    prompt_tokens/completion_tokens; the reservation must still be trued up
    to the much larger actual usage instead of being kept as-is.
    """
    handler, cache = rate_limiter

    _, settled = await _settle_project_io(
        handler,
        cache,
        "proj-responses",
        ResponsesAPIResponse(
            id="resp_test",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(input_tokens=80, output_tokens=400, total_tokens=480),
        ),
    )

    assert settled == (80, 400)


def test_audio_token_estimate_scales_with_payload_size():
    """
    Regression for veria-ai Low finding: audio token reservation was flat
    300 per block regardless of duration. A short clip and a long clip both
    reserved the same amount, letting a caller hide long audio in one block
    to exhaust ITPM quota while reserving almost nothing.

    The estimate must now grow proportionally with the base64 payload size
    (len(b64) * 3 // 4 // _AUDIO_BYTES_PER_TOKEN), floored at
    DEFAULT_AUDIO_TOKEN_ESTIMATE so reference-only blocks and genuinely
    short clips still get a non-trivial reservation.

    To exceed the floor the decoded payload must be > 300 * 1600 = 480 000
    bytes. We synthesise a fake b64-length string of 650 000 chars
    (decoded ≈ 487 500 bytes → 304 tokens) to avoid actually allocating
    and encoding ~480 kB of audio in every test run.
    """
    large_b64 = "A" * 650_000
    very_large_b64 = "A" * 12_900_000
    small_b64 = "A" * 1_000

    large_block = {
        "type": "input_audio",
        "input_audio": {"data": large_b64, "format": "wav"},
    }
    small_block = {
        "type": "input_audio",
        "input_audio": {"data": small_b64, "format": "wav"},
    }
    very_large_block = {
        "type": "input_audio",
        "input_audio": {"data": very_large_b64, "format": "wav"},
    }
    no_data_block = {"type": "input_audio", "input_audio": {"format": "wav"}}

    large_estimate = _estimate_audio_block_tokens(large_block)
    very_large_estimate = _estimate_audio_block_tokens(very_large_block)
    small_estimate = _estimate_audio_block_tokens(small_block)
    no_data_estimate = _estimate_audio_block_tokens(no_data_block)

    assert large_estimate > small_estimate, (
        f"Large payload ({large_estimate}) must reserve more than small payload "
        f"({small_estimate}); flat-rate bug is back"
    )
    assert very_large_estimate == len(very_large_b64) * 3 // 4 // _AUDIO_BYTES_PER_TOKEN
    assert very_large_estimate > 6_000
    assert no_data_estimate >= 300, (
        f"Reference-only block (no data) must use the DEFAULT_AUDIO_TOKEN_ESTIMATE floor; got {no_data_estimate}"
    )
    assert small_estimate >= 300, (
        f"Small payload must be floored at DEFAULT_AUDIO_TOKEN_ESTIMATE=300; got {small_estimate}"
    )


@pytest.mark.asyncio
async def test_itpm_otpm_reservation_is_kept_on_stream_disconnect(rate_limiter):
    handler, cache = rate_limiter
    reserved: Final = await _reserve_project_io(handler, cache, "proj-disconnect")

    await handler.async_release_max_parallel_requests_on_disconnect(
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-proj-disconnect"), project_id="proj-disconnect")
    )

    assert _project_io_tokens(handler, "proj-disconnect") == reserved
    stash: Final = get_request_stash()
    assert stash is not None
    assert stash.reservation_released is False


@pytest.mark.asyncio
async def test_responses_api_multimodal_input_counts_image_content(rate_limiter):
    """
    Regression for a Low-severity veria-ai finding: the Responses API's
    `input` is commonly a list of message/content-block dicts, but
    litellm.token_counter's `text` argument only joins plain string entries
    in a list and silently drops everything else -- so an `input_image`
    block contributed ~0 tokens to the ITPM estimate instead of the real
    image token count. _estimate_precise_input_tokens now converts Responses
    `input` to chat messages first (via the standard
    transform_responses_api_input_to_messages helper) so image content is
    counted the same way a chat completion's image content already is.
    """
    handler, _cache = rate_limiter

    text_only_estimate = estimate_precise_input_tokens(
        handler,
        data={"input": "hi"},
        model="bedrock_mantle/claude-opus",
        call_type="aresponses",
    )

    multimodal_estimate = estimate_precise_input_tokens(
        handler,
        data={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hi"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/some-image.png",
                        },
                    ],
                }
            ],
        },
        model="bedrock_mantle/claude-opus",
        call_type="aresponses",
    )

    assert multimodal_estimate > text_only_estimate + 100, (
        "Responses API input_image content block was not counted; got "
        f"text_only={text_only_estimate}, multimodal={multimodal_estimate}"
    )


def test_strip_audio_content_blocks_passthrough_non_list_messages():
    """Non-list input is returned unchanged (early return on line 2605)."""
    result = _strip_audio_content_blocks("not a list")
    assert result == "not a list"


def test_strip_audio_content_blocks_passthrough_non_dict_message():
    """Non-dict entries in the message list are appended unchanged."""
    messages = ["plain string message"]
    result = _strip_audio_content_blocks(messages)
    assert result == ["plain string message"]


def test_strip_audio_content_blocks_passthrough_non_list_content():
    """Messages with non-list content (e.g. plain string) pass through unchanged."""
    messages = [{"role": "user", "content": "hello"}]
    result = _strip_audio_content_blocks(messages)
    assert result == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_otpm_rejection_releases_stashed_parallel_slot(rate_limiter):
    """
    When OTPM is over limit after a parallel slot was acquired in the same
    pre-call, the rejection must release that slot.
    """
    handler, cache = rate_limiter
    api_key: Final = hash_token("sk-otpm-slot")
    user_api_key_dict: Final = UserAPIKeyAuth(
        api_key=api_key,
        max_parallel_requests=5,
        project_id="proj-slot",
        project_metadata={"model_otpm_limit": {"m": 5}},
    )
    data: Final[dict[str, Any]] = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler.async_pre_call_hook(user_api_key_dict, cache, data, "completion")

    assert getattr(exc_info.value, "status_code", None) == 429
    gauge: Final = RateLimitScope("api_key", api_key).gauge
    assert handler.rate_limits.local((gauge,))[0] in (None, 0, {})


@pytest.mark.asyncio
async def test_itpm_only_status_stored_when_no_prior_rate_limit_response(rate_limiter):
    """
    When only ITPM is configured (no combined TPM/RPM to pre-populate
    the request stash), a successful ITPM reservation must store its status
    there so post-call headers can read it.
    """
    handler, cache = rate_limiter
    user_api_key_dict: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-itpm-only-store"),
        project_id="proj-store",
        project_metadata={"model_itpm_limit": {"m": 100_000}},
    )

    await handler.async_pre_call_hook(
        user_api_key_dict, cache, {"model": "m", "messages": [{"role": "user", "content": "hello"}]}, "completion"
    )

    stash: Final = get_request_stash()
    assert stash is not None
    stored: Final = stash.rate_limit_response
    assert stored is not None, "ITPM status must be stored in litellm_proxy_rate_limit_response"
    assert stored.get("statuses"), "Stored response must contain statuses"


@pytest.mark.asyncio
async def test_resolve_io_token_usage_responses_api_with_cached_tokens(rate_limiter):
    """
    ResponsesAPIResponse whose usage.input_tokens_details.cached_tokens is set
    subtracts the cached portion from billable input.
    """
    handler, cache = rate_limiter

    _, settled = await _settle_project_io(
        handler,
        cache,
        "proj-resp-cached",
        ResponsesAPIResponse(
            id="resp_cached",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_tokens_details=InputTokensDetails(cached_tokens=25),
            ),
        ),
    )

    assert settled == (75, 50)


@pytest.mark.asyncio
async def test_resolve_io_token_usage_dict_format(rate_limiter):
    """
    Dict-shaped usage on a ModelResponse (older SDK versions or raw dicts in
    the usage field) is parsed correctly.
    """
    handler, cache = rate_limiter

    _, settled = await _settle_project_io(
        handler,
        cache,
        "proj-dict-usage",
        ModelResponse.model_construct(
            usage={"prompt_tokens": 80, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 20}}
        ),
    )

    assert settled == (60, 40)


@pytest.mark.asyncio
async def test_resolve_io_token_usage_unknown_type_returns_unresolved(rate_limiter):
    """
    A ModelResponse whose usage attribute is not a Usage, ResponseAPIUsage,
    or dict (e.g. a plain int) keeps the reservation rather than guessing.
    """
    handler, cache = rate_limiter

    reserved, settled = await _settle_project_io(
        handler, cache, "proj-unknown-usage", ModelResponse.model_construct(usage=42)
    )

    assert settled == reserved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("combined_usage", "expected"),
    [
        (None, None),
        (Usage(prompt_tokens=40, completion_tokens=15, total_tokens=55), (40, 15)),
    ],
)
async def test_zero_usage_keeps_reservations_unless_measured_fallback_exists(rate_limiter, combined_usage, expected):
    handler, cache = rate_limiter

    reserved, settled = await _settle_project_io(
        handler,
        cache,
        "proj-zero-usage",
        ModelResponse(usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)),
        **({} if combined_usage is None else {"combined_usage_object": combined_usage}),
    )

    assert settled == (reserved if expected is None else expected)


@pytest.mark.asyncio
async def test_a_responses_reply_reporting_no_tokens_keeps_the_reservations(rate_limiter):
    handler, cache = rate_limiter

    reserved, settled = await _settle_project_io(
        handler,
        cache,
        "proj-responses-zero",
        ResponsesAPIResponse(
            id="resp_zero",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        ),
    )

    assert settled == reserved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (Usage(prompt_tokens=40, completion_tokens=15, total_tokens=55), (40, 15)),
        (Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), None),
    ],
)
async def test_retry_success_charges_released_project_io_reservations(rate_limiter, usage, expected):
    handler, cache = rate_limiter
    reserved: Final = await _reserve_project_io(handler, cache, "proj-retry")
    await handler.async_log_failure_event(
        kwargs=_project_kwargs("proj-retry"),
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert _project_io_tokens(handler, "proj-retry") == (0, 0)

    await handler.async_log_success_event(
        kwargs=_project_kwargs("proj-retry"),
        response_obj=ModelResponse(usage=usage),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert _project_io_tokens(handler, "proj-retry") == (reserved if expected is None else expected)


@pytest.mark.asyncio
async def test_build_io_token_reservation_ops_skips_unresolvable_usage(rate_limiter):
    """
    When response_obj has no parseable usage, settlement keeps the reservation
    as-is rather than zeroing it out on a bad guess.
    """
    handler, cache = rate_limiter

    reserved, settled = await _settle_project_io(handler, cache, "proj-b", object())

    assert settled == reserved


def test_responses_input_is_counted_even_when_messages_is_present(rate_limiter):
    handler, _cache = rate_limiter
    small_estimate = estimate_precise_input_tokens(
        handler,
        data={"input": "short", "messages": [{"role": "user", "content": "ignored"}]},
        model="",
        call_type="aresponses",
    )
    large_estimate = estimate_precise_input_tokens(
        handler,
        data={"input": "large input " * 500, "messages": []},
        model="",
        call_type="aresponses",
    )

    assert large_estimate > small_estimate


@pytest.mark.asyncio
async def test_anthropic_messages_usage_reconciles_split_project_quota(rate_limiter):
    handler, cache = rate_limiter

    _, settled = await _settle_project_io(
        handler,
        cache,
        "proj-anthropic",
        {"usage": {"input_tokens": 100, "output_tokens": 25, "cache_read_input_tokens": 30}},
    )

    assert settled == (70, 25)


def test_rerank_input_estimate_falls_back_to_character_count(
    rate_limiter,
    monkeypatch,
):
    handler, _cache = rate_limiter
    data = {
        "query": "query text",
        "documents": ["first document", "second document"],
    }

    def token_counter(**_kwargs):
        raise ValueError("tokenizer unavailable")

    monkeypatch.setattr("litellm.token_counter", token_counter)
    rerank_text = _rerank_input_to_text(data)

    assert (
        estimate_precise_input_tokens(
            handler,
            data,
            model="custom-rerank-model",
            call_type="rerank",
        )
        == len(rerank_text) // 4
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_obj", "expected"),
    [
        (RerankResponse(meta={"tokens": {"input_tokens": 42, "output_tokens": 3}}), (42, 3)),
        (
            RerankResponse(
                meta={
                    "tokens": {"input_tokens": 0, "output_tokens": 0},
                    "billed_units": {"total_tokens": 57},
                }
            ),
            (57, 0),
        ),
        (
            RerankResponse(
                meta={
                    "tokens": {"input_tokens": 0, "output_tokens": 0},
                    "billed_units": {"total_tokens": 0},
                }
            ),
            None,
        ),
    ],
)
async def test_rerank_usage_reconciles_project_split_token_quota(rate_limiter, response_obj, expected):
    handler, cache = rate_limiter

    reserved, settled = await _settle_project_io(handler, cache, "proj-rerank", response_obj)

    assert settled == (reserved if expected is None else expected)


def test_split_quota_helpers_handle_non_mapping_inputs(rate_limiter):
    handler, _cache = rate_limiter

    assert _call_id_from_callback_kwargs(object()) is None
    assert _is_embedding_request(object(), None) is False
    assert _get_explicit_output_cap(object(), None) is None
    assert get_output_candidate_count(object()) == 1
    assert get_output_candidate_count({"n": 1e309}) == 1
    assert _get_explicit_output_cap({"max_output_tokens": []}, "responses") is None
    assert _apply_implicit_output_cap(object(), 100, "responses") is None
    assert estimate_input_and_output_tokens(object()) == (0, 0)


@pytest.mark.parametrize(
    ("data", "call_type", "expected"),
    [
        ({"max_tokens": "30.0"}, "", 30),
        ({"max_tokens": "not-a-number"}, "", None),
        ({"max_tokens": True}, "", None),
        ({"max_output_tokens": "30.0"}, "responses", 30),
        ({"max_output_tokens": "nan"}, "responses", None),
        ({"generationConfig": {"maxOutputTokens": "12.5"}}, "agenerate_content", 12),
        ({"generationConfig": {"maxOutputTokens": "oops"}}, "agenerate_content", None),
    ],
)
def test_get_explicit_output_cap_tolerates_unparseable_values(rate_limiter, data, call_type, expected):
    """A client-supplied cap the proxy cannot parse must fall back to the
    no-cap output estimate instead of raising ValueError and 500ing the
    request before it ever reaches the provider."""
    handler, _cache = rate_limiter

    assert _get_explicit_output_cap(data, call_type) == expected


def test_split_quota_multimodal_guards_handle_non_mapping_inputs(rate_limiter):
    handler, _cache = rate_limiter

    assert _estimate_audio_block_tokens(object()) == _estimate_audio_block_tokens({})
    assert _responses_input_to_chat_messages(object()) == ()
    assert estimate_precise_input_tokens(handler, object(), model=None) == 0


@pytest.mark.parametrize(
    ("call_type", "data", "expected_text"),
    [
        ("embedding", {"input": "embedding input"}, "embedding input"),
        (
            "embedding",
            {"input": ["first embedding", "second embedding"]},
            ["first embedding", "second embedding"],
        ),
        ("text_completion", {"prompt": "completion prompt"}, "completion prompt"),
    ],
)
def test_precise_input_estimate_selects_endpoint_text(
    rate_limiter,
    monkeypatch,
    call_type,
    data,
    expected_text,
):
    handler, _cache = rate_limiter
    captured = {}

    def token_counter(**kwargs):
        captured.update(kwargs)
        return 7

    monkeypatch.setattr("litellm.token_counter", token_counter)

    assert estimate_precise_input_tokens(handler, data, model="test", call_type=call_type) == 7
    assert captured["messages"] is None
    assert captured["text"] == expected_text


@pytest.mark.asyncio
async def test_streaming_combined_usage_reconciles_project_io_reservations(rate_limiter):
    handler, cache = rate_limiter

    _, settled = await _settle_project_io(
        handler,
        cache,
        "proj-stream",
        {"response": "stream body"},
        combined_usage_object=Usage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
    )

    assert settled == (40, 15)


@pytest.mark.asyncio
async def test_aggregate_only_combined_usage_reconciles_project_io_reservations(rate_limiter):
    handler, cache = rate_limiter

    _, settled = await _settle_project_io(
        handler, cache, "proj-aggregate", object(), combined_usage_object=Usage(total_tokens=55)
    )

    assert settled == (55, 55)


@pytest.mark.asyncio
async def test_raw_split_usage_dict_reconciles_project_io_tokens(rate_limiter):
    handler, cache = rate_limiter

    _, settled = await _settle_project_io(
        handler,
        cache,
        "proj-raw-split",
        {"input_tokens": 30, "output_tokens": 12, "input_tokens_details": {"cached_tokens": 5}},
    )

    assert settled == (25, 12)


@pytest.mark.asyncio
async def test_post_call_success_hook_contains_header_merge_failures(rate_limiter, monkeypatch):
    handler, _cache = rate_limiter
    response = ModelResponse()
    response._hidden_params = {}

    def raise_on_merge(**_kwargs):
        raise RuntimeError("header merge failed")

    monkeypatch.setattr(
        parallel_request_limiter_v3,
        "_merge_ratelimit_statuses_into_additional_headers",
        raise_on_merge,
    )

    await handler.async_post_call_success_hook(
        data={
            "litellm_proxy_rate_limit_response": {
                "overall_code": "OK",
                "statuses": (),
            }
        },
        user_api_key_dict=UserAPIKeyAuth(),
        response=response,
    )


@pytest.mark.asyncio
async def test_the_project_itpm_reservation_counts_the_request_off_the_event_loop(rate_limiter):
    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_or_create_request_stash
    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        uncounted_text,
        warm_tokenizer,
    )

    handler, cache = rate_limiter
    get_or_create_request_stash()
    warm_tokenizer("claude-fable-5")
    data: dict[str, object] = {
        "model": "claude-fable-5",
        "messages": [{"role": "user", "content": uncounted_text()}],
    }
    user_api_key_dict: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-itpm-loop"),
        project_id="proj-loop",
        project_metadata={"model_itpm_limit": {"claude-fable-5": 10_000_000}},
    )

    _, took, lags = await timed_with_loop_lags(
        lambda: handler.async_pre_call_hook(user_api_key_dict, cache, data, "completion")
    )

    stash: Final = get_request_stash()
    assert stash is not None
    assert stash.rate_limit_response is not None
    assert_loop_stayed_free(took, lags)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


_COMPACTION_SUMMARY: Final = "The user planned a Tokyo trip."
_NEXT_TURN: Final = {"type": "message", "role": "user", "content": "And Osaka?"}
_COMPACTED_INPUT: Final = [
    {"type": "message", "role": "user", "content": "Plan a Tokyo trip. " * 200},
    {"type": "message", "role": "assistant", "content": "Day one, Asakusa. " * 200},
    {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": json.dumps({"type": "compaction", "content": _COMPACTION_SUMMARY}),
    },
    _NEXT_TURN,
]
_SUMMARY_INPUT: Final = [{"type": "message", "role": "user", "content": _COMPACTION_SUMMARY}, _NEXT_TURN]


def test_reservation_skips_the_history_a_compaction_item_replaces(rate_limiter):
    handler, _cache = rate_limiter
    compacted_request: Final = {"model": "gpt-4o", "input": _COMPACTED_INPUT, "max_output_tokens": 1}

    assert estimate_input_and_output_tokens(
        data=compacted_request, call_type="aresponses"
    ) == estimate_input_and_output_tokens(
        data={**compacted_request, "input": _COMPACTED_INPUT[2:]}, call_type="aresponses"
    )
    assert estimate_precise_input_tokens(
        handler, compacted_request, model="gpt-4o", call_type="aresponses"
    ) == estimate_precise_input_tokens(
        handler, {**compacted_request, "input": _SUMMARY_INPUT}, model="gpt-4o", call_type="aresponses"
    )
