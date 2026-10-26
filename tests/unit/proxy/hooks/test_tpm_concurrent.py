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

import asyncio
import base64
import io
import wave
from functools import partial
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Final, Literal

import pytest
from fastapi import HTTPException
from pydantic import TypeAdapter

from litellm import Router, token_counter
from litellm.caching.caching import DualCache
from litellm.constants import DEFAULT_IMAGE_TOKEN_COUNT
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROJECT_ITPM_DESCRIPTOR_KEY,
    PROJECT_OTPM_DESCRIPTOR_KEY,
    RateLimitedModel,
    _resolve_media_models,
    PROXY_MaxParallelRequestsHandler_v3 as RateLimitHandler,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _call_id_from_callback_kwargs,
    _request_stash,
    get_or_create_request_stash,
    get_request_stash,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.responses.litellm_completion_transformation.transformation import (
    TOOL_CALLS_CACHE,
    LiteLLMCompletionResponsesConfig,
)
from litellm.router_utils.ptu_shares import PTUTeamCeiling
from litellm.types.llms.openai import (
    InputTokensDetails,
    ResponseAPIUsage,
    ResponsesAPIResponse,
)
from litellm.types.rerank import RerankResponse
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage


@pytest.fixture
def rate_limiter():
    cache = DualCache()
    handler = RateLimitHandler(internal_usage_cache=InternalUsageCache(cache))
    return handler, cache


@pytest.mark.parametrize("output_cap,configured_output", [(None, None), (None, 32), (0, 32), (16, None)])
def test_media_only_input_keeps_the_existing_output_budget(
    rate_limiter: tuple[RateLimitHandler, DualCache], output_cap: int | None, configured_output: int | None,
) -> None:
    handler, _ = rate_limiter
    data: Final = {
        "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "file", "file_id": "opaque"}}]}],
        "n": 2, **({"max_tokens": output_cap} if output_cap is not None else {}),
    }
    input_tokens, output_tokens = handler._estimate_input_and_output_tokens(
        data, call_type="acompletion", configured_output_tokens=configured_output,
    )
    expected_output: Final = output_cap if output_cap is not None else configured_output or handler.no_max_tokens_output_floor(None)
    assert input_tokens == DEFAULT_IMAGE_TOKEN_COUNT
    assert output_tokens == expected_output * 2


def _audio_message(seconds: int, sample_rate: int, channels: int) -> dict[str, object]:
    output: Final = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * seconds * sample_rate * channels)
    return {"role": "user", "content": [{
        "type": "input_audio", "input_audio": {
            "data": base64.b64encode(output.getvalue()).decode(), "format": "wav",
        },
    }]}


@pytest.mark.asyncio
@pytest.mark.parametrize("detail,admitted", [("low", True), ("high", False)])
async def test_model_specific_images_resolve_routes_before_admission(
    detail: str, admitted: bool, equivalent_image_urls: tuple[str, str],
) -> None:
    router: Final = Router(
        model_list=[{
            "model_name": "vision", "litellm_params": {
                "model": "azure/custom-deployment", "base_model": "gpt-4o-mini",
                "api_key": "test", "api_base": "https://example.invalid", "api_version": "2024-12-01-preview",
            }, "model_info": {"base_model": "gpt-6-astra"},
        }], model_group_alias={"alias": "vision"},
    )
    cache: Final = DualCache()
    handler: Final = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(cache), media_model_resolver=partial(_resolve_media_models, router),
    )
    auth: Final = UserAPIKeyAuth(api_key=hash_token("sk-image-policy"), tpm_limit=500)
    data: Final = {"model": "alias", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": equivalent_image_urls[1], "detail": detail}},
    ]}], "max_tokens": 16}
    if not admitted:
        with pytest.raises(ProxyRateLimitError):
            await handler.async_pre_call_hook(auth, cache, data, "acompletion")
        return
    await handler.async_pre_call_hook(auth, cache, data, "acompletion")
    stash: Final = get_request_stash()
    assert stash is not None
    assert 250 < stash.reserved_tokens < 500


@pytest.fixture(scope="module")
def equivalent_image_urls() -> tuple[str, str]:
    from PIL import Image

    def encoded_image(compression: int) -> str:
        output: Final = io.BytesIO()
        Image.new("RGB", (1024, 1024)).save(output, format="PNG", compress_level=compression)
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()

    return encoded_image(0), encoded_image(9)


def _structured_image_request(
    image_url: str, representation: str, detail: str = "high"
) -> tuple[str, dict[str, object]]:
    image: Final = {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
    text: Final = {"type": "text", "text": "Describe this image"}
    base: Final = {"model": "gpt-6-astra"}
    if representation in ("image_generation", "code_interpreter"):
        native: Final = (
            {"type": "image_generation_call", "result": image_url.partition(",")[2]}
            if representation == "image_generation"
            else {"type": "code_interpreter_call", "code": "render()", "container_id": "container_1", "outputs": [
                {"type": "logs", "logs": text["text"]}, {"type": "image", "url": image_url},
            ]}
        )
        return "aresponses", {**base, "input": [{**native, "id": "native_1", "status": "completed"}], "max_output_tokens": 16}
    if representation == "chat":
        return "acompletion", {**base, "messages": [{"role": "user", "content": [text, image]}], "max_tokens": 16}
    if representation == "chat_input_image":
        input_image: Final = {**image, "type": "input_image"}
        return "acompletion", {**base, "messages": [{"role": "user", "content": [text, input_image]}], "max_tokens": 16}
    if representation == "anthropic_tool":
        source_image: Final = {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": image_url.partition(",")[2],
        }}
        return "anthropic_messages", {**base, "messages": [{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "call_image", "content": [text, source_image],
        }]}], "max_tokens": 16}
    if representation == "document":
        return "anthropic_messages", {**base, "messages": [{"role": "user", "content": [{
            "type": "document", "source": {"type": "content", "content": [text, image]},
        }]}], "max_tokens": 16}
    response_content: Final = [
        {"type": "input_text", "text": "Describe this image"},
        {"type": "input_image", "image_url": image_url, "detail": detail},
    ]
    response_input: Final = (
        [{"type": "custom_tool_call_output", "call_id": "call_image", "output": response_content}]
        if representation == "custom_output"
        else [{"role": "user", "content": response_content}]
    )
    return "aresponses", {**base, "input": response_input, "max_output_tokens": 16}


@pytest.mark.parametrize("representation", [
    "chat", "chat_input_image", "anthropic_tool", "document", "responses", "custom_output",
    "image_generation", "code_interpreter",
])
def test_structured_media_estimates_ignore_transport_size_in_every_quota(
    rate_limiter: tuple[RateLimitHandler, DualCache], equivalent_image_urls: tuple[str, str], representation: str,
) -> None:
    handler, _ = rate_limiter
    uncompressed, compressed = equivalent_image_urls
    assert len(uncompressed) > 100 * len(compressed)
    call_type, first = _structured_image_request(uncompressed, representation)
    _, second = _structured_image_request(compressed, representation)
    original: Final = json.dumps(first)
    first_input: Final = handler._estimate_request_input_tokens(first, call_type=call_type, precise=True)
    second_input: Final = handler._estimate_request_input_tokens(second, call_type=call_type, precise=True)
    assert first_input == second_input
    assert 500 < first_input.cheap_tokens < 5000
    assert first_input.precise_tokens is not None and 500 < first_input.precise_tokens < 5000
    assert handler._estimate_input_and_output_tokens(first, call_type=call_type) == (
        first_input.cheap_tokens, 16,
    )
    assert json.dumps(first) == original


@pytest.mark.parametrize("instructions", [None, "Follow the supplied instructions. " * 20])
def test_text_reservation_agrees_across_chat_and_responses_representations(
    rate_limiter: tuple[RateLimitHandler, DualCache], instructions: str | None,
) -> None:
    handler, _ = rate_limiter
    text: Final = json.dumps({"type": "input_image", "image_url": "data:image/png;base64,opaque text"})
    message: Final = {"role": "user", "content": text}
    system: Final = [{"role": "system", "content": instructions}] if instructions is not None else []
    expected: Final = handler._estimate_input_and_output_tokens(
        {"messages": [*system, message], "max_tokens": 32}, call_type="acompletion"
    )
    requests: Final = tuple(
        {"input": input_value, "max_output_tokens": 32, **({"instructions": instructions} if instructions else {})}
        for input_value in (text, [message], [{"role": "user", "content": [{"type": "input_text", "text": text}]}])
    )
    original: Final = json.dumps(requests)
    assert expected == ((len(text) + len(instructions or "")) // 4, 32)
    for request in requests:
        assert handler._estimate_input_and_output_tokens(request, call_type="aresponses") == expected
    assert json.dumps(requests) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("instructions", [None, "b" * 400])
async def test_responses_instructions_can_exhaust_the_request_token_budget(
    rate_limiter: tuple[RateLimitHandler, DualCache], instructions: str | None,
) -> None:
    handler, cache = rate_limiter
    auth: Final = UserAPIKeyAuth(api_key=hash_token("sk-instructions"), tpm_limit=150)
    data: Final = {
        "model": "gpt-6-astra", "input": "a" * 400, "max_output_tokens": 16,
        **({"instructions": instructions} if instructions is not None else {}),
    }
    if instructions is not None:
        with pytest.raises(ProxyRateLimitError) as error:
            await handler.async_pre_call_hook(auth, cache, data, "aresponses")
        assert error.value.status_code == 429
        return
    await handler.async_pre_call_hook(auth, cache, data, "aresponses")
    stash: Final = get_request_stash()
    assert stash is not None and stash.reserved_tokens == 116


@pytest.mark.parametrize("item_type", ["file_search_call", "mcp_call"])
def test_native_responses_payload_text_grows_reservations_without_changing_the_chat_bridge(
    rate_limiter: tuple[RateLimitHandler, DualCache], item_type: str,
) -> None:
    handler, _ = rate_limiter
    items: Final = tuple(
        {"type": item_type, "id": "native_1", "status": "completed", **(
            {"queries": ["query"], "results": [{
                "file_id": "file_1", "filename": "result.txt", "score": 1.0, "text": "a" * size,
            }]}
            if item_type == "file_search_call"
            else {"name": "inspect", "server_label": "test", "arguments": "{}", "output": json.dumps({
                "type": "input_image", "image_url": "a" * size,
            })}
        )}
        for size in (8, 4008)
    )
    requests: Final = tuple({"model": "gpt-6-astra", "input": [item], "max_output_tokens": 16} for item in items)
    original: Final = json.dumps(requests)
    estimates: Final = tuple(
        handler._estimate_request_input_tokens(request, call_type="aresponses", precise=True) for request in requests
    )
    assert estimates[1].cheap_tokens - estimates[0].cheap_tokens == 1000
    assert estimates[0].precise_tokens is not None and estimates[1].precise_tokens is not None
    assert estimates[1].precise_tokens > estimates[0].precise_tokens > 0
    assert tuple(
        LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=request["input"], responses_api_request=request,
        )
        for request in requests
    ) == ([], [])
    assert json.dumps(requests) == original


@pytest.mark.parametrize("tool_kind", ["function_call", "custom_tool_call"])
def test_responses_tool_arguments_grow_reservations(
    rate_limiter: tuple[RateLimitHandler, DualCache],
    equivalent_image_urls: tuple[str, str],
    tool_kind: str,
) -> None:
    handler, _ = rate_limiter
    call_id: Final = f"call_reservation_{tool_kind}"
    output_request: Final = _structured_image_request(equivalent_image_urls[1], "custom_output")[1]
    tool_output: Final = TypeAdapter(list[dict[str, object]]).validate_python(output_request["input"])[0]
    requests: Final = tuple(
        {
            **output_request,
            "input": [
                {
                    "type": tool_kind, "call_id": call_id, "name": "inspect",
                    **(
                        {"arguments": json.dumps({"query": "a" * size})}
                        if tool_kind == "function_call" else {"input": "a" * size}
                    ),
                },
                {
                    **tool_output,
                    "type": tool_kind + "_output",
                    "call_id": call_id,
                },
            ],
        }
        for size in (8, 8008)
    )
    estimates: Final = tuple(
        handler._estimate_request_input_tokens(data, call_type="aresponses", precise=True) for data in requests
    )
    assert estimates[1].cheap_tokens - estimates[0].cheap_tokens == 2000
    long_input: Final = TypeAdapter(list[dict[str, object]]).validate_python(requests[1]["input"])
    TOOL_CALLS_CACHE.set_cache(key=call_id, value={
        "id": call_id, "type": "function", "function": {
            "name": "inspect", "arguments": (
                long_input[0]["arguments"] if tool_kind == "function_call"
                else json.dumps({"content": long_input[0]["input"]})
            ),
        },
    })
    try:
        assert handler._estimate_request_input_tokens(
            {**requests[1], "input": [long_input[1]]}, call_type="aresponses", precise=True,
        ) == estimates[1]
    finally:
        TOOL_CALLS_CACHE.delete_cache(key=call_id)

    chat_estimates: Final = tuple(
        handler._estimate_request_input_tokens(
            {"model": "gpt-6-astra", "messages": handler._responses_input_to_chat_messages(data)},
            call_type="acompletion",
        ).cheap_tokens
        for data in requests
    )
    assert chat_estimates[0] == chat_estimates[1]
    with_tools: Final = {
        **requests[1], "tools": [{"type": "function", "name": "inspect", "description": "d" * 12000}]
    }
    assert (
        handler._estimate_request_input_tokens(with_tools, call_type="aresponses").cheap_tokens
        == estimates[1].cheap_tokens
    )

    unsupported_content: Final = {
        **requests[1], "input": [
            *TypeAdapter(list[dict[str, object]]).validate_python(requests[1]["input"]),
            {"role": "user", "content": [
                {"type": "guarded_text", "text": ""},
                {"type": "input_file", "filename": "opaque.pdf", "file_data": "data:application/pdf;base64,eA=="},
            ]},
        ],
    }
    with pytest.raises(ValueError, match="guarded_text"):
        token_counter(
            model="gpt-6-astra", messages=handler._responses_input_to_chat_messages(unsupported_content),
            use_default_image_token_count=True,
        )
    fallback: Final = handler._estimate_request_input_tokens(unsupported_content, call_type="aresponses", precise=True)
    assert fallback.precise_tokens == fallback.cheap_tokens == estimates[1].cheap_tokens + DEFAULT_IMAGE_TOKEN_COUNT


@pytest.mark.parametrize("representation", ["chat", "responses", "anthropic_tool", "document_content"])
def test_opaque_documents_keep_a_cheap_reservation_and_existing_precise_count(
    rate_limiter: tuple[RateLimitHandler, DualCache], representation: str,
) -> None:
    from pypdf import PdfWriter

    handler, _ = rate_limiter
    writer: Final = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    output: Final = io.BytesIO()
    writer.write(output)
    encoded: Final = base64.b64encode(output.getvalue()).decode()
    file: Final = {"type": "file", "file": {"filename": "document.pdf", "file_data": "data:application/pdf;base64," + encoded}}
    document: Final = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": encoded}}
    content: Final = (
        [{"type": "tool_result", "tool_use_id": "read", "content": [document, document]}]
        if representation == "anthropic_tool"
        else [{"type": "document", "source": {"type": "content", "content": [file, file]}}]
        if representation == "document_content"
        else [file, file]
    )
    request: Final = (
        {"model": "gpt-6-astra", "input": [{"role": "user", "content": [
            {"type": "input_file", **file["file"]}, {"type": "input_file", **file["file"]},
        ]}], "max_output_tokens": 16}
        if representation == "responses"
        else {"model": "gpt-6-astra", "messages": [{"role": "user", "content": content}], "max_tokens": 16}
    )
    call_type: Final = "aresponses" if representation == "responses" else "acompletion"
    original: Final = json.dumps(request)
    estimate: Final = handler._estimate_request_input_tokens(request, call_type=call_type, precise=True)
    assert estimate.cheap_tokens == 2 * DEFAULT_IMAGE_TOKEN_COUNT
    assert handler._estimate_input_and_output_tokens(request, call_type=call_type) == (estimate.cheap_tokens, 16)
    messages: Final = handler._responses_input_to_chat_messages(request) if representation == "responses" else request["messages"]
    assert estimate.precise_tokens == token_counter(
        model="gpt-6-astra", messages=messages, use_default_image_token_count=True,
    )
    assert estimate.precise_tokens > estimate.cheap_tokens
    assert json.dumps(request) == original
    for _ in range(18):
        writer.add_blank_page(width=612, height=792)
    large_output: Final = io.BytesIO()
    writer.write(large_output)
    large_encoded: Final = base64.b64encode(large_output.getvalue()).decode()
    large_request: Final = TypeAdapter(dict[str, object]).validate_json(original.replace(encoded, large_encoded))
    large_estimate: Final = handler._estimate_request_input_tokens(large_request, call_type=call_type)
    assert large_estimate.cheap_tokens > estimate.cheap_tokens
    assert large_estimate.cheap_tokens >= len(large_encoded) // 2


@pytest.mark.asyncio
@pytest.mark.parametrize("quota", ["combined", "project", "ptu"])
@pytest.mark.parametrize("detail,limit,admitted", [("low", 500, True), ("high", 500, False), ("high", 2000, True)])
async def test_image_detail_controls_each_admission_path(
    equivalent_image_urls: tuple[str, str], quota: str, detail: str, limit: int, admitted: bool,
) -> None:
    def ceiling(_team: str, model: str) -> PTUTeamCeiling | None:
        return PTUTeamCeiling(model_group=model, tpm_limit=limit, output_to_input_ratio=4.0, cached_input_ratio=0.0) if quota == "ptu" else None

    cache: Final = DualCache()
    handler: Final = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(cache), ptu_team_ceiling_resolver=ceiling,
    )
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-detail-quota"), team_id="media-team",
        tpm_limit=limit if quota == "combined" else None, project_id="media-project",
        project_metadata={"model_itpm_limit": {"gpt-6-astra": limit}} if quota == "project" else {},
    )
    call_type, data = _structured_image_request(equivalent_image_urls[1], "chat", detail)
    if admitted:
        await handler.async_pre_call_hook(auth, cache, data, call_type)
        stash: Final = get_request_stash()
        assert stash is not None
        assert max(stash.reserved_tokens, stash.itpm_reserved_tokens, stash.ptu_reserved_tokens) > 250
        return
    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(auth, cache, data, call_type)
    assert error.value.status_code == (400 if quota == "ptu" else 429)


def test_precise_tokenizer_fallback_reuses_nested_media_and_visible_text_once(
    rate_limiter: tuple[RateLimitHandler, DualCache], equivalent_image_urls: tuple[str, str],
) -> None:
    handler, _ = rate_limiter
    call_type, request = _structured_image_request(equivalent_image_urls[1], "anthropic_tool")
    data: Final = {**request, "messages": [
        *TypeAdapter(list[object]).validate_python(request["messages"]),
        _audio_message(31, 48000, 2),
        {"role": "user", "content": [{"type": "unsupported_counter_block"}]},
    ]}
    measured: Final = handler._estimate_request_input_tokens(data, call_type=call_type, precise=True)
    image_only: Final = handler._estimate_request_input_tokens(request, call_type=call_type)
    assert measured.precise_tokens == measured.cheap_tokens == image_only.cheap_tokens + 310


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "partial", "project_rejection"])
async def test_mixed_media_reservations_settle_each_bucket_in_its_own_units(
    equivalent_image_urls: tuple[str, str], outcome: Literal["success", "failure", "partial", "project_rejection"],
) -> None:
    now: Final = datetime(2026, 1, 2, 3, 4, 5)

    def ceiling(_team: str, model: str) -> PTUTeamCeiling:
        return PTUTeamCeiling(model_group=model, tpm_limit=10000, output_to_input_ratio=4.0, cached_input_ratio=0.0)

    cache: Final = DualCache()
    handler: Final = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(cache), time_provider=lambda: now, ptu_team_ceiling_resolver=ceiling,
    )
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-media-settlement"), tpm_limit=10000, team_id="media-team", project_id="media-project",
        project_metadata={
            "model_itpm_limit": {"gpt-6-astra": 500 if outcome == "project_rejection" else 2000},
            "model_otpm_limit": {"gpt-6-astra": 10000},
        },
    )
    _, image_request = _structured_image_request(equivalent_image_urls[1], "chat")
    data: Final = {**image_request, "messages": [*TypeAdapter(list[object]).validate_python(image_request["messages"]), _audio_message(31, 24000, 1)]}
    scopes: Final = (
        ("api_key", auth.api_key),
        ("model_per_team_ptu", "media-team:gpt-6-astra"),
        (PROJECT_ITPM_DESCRIPTOR_KEY, "media-project:gpt-6-astra"),
        (PROJECT_OTPM_DESCRIPTOR_KEY, "media-project:gpt-6-astra"),
    )
    if outcome == "project_rejection":
        with pytest.raises(ProxyRateLimitError):
            await handler.async_pre_call_hook(auth, cache, data, "acompletion")
    else:
        await handler.async_pre_call_hook(auth, cache, data, "acompletion")
        stash: Final = get_request_stash()
        assert stash is not None
        assert stash.reserved_tokens > 1500
        assert stash.itpm_reserved_tokens > 1500
        assert stash.ptu_reserved_tokens == stash.reserved_tokens + 3 * stash.otpm_reserved_tokens
        usage: Final = Usage(
            prompt_tokens=40, completion_tokens=10, total_tokens=50,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=15),
        )
        kwargs: Final = {
            "litellm_call_id": data.get("litellm_call_id"),
            "model": "gpt-6-astra",
            "standard_logging_object": {"metadata": {
                "user_api_key_hash": auth.api_key, "user_api_key_team_id": auth.team_id,
                "user_api_key_project_id": auth.project_id,
            }},
            **({"combined_usage_object": usage} if outcome == "partial" else {}),
        }
        if outcome == "success":
            response: Final = ModelResponse(model="gpt-6-astra", usage=usage, choices=[])
            await handler.async_log_success_event(kwargs, response, now, now)
        else:
            await handler.async_log_failure_event(kwargs, None, now, now)
            await handler.async_log_failure_event(kwargs, None, now, now)
    expected: Final = (35, 65, 25, 10) if outcome in ("success", "partial") else (0, 0, 0, 0)
    for (scope, value), amount in zip(scopes, expected):
        key: Final = handler.create_rate_limit_keys(scope, value or "", "tokens")
        assert int(await cache.async_get_cache(key=key, local_only=True) or 0) == amount


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", ["chat", "inlineData"])
@pytest.mark.parametrize("seconds,admitted", [(1, True), (31, False)])
async def test_native_and_chat_audio_share_duration_based_quota_admission(
    rate_limiter: tuple[RateLimitHandler, DualCache], representation: str, seconds: int, admitted: bool,
) -> None:
    handler, cache = rate_limiter
    model: Final = "gemini/gemini-3.8-flash"
    message: Final = _audio_message(seconds, 24000, 1)
    blocks: Final = TypeAdapter(list[dict[str, object]]).validate_python(message["content"])
    audio: Final = TypeAdapter(dict[str, str]).validate_python(blocks[0]["input_audio"])
    native: Final = representation != "chat"
    data: Final = (
        {"model": model, "contents": [{"role": "user", "parts": [{
            "inlineData": {"mimeType": "audio/wav", "data": audio["data"]},
        }]}], "config": {"maxOutputTokens": 16}}
        if native
        else {"model": model, "messages": [message], "max_tokens": 16}
    )
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-native-duration"), tpm_limit=100, project_id="audio-project",
        project_metadata={"model_itpm_limit": {model: 100}},
    )
    call_type: Final = "agenerate_content" if native else "acompletion"
    if admitted:
        await handler.async_pre_call_hook(auth, cache, data, call_type)
        stash: Final = get_request_stash()
        assert stash is not None
        assert 26 < stash.reserved_tokens < 100
        assert 10 < stash.itpm_reserved_tokens < 100
        return
    with pytest.raises(ProxyRateLimitError):
        await handler.async_pre_call_hook(auth, cache, data, call_type)


@pytest.fixture(autouse=True)
def _isolated_request_stash():
    token = _request_stash.set(None)
    yield
    _request_stash.reset(token)


@pytest.mark.asyncio
async def test_token_reservation_prevents_concurrent_bypass(rate_limiter):
    """
    With a 100 TPM limit and 5 concurrent requests each estimated at ~50+ tokens,
    upfront reservation must reject the late arrivals — not let all 5 through.
    Exercises the in-memory fallback in ``atomic_check_and_increment_by_n``.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-test-key")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        tpm_limit=100,
    )

    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "Hello, this is a test message for concurrent bypass testing.",
            }
        ],
        "max_tokens": 50,
    }

    async def make_request(request_id: int) -> Dict[str, Any]:
        data = request_data.copy()
        try:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="",
            )
            return {
                "request_id": request_id,
                "success": True,
                "reserved_tokens": get_request_stash().reserved_tokens,
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "status_code": getattr(e, "status_code", None),
            }

    tasks = [make_request(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    rate_limited = [r for r in failed if r.get("status_code") == 429]

    assert len(rate_limited) > 0, (
        f"Expected some rate-limited requests but all {len(successful)} succeeded — "
        f"the concurrent bypass bug is still present."
    )


@pytest.mark.asyncio
async def test_no_leak_on_over_limit_rejection(rate_limiter):
    """
    When a reservation would exceed the TPM limit, the counter must NOT be
    bumped. Otherwise rejected requests would silently consume quota with no
    path to refund (the failure callback only fires after the reservation
    was successfully stashed).
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-no-leak"),
        tpm_limit=10,  # tiny limit, easy to blow past
    )
    counter_key = handler.create_rate_limit_keys(
        key="api_key", value=user_api_key_dict.api_key, rate_limit_type="tokens"
    )

    # Reservation will estimate >> 10 tokens, so this should be rejected.
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "x" * 200}],
        "max_tokens": 200,
    }

    estimated = handler._estimate_tokens_for_request(data=data)
    assert estimated > user_api_key_dict.tpm_limit, (
        "Test assumes the reservation amount blows past the limit; "
        f"estimated={estimated}, limit={user_api_key_dict.tpm_limit}"
    )

    with pytest.raises(Exception, match='Limit type: tokens\\. Current limit') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429

    # The reservation bump (estimated_tokens) must NOT have committed. The
    # counter may carry a tiny pre-existing bump from should_rate_limit's
    # per-request +1 sliding-window logic, but it must be far below the
    # reservation amount — proving the all-or-nothing primitive rolled back
    # cleanly on rejection.
    cached_value = await cache.async_get_cache(key=counter_key, local_only=True)
    cached_int = int(cached_value or 0)
    assert cached_int < estimated, (
        f"Reservation leaked: counter={cached_int} after rejection of an "
        f"estimated_tokens={estimated} reservation."
    )


@pytest.mark.asyncio
async def test_token_adjustment_on_success(rate_limiter):
    """
    On success a reserved scope's counter is reconciled to actual via
    `actual - reserved`. With actual=50 and reserved=100, the api_key
    counter should see a -50 delta — and only because api_key was
    reserved against. Unreserved scopes get the full +actual instead.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-test-adjust")

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
            }
        },
        "model": "gpt-3.5-turbo",
    }

    mock_response = ModelResponse(
        id="test",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-3.5-turbo",
        usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        choices=[],
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append(
                {
                    "key": op["key"],
                    "increment": op["increment_value"],
                }
            )

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_adjustments = [i for i in increments if "tokens" in i["key"]]

    assert any(i["increment"] == -50 for i in token_adjustments), (
        f"Expected a -50 token adjustment (50 actual - 100 reserved) but got: "
        f"{token_adjustments}"
    )


@pytest.mark.asyncio
async def test_token_release_on_failure(rate_limiter):
    """On failure the entire reservation must be refunded — but only against
    scopes that were actually charged at pre-call. Unreserved scopes were
    never incremented and must not receive a -reserved op (would drift
    negative)."""
    handler, _cache = rate_limiter

    api_key = hash_token("sk-test-fail")

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append(
                {
                    "key": op["key"],
                    "increment": op["increment_value"],
                }
            )

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_releases = [i for i in increments if "tokens" in i["key"]]

    assert any(
        i["increment"] == -100 for i in token_releases
    ), f"Expected the full reservation (-100) to be released, got: {token_releases}"


@pytest.mark.asyncio
async def test_model_scope_refund_targets_reserved_model(rate_limiter):
    """
    The pre-call reservation is charged against ``data["model"]`` but the
    router later writes ``model_group`` into ``litellm_params.metadata``,
    which can be ``None`` or a different value. Reconciliation MUST refund the
    same model-scoped counter that was incremented; otherwise model-level
    counters (model_per_team / model_per_key / etc.) drift up forever.

    This test makes ``model_group`` absent from kwargs (the failure mode in
    the Greptile P1) and asserts the refund still targets the model the
    reservation used.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-test-model-mismatch")
    team_id = "team-abc"
    reserved_model = "gpt-4o-mini"

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_model = RateLimitedModel(requested=reserved_model, group=reserved_model)
    stash.reserved_scopes = frozenset({("model_per_team", f"{team_id}:{reserved_model}")})

    mock_kwargs = {
        # NOTE: no litellm_params.metadata.model_group — get_model_group_from_litellm_kwargs
        # returns None on this kwargs dict.
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_model_per_team_key = handler.create_rate_limit_keys(
        key="model_per_team",
        value=f"{team_id}:{reserved_model}",
        rate_limit_type="tokens",
    )
    matching = [i for i in increments if i["key"] == expected_model_per_team_key]
    assert matching, (
        f"Expected a refund on the reserved model_per_team counter "
        f"({expected_model_per_team_key}) but got: "
        f"{[i['key'] for i in increments]}"
    )
    assert matching[0]["increment"] == -100, (
        f"Expected full -100 refund on model_per_team counter, got "
        f"{matching[0]['increment']}"
    )


@pytest.mark.asyncio
async def test_should_rate_limit_does_not_inflate_tokens_counter(rate_limiter):
    """
    The pre-call sliding-window check (`should_rate_limit`) must not bump the
    `:tokens` counter. That counter is owned exclusively by the atomic
    `reserve_tpm_tokens` path; double-handling shrinks the effective TPM
    budget by 1 per concurrent in-flight request.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-no-tokens-inflation")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        rpm_limit=100,
        tpm_limit=10_000,
    )

    tokens_counter_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }

    estimated = handler._estimate_tokens_for_request(data=data)

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    cached = int(
        await cache.async_get_cache(key=tokens_counter_key, local_only=True) or 0
    )

    # The :tokens counter should reflect ONLY the reservation amount — not
    # an additional +1 from the should_rate_limit pre-pass.
    assert cached == estimated, (
        f"Expected :tokens counter to equal the reservation ({estimated}) "
        f"with no +1 inflation from should_rate_limit, got {cached}"
    )


@pytest.mark.asyncio
async def test_concurrent_burst_within_tpm_budget_all_succeed(rate_limiter):
    """
    With a TPM limit comfortably above (N concurrent × per-request reservation),
    all N requests must succeed. Pre-fix the should_rate_limit +1-per-key
    inflation could 429 late arrivals on tight budgets.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-burst-budget"),
        tpm_limit=1000,
        rpm_limit=100,
    )

    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "x" * 40}],  # ~10 input tokens
        "max_tokens": 100,
    }

    estimated_per_request = handler._estimate_tokens_for_request(data=request_data)
    n_concurrent = 3
    # Sanity: total reservation must fit within tpm_limit and we want enough
    # headroom that any +1 inflation would NOT push us over.
    assert estimated_per_request * n_concurrent < user_api_key_dict.tpm_limit

    async def make_request(request_id: int):
        data = request_data.copy()
        try:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="",
            )
            return True
        except Exception:
            return False

    results = await asyncio.gather(*[make_request(i) for i in range(n_concurrent)])

    assert all(results), (
        f"All {n_concurrent} requests should fit within tpm_limit="
        f"{user_api_key_dict.tpm_limit} (estimated_per_request="
        f"{estimated_per_request}), but only {sum(results)} succeeded — "
        f"the should_rate_limit :tokens-counter inflation bug is back."
    )


@pytest.mark.asyncio
async def test_org_scope_refund_on_failure(rate_limiter):
    """
    The plain `organization` scope is reserved upfront (it carries
    tokens_per_unit) — so on failure, the full reservation must be released
    against {organization:org_id}:tokens. Pre-fix this scope was missing
    from `_build_tpm_scope_pipeline_operations`, leaking forever.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-org-refund")
    org_id = "org-acme"

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("organization", org_id)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_org_id": org_id,
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_org_key = handler.create_rate_limit_keys(
        key="organization", value=org_id, rate_limit_type="tokens"
    )
    matching = [i for i in increments if i["key"] == expected_org_key]
    assert matching, (
        f"Expected a refund on the org tokens counter ({expected_org_key}) "
        f"but got keys: {[i['key'] for i in increments]}"
    )
    assert (
        matching[0]["increment"] == -100
    ), f"Expected full -100 refund on org counter, got {matching[0]['increment']}"


@pytest.mark.asyncio
async def test_org_scope_reconciled_on_success(rate_limiter):
    """
    On success the org tokens counter must be reconciled to actual usage.
    With reserved=100 and actual=50, the org scope should see a -50 delta.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-org-success")
    org_id = "org-acme"

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("organization", org_id)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_org_id": org_id,
            }
        },
        "model": "gpt-3.5-turbo",
    }

    mock_response = ModelResponse(
        id="test",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-3.5-turbo",
        usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        choices=[],
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_org_key = handler.create_rate_limit_keys(
        key="organization", value=org_id, rate_limit_type="tokens"
    )
    matching = [i for i in increments if i["key"] == expected_org_key]
    assert matching, (
        f"Expected a reconciliation op on the org tokens counter "
        f"({expected_org_key}), got keys: {[i['key'] for i in increments]}"
    )
    assert matching[0]["increment"] == -50, (
        f"Expected -50 delta on org counter (50 actual - 100 reserved), got "
        f"{matching[0]['increment']}"
    )


@pytest.mark.asyncio
async def test_estimate_tokens_uses_max_tokens_when_explicit(rate_limiter):
    """When max_tokens is set explicitly, reservation should equal input + max_tokens."""
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={
            "messages": [
                {"role": "user", "content": "abcd" * 4}
            ],  # 16 chars ~ 4 tokens
            "max_tokens": 25,
        }
    )
    # input ~= 16/4 = 4 tokens; max_tokens = 25; total ~= 29
    assert estimate == 4 + 25


@pytest.mark.asyncio
async def test_estimate_tokens_honors_explicit_zero_max_tokens(rate_limiter):
    """
    Regression for a Greptile finding: explicit_max_tokens was resolved via
    `data.get("max_tokens") or data.get("max_completion_tokens") or
    data.get("max_output_tokens")`, so an explicit 0 in the first field was
    falsy and fell through to the next field (or the no-max_tokens floor),
    silently discarding a caller's explicit zero-output request.
    """
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={
            "messages": [
                {"role": "user", "content": "abcd" * 4}
            ],  # 16 chars ~ 4 tokens
            "max_tokens": 0,
        }
    )
    assert estimate == 4, (
        f"expected input-only reservation (4) for an explicit max_tokens=0, got {estimate}"
    )


@pytest.mark.asyncio
async def test_estimate_tokens_honors_explicit_zero_max_output_tokens_for_responses(
    rate_limiter,
):
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={
            "input": "describe this image in detail",  # 29 chars ~ 7 tokens
            "max_output_tokens": 0,
        },
        min_configured_tpm_limit=40,
        call_type="aresponses",
    )
    assert estimate == 23


@pytest.mark.asyncio
async def test_estimate_tokens_zero_for_empty_embeddings(rate_limiter):
    """Embeddings have no output budget — reservation should equal input only."""
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={"input": "hello world"}  # 11 chars
    )
    # input ~= 11/4 = 2 tokens (max(1, 11//4)); max_tokens = 0
    assert estimate == 2


@pytest.mark.asyncio
async def test_contentless_request_reserves_minimum(rate_limiter):
    """
    A contentless request (no messages/prompt/input — e.g. /responses,
    tool-call continuations) must still hit the atomic counter so concurrent
    contentless requests don't all observe "under limit". Pre-fix the
    `has_estimable_content` short-circuit skipped the reservation entirely
    and post-call reconciliation provided no backpressure.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-contentless")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, tpm_limit=2)

    counter_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )

    # Two contentless requests should consume two slots of the 2-token
    # budget. The third must 429.
    for _ in range(2):
        data = {"model": "gpt-3.5-turbo"}
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
        assert (
            get_request_stash().reserved_tokens == 1
        ), "Contentless request should reserve the floor of 1 token"

    counter_after_two = int(
        await cache.async_get_cache(key=counter_key, local_only=True) or 0
    )
    assert counter_after_two == 2, (
        f"After two contentless requests at the floor, the api_key tokens "
        f"counter should be 2, got {counter_after_two}"
    )

    with pytest.raises(Exception, match='Limit type: tokens\\. Current limit') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429, (
        "Third contentless request must be rate-limited; pre-fix it would "
        "have bypassed the TPM check entirely."
    )


@pytest.mark.asyncio
async def test_atomic_keys_share_hash_tag_per_descriptor(rate_limiter):
    """
    Cluster safety: every key in a single descriptor's Lua payload must
    share a `{key:value}` hash tag so the call lands on a single Redis
    Cluster slot. Otherwise the Lua script raises CROSSSLOT in cluster mode.
    """
    handler, _cache = rate_limiter

    descriptors = [
        {
            "key": "api_key",
            "value": "abc",
            "rate_limit": {
                "requests_per_unit": 10,
                "tokens_per_unit": 100,
                "window_size": 60,
            },
        },
        {
            "key": "user",
            "value": "xyz",
            "rate_limit": {"tokens_per_unit": 200, "window_size": 60},
        },
    ]
    increments = [{"requests": 1, "tokens": 10}, {"tokens": 10}]

    for descriptor, inc in zip(descriptors, increments):
        keys, _args, _meta = handler._build_descriptor_atomic_payload(
            descriptor=descriptor,
            increment_amounts=inc,
        )
        # All keys in a descriptor's payload must share the same {tag}
        # — that's the prefix between the first '{' and '}'.
        tags = {k[: k.index("}") + 1] for k in keys}
        assert len(tags) == 1, (
            f"Descriptor {descriptor['key']}:{descriptor['value']} produced "
            f"keys spanning multiple hash tags: {tags}. Redis Cluster would "
            f"reject this Lua call with CROSSSLOT."
        )
        expected_tag = f"{{{descriptor['key']}:{descriptor['value']}}}"
        assert tags == {expected_tag}, f"Expected hash tag {expected_tag}, got {tags}"


@pytest.mark.asyncio
async def test_reservation_released_on_proxy_rejection(rate_limiter):
    """
    If the request is rejected after the pre-call reservation succeeds but
    before the LLM call (e.g. a downstream guardrail/auth hook raises),
    `async_post_call_failure_hook` must release the reservation. Otherwise
    the tokens leak — `async_log_failure_event` is a litellm completion
    callback and never fires for proxy-side rejections.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-leak-fix")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, tpm_limit=1000)

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )
    reserved = get_request_stash().reserved_tokens
    assert reserved > 0

    counter_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )
    counter_after_reserve = int(
        await cache.async_get_cache(key=counter_key, local_only=True) or 0
    )
    assert counter_after_reserve == reserved

    # Simulate a downstream guardrail rejecting the request.
    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    counter_after_release = int(
        await cache.async_get_cache(key=counter_key, local_only=True) or 0
    )
    assert counter_after_release == 0, (
        f"Reservation leaked: counter={counter_after_release} after "
        f"proxy-level rejection refund (expected 0)."
    )
    assert get_request_stash().reservation_released is True, (
        "Released flag must be set to prevent "
        "async_log_failure_event from double-refunding."
    )


@pytest.mark.asyncio
async def test_reservation_release_idempotent(rate_limiter):
    """
    If both `async_post_call_failure_hook` and `async_log_failure_event` end
    up firing for the same request, only the first refund applies — the
    second sees the released marker and no-ops.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-idempotent")

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    # Both hooks read the same per-request ContextVar stash: the
    # post-call-failure-hook flips reservation_released on it, and the
    # log-failure-event observes the flip.
    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    await handler.async_post_call_failure_hook(
        request_data={},
        original_exception=Exception("rejected"),
        user_api_key_dict=UserAPIKeyAuth(api_key=api_key),
    )

    first_refund_count = len([i for i in increments if "tokens" in i["key"]])
    assert first_refund_count > 0, "First refund should have applied"

    # Now simulate async_log_failure_event firing afterwards. It must see
    # the released flag on the stash and not double-refund.
    await handler.async_log_failure_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": api_key}},
        },
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    second_refund_count = len([i for i in increments if "tokens" in i["key"]])
    assert second_refund_count == first_refund_count, (
        f"Idempotency violated: refund count went from {first_refund_count} "
        f"to {second_refund_count} after second hook fired."
    )


@pytest.mark.asyncio
async def test_unreserved_scopes_charged_actual_not_delta_on_success(rate_limiter):
    """
    Counter-drift fix: a scope present in metadata but NOT reserved at
    pre-call (no configured TPM limit for it) must be charged the full
    `actual_tokens` on success — never the `delta = actual - reserved`.
    Otherwise that scope's counter goes negative whenever `actual < reserved`
    (the common case, since the reservation includes a conservative output
    pad).
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-mixed-scopes")
    team_id = "team-no-tpm-limit"

    # Reservation ONLY hit api_key — team had no TPM limit configured.
    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
            }
        },
        "model": "gpt-3.5-turbo",
    }

    mock_response = ModelResponse(
        id="t",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-3.5-turbo",
        usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        choices=[],
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    api_key_token_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )
    team_token_key = handler.create_rate_limit_keys(
        key="team", value=team_id, rate_limit_type="tokens"
    )

    api_key_ops = [i for i in increments if i["key"] == api_key_token_key]
    team_ops = [i for i in increments if i["key"] == team_token_key]

    assert api_key_ops and api_key_ops[0]["increment"] == -50, (
        f"Reserved api_key scope must reconcile via delta (50-100=-50), "
        f"got {api_key_ops}"
    )
    assert team_ops and team_ops[0]["increment"] == 50, (
        f"Unreserved team scope must be charged full actual (+50), not the "
        f"-50 delta (which would drift its counter negative). Got {team_ops}"
    )


@pytest.mark.asyncio
async def test_unreserved_scopes_not_refunded_on_failure(rate_limiter):
    """
    Failure refund must only emit ops against scopes the reservation
    actually charged. Refunding an unreserved scope (which was never
    incremented at pre-call) would drive its counter to -reserved.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-mixed-fail")
    team_id = "team-no-tpm"

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    team_token_key = handler.create_rate_limit_keys(
        key="team", value=team_id, rate_limit_type="tokens"
    )
    api_key_token_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )

    team_ops = [i for i in increments if i["key"] == team_token_key]
    api_key_ops = [i for i in increments if i["key"] == api_key_token_key]

    assert not team_ops, (
        f"Unreserved team scope must NOT be refunded (would drift negative), "
        f"got {team_ops}"
    )
    assert (
        api_key_ops and api_key_ops[0]["increment"] == -100
    ), f"Reserved api_key scope must be refunded -100, got {api_key_ops}"


@pytest.mark.asyncio
async def test_token_rate_limit_headers_present_in_stored_response(rate_limiter):
    """
    With `skip_tpm_check=True` on the RPM sliding-window pass, token statuses
    only come from `reserve_tpm_tokens`. They must be merged into the stashed
    rate-limit response so the post-call hook can emit
    `x-ratelimit-{key}-remaining-tokens` / `-limit-tokens` headers to the
    client.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-headers")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        rpm_limit=100,
        tpm_limit=10_000,
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 20,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    response = get_request_stash().rate_limit_response
    assert isinstance(
        response, dict
    ), "Expected the stashed rate-limit response to be set after pre-call"

    statuses = response.get("statuses") or []
    token_statuses = [s for s in statuses if s.get("rate_limit_type") == "tokens"]
    request_statuses = [s for s in statuses if s.get("rate_limit_type") == "requests"]

    assert token_statuses, (
        f"Token rate-limit status missing from stored response. Without it, "
        f"x-ratelimit-*-tokens headers never reach the client. Got "
        f"statuses: {[(s.get('descriptor_key'), s.get('rate_limit_type')) for s in statuses]}"
    )
    assert request_statuses, (
        "RPM rate-limit status was clobbered by the TPM merge — both must "
        "coexist in the stored response."
    )

    # The token status carries the limit and a positive remaining budget.
    api_key_tokens = next(
        (s for s in token_statuses if s.get("descriptor_key") == "api_key"),
        None,
    )
    assert api_key_tokens is not None, f"api_key token status absent: {token_statuses}"
    assert api_key_tokens["current_limit"] == 10_000
    assert api_key_tokens["limit_remaining"] >= 0


@pytest.mark.asyncio
async def test_estimate_tokens_floor_caps_at_smallest_configured_tpm(rate_limiter):
    """
    Regression: with a small configured TPM cap and no max_tokens, the
    output-budget floor must be capped at a fraction of that limit so the
    reservation alone can't trip the limit.
    """
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={"messages": [{"role": "user", "content": "hello"}]},
        min_configured_tpm_limit=1000,
    )
    # input ~= 5//4 = 1 token; output floor capped at 1000//4 = 250;
    # total ~= 251 (well under 1000).
    assert (
        estimate <= 1000 // 2
    ), f"With TPM=1000, reservation must stay well under the limit; got {estimate}"
    assert estimate >= 1, "Estimate must be at least the call-site floor of 1"


@pytest.mark.asyncio
async def test_estimate_tokens_floor_unchanged_for_large_tpm(rate_limiter):
    """
    Large TPM budgets must keep the 1024-token floor so a stream of small
    concurrent requests can't collectively bypass the limit.
    """
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={"messages": [{"role": "user", "content": "hello"}]},
        min_configured_tpm_limit=100_000,
    )
    # input ~= 1; output floor = min(1024, 100_000//4=25_000) = 1024;
    # total ~= 1025.
    assert estimate == 1 + 1024


@pytest.mark.asyncio
async def test_estimate_tokens_floor_unchanged_when_kwarg_omitted(rate_limiter):
    """
    Callers that don't pass min_configured_tpm_limit (legacy path, tests that
    stub the estimator) must observe the pre-fix floor.
    """
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert estimate == 1 + 1024


@pytest.mark.asyncio
async def test_small_tpm_cap_admits_no_max_tokens_request(rate_limiter):
    """
    Regression (end-to-end at the hook level): a project-level model_tpm_limit
    of 1000 with a tiny no-max_tokens request must not 429 on the first call.
    Pre-fix the 1024-token floor tripped OVER_LIMIT against the 1000-token cap
    on every request.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-small-tpm")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-small-tpm",
        project_metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 1000},
            "model_rpm_limit": {"gpt-3.5-turbo": 60},
        },
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
    }

    # Must not raise — pre-fix this was a 429.
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    reserved = get_request_stash().reserved_tokens
    assert reserved > 0, "Reservation should have been stashed"
    assert reserved <= 1000 // 2, (
        f"Capped floor must keep the reservation well under the 1000 TPM "
        f"cap; got {reserved}"
    )


@pytest.mark.asyncio
async def test_small_tpm_cap_injects_matching_max_tokens(rate_limiter):
    """
    When a small TPM cap forces the no-max_tokens floor below the baseline,
    the hook must also write data['max_tokens'] = capped_floor so the actual
    model output is bounded by the reservation. Without this cap, concurrent
    no-max_tokens generations can spend past the TPM limit before post-call
    reconciliation runs.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-small-tpm-cap"),
        project_id="proj-small-tpm-cap",
        project_metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 1000},
        },
    )

    data: dict = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    assert data.get("max_tokens") == 1000 // 4, (
        f"Capped floor must be written to max_tokens to bound the actual "
        f"model output; got {data.get('max_tokens')}"
    )


@pytest.mark.asyncio
async def test_large_tpm_cap_does_not_inject_max_tokens(rate_limiter):
    """
    A TPM cap that doesn't constrain the floor must not silently inject
    max_tokens — that would change behaviour for tenants who already have
    plenty of budget.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-large-tpm-cap"),
        project_id="proj-large-tpm-cap",
        project_metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 100_000},
        },
    )

    data: dict = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    assert "max_tokens" not in data, (
        f"Large TPM caps should leave max_tokens alone; got "
        f"{data.get('max_tokens')}"
    )


@pytest.mark.asyncio
async def test_small_tpm_cap_preserves_explicit_max_tokens(rate_limiter):
    """
    Explicit max_tokens from the caller must never be overwritten by the
    bypass mitigation — the user already declared their budget.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-explicit-max-tokens"),
        project_id="proj-explicit-max-tokens",
        project_metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 1000},
        },
    )

    data: dict = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 500,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    assert data["max_tokens"] == 500


@pytest.mark.asyncio
async def test_project_otpm_reservation_prevents_concurrent_bypass(rate_limiter):
    """
    Bedrock Mantle-style OTPM: with a 100 OTPM limit and 5 concurrent
    requests each reserving 50+ output tokens, upfront reservation must
    reject the late arrivals -- not let all 5 through. Exercises the
    in-memory fallback in ``atomic_check_and_increment_by_n`` for the
    project-scoped ITPM/OTPM descriptors specifically.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-bypass"),
        project_id="proj-mantle-bypass",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 100},
        },
    )

    request_data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    async def make_request(request_id: int) -> Dict[str, Any]:
        data = request_data.copy()
        try:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="",
            )
            return {"request_id": request_id, "success": True}
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "status_code": getattr(e, "status_code", None),
            }

    results = await asyncio.gather(*[make_request(i) for i in range(5)])

    successful = [r for r in results if r["success"]]
    rate_limited = [
        r for r in results if not r["success"] and r.get("status_code") == 429
    ]

    assert len(rate_limited) > 0, (
        f"Expected some OTPM-rate-limited requests but all {len(successful)} succeeded."
    )


@pytest.mark.asyncio
async def test_project_otpm_rejects_multiple_completion_candidates(rate_limiter):
    handler, cache = rate_limiter
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-multiple-candidates"),
        project_id="proj-multiple-candidates",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 500},
        },
    )
    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100,
        "n": 10,
    }

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_otpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="acompletion",
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
async def test_project_otpm_reserves_largest_conflicting_output_cap(rate_limiter):
    handler, cache = rate_limiter
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-conflicting-caps"),
        project_id="proj-conflicting-caps",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 50},
        },
    )
    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1,
        "max_completion_tokens": 100,
    }

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_otpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="acompletion",
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["agenerate_content", "agenerate_content_stream"],
)
@pytest.mark.parametrize("config_field", ["config", "generationConfig"])
async def test_project_otpm_rejects_google_genai_native_output_cap(
    rate_limiter,
    call_type,
    config_field,
):
    handler, cache = rate_limiter
    model = "gemini/gemini-3-flash-preview"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-google-genai-native-otpm"),
        project_id="project-google-genai-native-otpm",
        project_metadata={"model_otpm_limit": {model: 50}},
    )

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_otpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={
                "model": model,
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                config_field: {"maxOutputTokens": 100},
            },
            call_type=call_type,
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["agenerate_content", "agenerate_content_stream"],
)
@pytest.mark.parametrize("candidate_count_field", ["candidateCount", "candidate_count"])
async def test_project_otpm_rejects_google_genai_native_candidate_count(
    rate_limiter,
    call_type,
    candidate_count_field,
):
    handler, cache = rate_limiter
    model = "gemini/gemini-3-flash-preview"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-google-genai-native-candidate-count"),
        project_id="project-google-genai-native-candidate-count",
        project_metadata={"model_otpm_limit": {model: 150}},
    )

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_otpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={
                "model": model,
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                "config": {
                    "maxOutputTokens": 50,
                    candidate_count_field: 4,
                },
            },
            call_type=call_type,
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["agenerate_content", "agenerate_content_stream"],
)
@pytest.mark.parametrize("config_field", [None, "config", "generationConfig"])
async def test_project_otpm_injects_google_genai_native_output_cap(
    rate_limiter,
    call_type,
    config_field,
):
    handler, cache = rate_limiter
    model = "gemini/gemini-3-flash-preview"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-google-genai-native-implicit-otpm"),
        project_id="project-google-genai-native-implicit-otpm",
        project_metadata={"model_otpm_limit": {model: 40}},
    )
    data = {
        "model": model,
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    }
    if config_field is not None:
        data[config_field] = {"temperature": 0}

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type=call_type,
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.otpm_reserved_tokens == 10
    expected_config_field = config_field or "config"
    assert data[expected_config_field]["maxOutputTokens"] == 10
    assert "max_tokens" not in data


@pytest.mark.asyncio
async def test_project_otpm_over_limit_rolls_back_itpm_reservation(rate_limiter):
    """
    When ITPM reserves fine but OTPM is then over limit, the ITPM
    reservation this same pre-call already made must be rolled back --
    otherwise it leaks until the window's TTL, silently shrinking the ITPM
    budget for every other request in that minute.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-rollback"),
        project_id="proj-mantle-rollback",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 1000000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 10},
        },
    )

    itpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_itpm",
        value="proj-mantle-rollback:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 500,  # blows past the 10-token OTPM limit
    }

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_otpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429

    cached_value = await cache.async_get_cache(key=itpm_counter_key, local_only=True)
    assert int(cached_value or 0) == 0, (
        f"ITPM reservation leaked after OTPM rejection: counter={cached_value}"
    )


@pytest.mark.asyncio
async def test_project_itpm_reconciled_on_success_excludes_cached_tokens(rate_limiter):
    """
    On success, ITPM reconciles to billable input tokens (prompt_tokens
    minus cached_tokens) -- not raw prompt_tokens. Cached prompt-read tokens
    are free under Bedrock Mantle and must not count against the ITPM quota,
    even though they still appear in usage/cost logging elsewhere.
    """
    handler, _cache = rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-mantle:model")
    otpm_scope = ("model_per_project_otpm", "proj-mantle:model")

    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    mock_kwargs = {}

    mock_response = ModelResponse(
        id="test",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="bedrock_mantle/claude-opus",
        usage=Usage(
            prompt_tokens=80,
            completion_tokens=40,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=30),
        ),
        choices=[],
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_adjustments = [i for i in increments if "model_per_project_itpm" in i["key"]]
    otpm_adjustments = [i for i in increments if "model_per_project_otpm" in i["key"]]

    # billable_input = 80 - 30 cached = 50; delta = 50 - 100 reserved = -50
    assert any(i["increment"] == -50 for i in itpm_adjustments), (
        f"Expected a -50 ITPM adjustment (50 billable - 100 reserved), got: {itpm_adjustments}"
    )
    # delta = 40 actual completion - 60 reserved = -20
    assert any(i["increment"] == -20 for i in otpm_adjustments), (
        f"Expected a -20 OTPM adjustment (40 actual - 60 reserved), got: {otpm_adjustments}"
    )


@pytest.mark.asyncio
async def test_project_reconciliation_does_not_decrement_later_window():
    current_time = datetime(2026, 8, 5, 12, 0, 0)
    cache = DualCache()
    handler = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(cache),
        time_provider=lambda: current_time,
    )
    handler.window_size = 60
    scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    descriptor = {
        "key": scope[0],
        "value": scope[1],
        "rate_limit": {"tokens_per_unit": 1000, "window_size": 60},
    }

    reservation = await handler.atomic_check_and_increment_by_n(
        descriptors=[descriptor],
        increments=[{"tokens": 100}],
    )
    counter_key = handler.create_rate_limit_keys(*scope, rate_limit_type="tokens")
    window_identity = next(
        identity
        for identity in reservation["reservation_windows"]
        if identity[0] == counter_key
    )
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({scope})
    stash.itpm_reserved_window_identities = frozenset(
        {window_identity}
    )

    current_time += timedelta(seconds=61)
    later_reservation = await handler.atomic_check_and_increment_by_n(
        descriptors=[descriptor],
        increments=[{"tokens": 20}],
    )
    assert window_identity not in later_reservation["reservation_windows"]

    await handler.async_log_success_event(
        kwargs={},
        response_obj=ModelResponse(
            usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
        ),
        start_time=current_time,
        end_time=current_time,
    )

    assert float(await cache.async_get_cache(key=counter_key, local_only=True) or 0) == 20


@pytest.mark.asyncio
async def test_project_reconciliation_decrements_its_active_window(rate_limiter):
    handler, cache = rate_limiter
    scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    descriptor = {
        "key": scope[0],
        "value": scope[1],
        "rate_limit": {"tokens_per_unit": 1000, "window_size": 60},
    }
    reservation = await handler.atomic_check_and_increment_by_n(
        descriptors=[descriptor],
        increments=[{"tokens": 100}],
    )
    counter_key = handler.create_rate_limit_keys(*scope, rate_limit_type="tokens")
    window_identity = next(
        identity
        for identity in reservation["reservation_windows"]
        if identity[0] == counter_key
    )
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({scope})
    stash.itpm_reserved_window_identities = frozenset(
        {window_identity}
    )

    await handler.async_log_success_event(
        kwargs={},
        response_obj=ModelResponse(
            usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
        ),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert float(await cache.async_get_cache(key=counter_key, local_only=True) or 0) == 10


@pytest.mark.asyncio
async def test_redis_window_guard_uses_reservation_identity_and_never_falls_back_negative(
    rate_limiter,
):
    handler, _cache = rate_limiter
    calls = []

    async def failing_guard(*, keys, args):
        calls.append((keys, args))
        raise RuntimeError("redis unavailable")

    unguarded_calls = []

    async def capture_unguarded(pipeline_operations, **_kwargs):
        unguarded_calls.extend(pipeline_operations)

    handler.window_guarded_token_increment_script = failing_guard
    handler.async_increment_tokens_with_ttl_preservation = capture_unguarded
    await handler.async_increment_reservation_aware_tokens(
        pipeline_operations=[
            {
                "key": "{model_per_project_itpm:project:model}:tokens",
                "increment_value": -90,
                "ttl": 60,
                "window_key": "{model_per_project_itpm:project:model}:window",
                "expected_window_start": "1234",
                "reservation_backend": "redis",
            }
        ]
    )

    assert calls == [
        (
            [
                "{model_per_project_itpm:project:model}:window",
                "{model_per_project_itpm:project:model}:tokens",
            ],
            ["1234", -90, 60],
        )
    ]
    assert unguarded_calls == []


@pytest.mark.asyncio
async def test_atomic_lua_response_carries_redis_window_identity(rate_limiter):
    handler, _cache = rate_limiter
    counter_key = "{model_per_project_itpm:project:model}:tokens"
    meta = [
        {
            "descriptor_key": PROJECT_ITPM_DESCRIPTOR_KEY,
            "descriptor_value": "project:model",
            "current_limit": 100,
            "rate_limit_type": "tokens",
            "counter_key": counter_key,
        }
    ]

    async def successful_reservation(*, keys, args):
        return [0, 25, 1234]

    handler.check_and_increment_by_n_script = successful_reservation
    assert await handler._atomic_lua_per_descriptor([]) == {
        "overall_code": "OK",
        "statuses": [],
    }

    response = await handler._atomic_lua_per_descriptor(
        descriptor_groups=[
            (
                [
                    "{model_per_project_itpm:project:model}:window",
                    counter_key,
                ],
                [100, 25, 60, 60],
                meta,
            )
        ]
    )

    assert response["statuses"][0]["limit_remaining"] == 75
    assert response["reservation_windows"] == frozenset(
        {(counter_key, "1234", "redis")}
    )


@pytest.mark.asyncio
async def test_project_itpm_otpm_released_on_failure(rate_limiter):
    """On failure, the full ITPM and OTPM reservations must be refunded."""
    handler, _cache = rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-mantle:model")
    otpm_scope = ("model_per_project_otpm", "proj-mantle:model")

    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    mock_kwargs = {}

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_releases = [i for i in increments if "model_per_project_itpm" in i["key"]]
    otpm_releases = [i for i in increments if "model_per_project_otpm" in i["key"]]

    assert any(i["increment"] == -100 for i in itpm_releases), itpm_releases
    assert any(i["increment"] == -60 for i in otpm_releases), otpm_releases


@pytest.mark.asyncio
async def test_proxy_rejection_refunds_itpm_otpm_by_their_own_amount_not_combined(
    rate_limiter,
):
    """
    Regression for a Greptile-flagged bug: when a project configures both a
    combined model_tpm_limit and split model_itpm_limit/model_otpm_limit for
    the same model, async_post_call_failure_hook's proxy-side refund path
    used to decrement every token descriptor -- including the ITPM/OTPM
    ones -- by the flat combined reservation amount, instead of each
    bucket's own reserved amount. That drives the split counters negative
    (or under-refunds them) instead of returning them to exactly zero.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-mixed-tpm-io")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-mixed",
        project_metadata={
            "model_tpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 100000},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [
            {"role": "user", "content": "hello there, this is a test message"}
        ],
        "max_tokens": 60,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    tpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project",
        value="proj-mixed:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )
    itpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_itpm",
        value="proj-mixed:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )
    otpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_otpm",
        value="proj-mixed:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )

    tpm_reserved = int(
        await cache.async_get_cache(key=tpm_counter_key, local_only=True) or 0
    )
    itpm_reserved = int(
        await cache.async_get_cache(key=itpm_counter_key, local_only=True) or 0
    )
    otpm_reserved = int(
        await cache.async_get_cache(key=otpm_counter_key, local_only=True) or 0
    )
    assert tpm_reserved > 0 and itpm_reserved > 0 and otpm_reserved > 0

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    tpm_after = int(
        await cache.async_get_cache(key=tpm_counter_key, local_only=True) or 0
    )
    itpm_after = int(
        await cache.async_get_cache(key=itpm_counter_key, local_only=True) or 0
    )
    otpm_after = int(
        await cache.async_get_cache(key=otpm_counter_key, local_only=True) or 0
    )

    assert tpm_after == 0, f"combined TPM counter leaked: {tpm_after}"
    assert itpm_after == 0, (
        f"ITPM counter corrupted by combined-amount refund: {itpm_after}"
    )
    assert otpm_after == 0, (
        f"OTPM counter corrupted by combined-amount refund: {otpm_after}"
    )


@pytest.mark.asyncio
async def test_proxy_rejection_refunds_itpm_otpm_only_reservation_with_no_combined_tpm(
    rate_limiter,
):
    """
    Regression for the second half of the same bug: with only
    model_itpm_limit/model_otpm_limit configured (no model_tpm_limit), the
    combined reserved_tokens is 0, and the proxy-side refund path used to
    return immediately on that -- leaking the ITPM/OTPM reservations until
    the rate-limit window's TTL expired.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-io-only")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-io-only",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 100000},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [
            {"role": "user", "content": "hello there, this is a test message"}
        ],
        "max_tokens": 60,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    itpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_itpm",
        value="proj-io-only:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )
    otpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_otpm",
        value="proj-io-only:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )
    assert (
        int(await cache.async_get_cache(key=itpm_counter_key, local_only=True) or 0) > 0
    )
    assert (
        int(await cache.async_get_cache(key=otpm_counter_key, local_only=True) or 0) > 0
    )

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    itpm_after = int(
        await cache.async_get_cache(key=itpm_counter_key, local_only=True) or 0
    )
    otpm_after = int(
        await cache.async_get_cache(key=otpm_counter_key, local_only=True) or 0
    )
    assert itpm_after == 0, (
        f"ITPM-only reservation leaked on proxy rejection: {itpm_after}"
    )
    assert otpm_after == 0, (
        f"OTPM-only reservation leaked on proxy rejection: {otpm_after}"
    )


@pytest.mark.asyncio
async def test_otpm_rejection_does_not_double_refund_combined_tpm(rate_limiter):
    """
    Regression for a High-severity review finding: when the project ITPM
    reservation succeeds but OTPM is then over limit,
    _reserve_project_io_tokens_or_raise rolls back the combined-TPM
    reservation that already succeeded earlier in the same pre-call, then
    raises. If it doesn't also mark that reservation released,
    async_post_call_failure_hook -- which fires next in the real request
    lifecycle, since raising from async_pre_call_hook triggers it -- sees
    the same still-stashed reservation and refunds it a second time,
    driving the combined TPM counter negative and letting a caller push
    past the project's real TPM budget by repeatedly triggering OTPM
    rejections.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-double-refund")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-double-refund",
        project_metadata={
            "model_tpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 5},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [
            {"role": "user", "content": "hello there, this is a test message"}
        ],
        "max_tokens": 60,  # blows past the 5-token OTPM limit
    }

    tpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project",
        value="proj-double-refund:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_otpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429

    tpm_after_pre_call = int(
        await cache.async_get_cache(key=tpm_counter_key, local_only=True) or 0
    )
    assert tpm_after_pre_call == 0, (
        f"combined TPM reservation not rolled back: {tpm_after_pre_call}"
    )

    # In the real request lifecycle, async_post_call_failure_hook fires next
    # for a pre-call rejection. It must not refund the same reservation again.
    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=exc_info.value,
        user_api_key_dict=user_api_key_dict,
    )

    tpm_after_failure_hook = int(
        await cache.async_get_cache(key=tpm_counter_key, local_only=True) or 0
    )
    assert tpm_after_failure_hook == 0, (
        f"combined TPM counter went negative from a double refund: {tpm_after_failure_hook}"
    )


@pytest.mark.parametrize(
    "embedding_input",
    [
        list(range(51)),
        [list(range(25)), list(range(26))],
    ],
)
@pytest.mark.asyncio
async def test_project_itpm_rejects_pretokenized_embedding_input(
    rate_limiter,
    embedding_input,
):
    handler, cache = rate_limiter
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-pretokenized-embedding-itpm"),
        project_id="proj-pretokenized-embedding",
        project_metadata={
            "model_itpm_limit": {"text-embedding-3-small": 50},
        },
    )
    data = {
        "model": "text-embedding-3-small",
        "input": embedding_input,
    }

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_itpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="aembedding",
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
async def test_responses_api_not_misclassified_as_embedding_for_output_estimate(
    rate_limiter,
):
    """
    Regression for a High-severity review finding: the Responses API also
    puts its prompt in data["input"], the same field embeddings use, so the
    output-token estimate treated every Responses call as an embedding and
    reserved zero output tokens. call_type now disambiguates the two: the
    same input-only payload gets zero output tokens for an embedding call
    but a real floor for a Responses API call.
    """
    handler, _cache = rate_limiter

    data = {"input": "describe this image in detail"}

    _, embedding_output_estimate = handler._estimate_input_and_output_tokens(
        data=data, call_type="aembedding"
    )
    assert embedding_output_estimate == 0

    _, responses_output_estimate = handler._estimate_input_and_output_tokens(
        data=data, call_type="aresponses"
    )
    assert responses_output_estimate > 0, (
        "Responses API call was misclassified as an embedding and reserved zero output tokens"
    )


@pytest.mark.parametrize(
    ("data", "call_type", "expected_output_tokens"),
    [
        (
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "n": 10,
            },
            "acompletion",
            1000,
        ),
        (
            {
                "prompt": "hello",
                "max_tokens": 100,
                "n": 2,
                "best_of": 5,
            },
            "text_completion",
            500,
        ),
        (
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "n": 0,
                "best_of": "invalid",
            },
            "acompletion",
            100,
        ),
    ],
)
def test_output_estimate_accounts_for_completion_candidates(
    rate_limiter,
    data,
    call_type,
    expected_output_tokens,
):
    handler, _cache = rate_limiter

    _, estimated_output_tokens = handler._estimate_input_and_output_tokens(
        data=data,
        call_type=call_type,
    )

    assert estimated_output_tokens == expected_output_tokens


@pytest.mark.asyncio
async def test_responses_api_usage_reconciles_using_input_output_tokens_fields(
    rate_limiter,
):
    """
    Regression for the other half of the same finding: ResponseAPIUsage
    exposes input_tokens/output_tokens, not prompt_tokens/completion_tokens.
    Before this fix, _resolve_io_token_reconcile_usage couldn't resolve
    Responses API usage at all, so the reservation was silently kept as-is
    instead of being trued up to the much larger actual usage.
    """
    handler, _cache = rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-responses:model")
    otpm_scope = ("model_per_project_otpm", "proj-responses:model")

    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 10
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 10
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    mock_kwargs = {}

    mock_response = ResponsesAPIResponse(
        id="resp_test",
        created_at=int(datetime.now().timestamp()),
        output=[],
        usage=ResponseAPIUsage(input_tokens=80, output_tokens=400, total_tokens=480),
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_adjustments = [i for i in increments if "model_per_project_itpm" in i["key"]]
    otpm_adjustments = [i for i in increments if "model_per_project_otpm" in i["key"]]

    # delta = 80 actual input - 10 reserved = +70
    assert any(i["increment"] == 70 for i in itpm_adjustments), (
        f"ITPM reservation was never trued up to actual Responses API usage: {itpm_adjustments}"
    )
    # delta = 400 actual output - 10 reserved = +390
    assert any(i["increment"] == 390 for i in otpm_adjustments), (
        f"OTPM reservation was never trued up to actual Responses API usage: {otpm_adjustments}"
    )


@pytest.mark.asyncio
async def test_itpm_reservation_accounts_for_audio_content_not_just_text(rate_limiter):
    """
    Regression for the audio half of a Medium-severity review finding:
    litellm.token_counter has no per-type handling for `input_audio`
    content blocks (unlike images, which it does count via
    use_default_image_token_count), so it silently contributes 0 tokens for
    them. Without DEFAULT_AUDIO_TOKEN_ESTIMATE, a burst of audio-heavy
    requests with minimal text would each reserve only the one-token floor
    and blow past the project ITPM limit before post-call reconciliation.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-audio-itpm")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-audio",
        project_metadata={
            # Tighter than DEFAULT_AUDIO_TOKEN_ESTIMATE (300), but far bigger
            # than the handful of tokens the bare text "hi" would cost.
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 50},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "base64-audio-bytes", "format": "wav"},
                    },
                ],
            }
        ],
    }

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_itpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429, (
        "Expected the audio content to push the ITPM reservation over the "
        "50-token limit; if this doesn't raise, audio content isn't being "
        "counted again."
    )


@pytest.mark.asyncio
async def test_itpm_rejects_large_audio_payload_that_would_pass_flat_estimate(
    rate_limiter,
):
    """
    Regression: a caller placing a long audio clip in one block previously
    reserved only 300 tokens (the flat estimate). With the size-proportional
    estimate, the same clip now reserves proportionally more and must trip
    the ITPM limit when the limit is tuned to exactly expose the difference.

    1 100 000 b64 chars → decoded ≈ 825 000 bytes → 825 000 // 1600 ≈ 515
    tokens > the 400-token limit. The flat estimate (300) would have passed.
    """
    handler, cache = rate_limiter

    large_b64 = "A" * 1_100_000

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-large-audio"),
        project_id="proj-large-audio",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 400},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": large_b64, "format": "wav"},
                    },
                ],
            }
        ],
    }

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_itpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429, (
        "Large audio payload must exceed the 400-token ITPM limit under the "
        "size-proportional estimate; the old flat-rate estimate (300 tokens) "
        "would have passed this limit silently"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_type", "request_data"),
    [
        (
            "acompletion",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/high-resolution.png",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ]
            },
        ),
        (
            "acompletion",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "summarize this"},
                            {
                                "type": "file",
                                "file": {
                                    "filename": "document.pdf",
                                    "file_data": "data:application/pdf;base64,dGVzdA==",
                                },
                            },
                        ],
                    }
                ]
            },
        ),
        (
            "aresponses",
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "describe this"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/high-resolution.png",
                                "detail": "high",
                            },
                        ],
                    }
                ]
            },
        ),
        (
            "aresponses",
            {"input": "continue", "previous_response_id": "resp-123"},
        ),
    ],
)
async def test_multimodal_requests_reserve_measured_project_itpm_not_full_limit(
    rate_limiter,
    call_type,
    request_data,
):
    """
    Regression: image, file, and previous_response_id requests used to
    reserve the project's whole ITPM limit up front. Because the atomic
    check is ``current + increment > limit``, that made every such request
    429 as soon as the window carried any usage at all and, while in
    flight, blocked every other request for the same project + model. They
    now reserve the token_counter estimate like everything else, so two
    multimodal requests fit in the same window.
    """
    handler, cache = rate_limiter
    model = "bedrock_mantle/claude-opus"
    project_itpm_limit = 10_000
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-multimodal-measured"),
        project_id="project-multimodal-measured",
        project_metadata={"model_itpm_limit": {model: project_itpm_limit}},
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data={"model": model, **request_data},
        call_type=call_type,
    )
    first_stash = get_request_stash()
    assert first_stash is not None
    first_reservation = first_stash.itpm_reserved_tokens
    assert 0 < first_reservation < project_itpm_limit // 2

    _request_stash.set(None)
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data={"model": model, **request_data},
        call_type=call_type,
    )
    second_stash = get_request_stash()
    assert second_stash is not None
    assert second_stash is not first_stash
    assert second_stash.itpm_reserved_tokens == first_reservation


@pytest.mark.asyncio
async def test_itpm_otpm_reservation_is_kept_on_stream_disconnect(rate_limiter):
    handler, cache = rate_limiter

    api_key = hash_token("sk-disconnect-test")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-disconnect",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 1000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 500},
        },
    )

    data: Dict[str, Any] = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.itpm_reserved_tokens > 0, (
        "pre-call hook must stash an ITPM reservation"
    )
    assert stash.otpm_reserved_tokens > 0, (
        "pre-call hook must stash an OTPM reservation"
    )

    increment_calls: list[dict] = []

    async def mock_increment(increment_list, litellm_parent_otel_span=None):
        for op in increment_list:
            increment_calls.append(
                {"key": op["key"], "increment": op["increment_value"]}
            )

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_release_max_parallel_requests_on_disconnect(
        user_api_key_dict=user_api_key_dict
    )

    itpm_refunds = [
        c
        for c in increment_calls
        if "model_per_project_itpm" in c["key"] and c["increment"] < 0
    ]
    otpm_refunds = [
        c
        for c in increment_calls
        if "model_per_project_otpm" in c["key"] and c["increment"] < 0
    ]

    assert not itpm_refunds
    assert not otpm_refunds
    assert stash.reservation_released is False


@pytest.mark.asyncio
async def test_responses_api_otpm_output_cap_applied_not_skipped_as_embedding(
    rate_limiter,
):
    """
    Regression for a Greptile P1 finding: _reserve_project_io_tokens_or_raise
    classified any request with data["input"] set as an embedding (no output
    tokens), which also misclassifies the Responses API -- it puts its prompt
    in "input" too, but does generate output. That skipped the output cap
    applied whenever the configured OTPM limit is small enough to need it,
    letting an unbounded Responses generation blow past OTPM before
    post-call reconciliation catches up.

    The cap must land on data["max_output_tokens"], not data["max_tokens"]:
    the Responses-to-chat-completion transformation only reads
    max_output_tokens, so a max_tokens cap is silently dropped before
    provider dispatch (a second Greptile finding on the same code path).
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-responses-otpm-cap")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-responses-otpm",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 40},
        },
    )

    data: Dict[str, Any] = {
        "model": "bedrock_mantle/claude-opus",
        "input": "describe this image in detail",
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="aresponses",
    )

    assert data.get("max_output_tokens") is not None, (
        "Responses call was misclassified as an embedding and skipped the OTPM output cap"
    )
    assert data["max_output_tokens"] == 16
    assert data.get("max_tokens") is None, (
        "OTPM output cap was written to max_tokens, which the Responses transformation ignores"
    )


@pytest.mark.asyncio
async def test_explicit_zero_output_responses_call_reserves_effective_provider_minimum(
    rate_limiter,
):
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-responses-zero-output"),
        project_id="proj-responses-zero-output",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 5},
        },
    )

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_otpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={
                "model": "bedrock_mantle/claude-opus",
                "input": "describe this image in detail",
                "max_output_tokens": 0,
            },
            call_type="aresponses",
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
async def test_responses_api_combined_tpm_output_cap_applied_not_skipped_as_embedding(
    rate_limiter,
):
    """
    Regression for the same misclassification bug in the combined-TPM
    output-cap block of async_pre_call_hook (a second, independent
    `is_embedding = data.get("input") is not None` check). A project with
    only a combined model_tpm_limit (no split itpm/otpm) configured small
    enough to need the output cap must still apply it to a Responses call,
    and must write it to max_output_tokens for the same reason as the OTPM
    case above.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-responses-tpm-cap")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-responses-tpm",
        project_metadata={
            "model_tpm_limit": {"bedrock_mantle/claude-opus": 40},
        },
    )

    data: Dict[str, Any] = {
        "model": "bedrock_mantle/claude-opus",
        "input": "describe this image in detail",
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="aresponses",
    )

    assert data.get("max_output_tokens") is not None, (
        "Responses call was misclassified as an embedding and skipped the combined-TPM output cap"
    )
    assert data["max_output_tokens"] == 16
    assert data.get("max_tokens") is None, (
        "combined-TPM output cap was written to max_tokens, which the Responses transformation ignores"
    )


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

    text_only_estimate = handler._estimate_precise_input_tokens(
        data={"input": "hi"},
        model="bedrock_mantle/claude-opus",
        call_type="aresponses",
    )

    multimodal_estimate = handler._estimate_precise_input_tokens(
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


@pytest.mark.asyncio
async def test_refund_reserved_tokens_noop_when_amount_zero(rate_limiter):
    """_refund_reserved_tokens returns immediately without calling Redis when amount=0."""
    handler, _cache = rate_limiter

    calls = []

    async def mock_increment(pipeline_operations, **kwargs):
        calls.extend(pipeline_operations)

    handler.async_increment_tokens_with_ttl_preservation = mock_increment

    await handler._refund_reserved_tokens(
        scopes=[("api_key", "sk-test")],
        amount=0,
    )

    assert not calls, "No Redis ops expected when amount is zero"


@pytest.mark.asyncio
async def test_reserve_io_tokens_noop_when_no_itpm_otpm_descriptors(rate_limiter):
    """reserve_io_tokens returns OK immediately when no ITPM/OTPM descriptors present."""
    handler, _cache = rate_limiter

    non_io_descriptor = {
        "key": "api_key",
        "value": "sk-test",
        "rate_limit": {"tokens_per_unit": 1000, "window_size": 60},
    }
    response, itpm_reserved, otpm_reserved = await handler.reserve_io_tokens(
        descriptors=[non_io_descriptor],
        estimated_input_tokens=50,
        estimated_output_tokens=50,
    )

    assert response["overall_code"] == "OK"
    assert itpm_reserved == 0
    assert otpm_reserved == 0


@pytest.mark.asyncio
async def test_reserve_io_tokens_itpm_only_no_otpm(rate_limiter):
    """When only ITPM descriptors are present (no OTPM), returns itpm_reserved with otpm=0."""
    handler, cache = rate_limiter

    itpm_descriptor = {
        "key": PROJECT_ITPM_DESCRIPTOR_KEY,
        "value": "proj-a:model",
        "rate_limit": {"tokens_per_unit": 10000, "window_size": 60},
    }
    response, itpm_reserved, otpm_reserved = await handler.reserve_io_tokens(
        descriptors=[itpm_descriptor],
        estimated_input_tokens=100,
        estimated_output_tokens=50,
    )

    assert response["overall_code"] == "OK"
    assert itpm_reserved == 100
    assert otpm_reserved == 0


@pytest.mark.asyncio
async def test_otpm_rejection_releases_stashed_parallel_slot(rate_limiter):
    """
    When OTPM is over limit and a parallel slot was already acquired, the
    disconnect cleanup path in _reserve_project_io_tokens_or_raise must
    release that slot. Exercises lines 2773-2777.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-slot"),
        project_id="proj-slot",
        project_metadata={"model_otpm_limit": {"m": 5}},
    )

    data: Dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    slot_released = []

    async def mock_release(acquisition, parent_otel_span=None):
        slot_released.append(acquisition)

    handler._release_parallel_request_slots = mock_release

    stash = get_or_create_request_stash()
    stash.parallel_slot = {
        "slot_id": "test-slot-id",
        "counter_keys": ["some-key"],
    }

    otpm_descriptor = {
        "key": PROJECT_OTPM_DESCRIPTOR_KEY,
        "value": "proj-slot:m",
        "rate_limit": {"tokens_per_unit": 5, "window_size": 60},
    }

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_otpm') as exc_info:
        await handler._reserve_project_io_tokens_or_raise(
            descriptors=[otpm_descriptor],
            data=data,
            requested_model="m",
            user_api_key_dict=user_api_key_dict,
            tpm_reservation_scopes=[],
            tpm_reservation_amount=0,
        )
    assert getattr(exc_info.value, "status_code", None) == 429
    assert slot_released, "Parallel slot must be released when OTPM rejects"
    assert stash.parallel_slot is None


@pytest.mark.asyncio
async def test_itpm_only_status_stored_when_no_prior_rate_limit_response(rate_limiter):
    """
    When only ITPM is configured (no combined TPM/RPM to pre-populate
    the request stash), a successful ITPM reservation must store its status
    there so post-call headers can read it.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-itpm-only-store"),
        project_id="proj-store",
    )

    data: Dict[str, Any] = {"model": "m", "messages": []}

    itpm_descriptor = {
        "key": PROJECT_ITPM_DESCRIPTOR_KEY,
        "value": "proj-store:m",
        "rate_limit": {"tokens_per_unit": 100000, "window_size": 60},
    }

    await handler._reserve_project_io_tokens_or_raise(
        descriptors=[itpm_descriptor],
        data=data,
        requested_model="m",
        user_api_key_dict=user_api_key_dict,
        tpm_reservation_scopes=[],
        tpm_reservation_amount=0,
    )

    stash = get_request_stash()
    assert stash is not None
    stored = stash.rate_limit_response
    assert stored is not None, (
        "ITPM status must be stored in litellm_proxy_rate_limit_response"
    )
    assert stored.get("statuses"), "Stored response must contain statuses"


def test_resolve_io_token_usage_responses_api_with_cached_tokens(rate_limiter):
    """
    ResponsesAPIResponse whose usage.input_tokens_details.cached_tokens is set
    subtracts the cached portion from billable input. Covers line 3501.
    """
    handler, _cache = rate_limiter

    response_obj = ResponsesAPIResponse(
        id="resp_cached",
        created_at=int(datetime.now().timestamp()),
        output=[],
        usage=ResponseAPIUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_tokens_details=InputTokensDetails(cached_tokens=25),
        ),
    )
    billable_input, completion_tokens, resolved = (
        handler._resolve_io_token_reconcile_usage(response_obj)
    )

    assert resolved is True
    assert billable_input == 75, f"Expected 100 - 25 cached = 75, got {billable_input}"
    assert completion_tokens == 50


def test_resolve_io_token_usage_dict_format(rate_limiter):
    """
    Dict-shaped usage on a ModelResponse (older SDK versions or raw dicts in
    the usage field) is parsed correctly. Covers lines 3502-3506.
    """
    handler, _cache = rate_limiter

    response_obj = ModelResponse.model_construct(
        usage={
            "prompt_tokens": 80,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 20},
        }
    )
    billable_input, completion_tokens, resolved = (
        handler._resolve_io_token_reconcile_usage(response_obj)
    )

    assert resolved is True
    assert billable_input == 60, f"Expected 80 - 20 cached = 60, got {billable_input}"
    assert completion_tokens == 40


def test_resolve_io_token_usage_unknown_type_returns_unresolved(rate_limiter):
    """
    A ModelResponse whose usage attribute is not a Usage, ResponseAPIUsage,
    or dict (e.g. a plain int) returns (0, 0, False) so the reservation is
    kept rather than guessed. Covers lines 3507-3508.
    """
    handler, _cache = rate_limiter

    response_obj = ModelResponse.model_construct(usage=42)
    billable_input, completion_tokens, resolved = (
        handler._resolve_io_token_reconcile_usage(response_obj)
    )

    assert resolved is False
    assert billable_input == 0
    assert completion_tokens == 0


@pytest.mark.parametrize(
    ("combined_usage", "expected_increments"),
    [
        (None, ()),
        (
            Usage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
            (-60, -45),
        ),
    ],
)
def test_zero_usage_keeps_reservations_unless_measured_fallback_exists(
    rate_limiter,
    combined_usage,
    expected_increments,
):
    handler, _cache = rate_limiter
    itpm_scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    otpm_scope = (PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    kwargs = {} if combined_usage is None else {"combined_usage_object": combined_usage}
    response_obj = ModelResponse(
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )

    operations = handler._build_io_token_reservation_ops(kwargs, response_obj)

    assert tuple(operation["increment_value"] for operation in operations) == expected_increments


def test_a_responses_reply_reporting_no_tokens_keeps_the_reservations(rate_limiter):
    handler, _cache = rate_limiter
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({(PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({(PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")})
    response_obj = ResponsesAPIResponse(
        id="resp_zero",
        created_at=int(datetime.now().timestamp()),
        output=[],
        usage=ResponseAPIUsage(input_tokens=0, output_tokens=0, total_tokens=0),
    )

    assert tuple(handler._build_io_token_reservation_ops({}, response_obj)) == ()


@pytest.mark.parametrize(
    ("usage", "expected_increments"),
    [
        (
            Usage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
            (40, 15),
        ),
        (
            Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            (100, 60),
        ),
    ],
)
def test_retry_success_charges_released_project_io_reservations(
    rate_limiter,
    usage,
    expected_increments,
):
    handler, _cache = rate_limiter
    itpm_scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    otpm_scope = (PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    stash.reservation_released = True

    operations = handler._build_io_token_reservation_ops(
        {},
        ModelResponse(usage=usage),
    )

    assert tuple(operation["increment_value"] for operation in operations) == expected_increments


@pytest.mark.asyncio
async def test_build_io_token_reservation_ops_skips_unresolvable_usage(rate_limiter):
    """
    When response_obj has no parseable usage, _build_io_token_reservation_ops
    returns [] to keep the reservation as-is rather than zeroing it out on a
    bad guess. Covers line 3538.
    """
    handler, _cache = rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-b:model")
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 50
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    mock_kwargs = {}

    ops = handler._build_io_token_reservation_ops(
        kwargs=mock_kwargs,
        response_obj=object(),
    )

    assert not ops, f"Expected empty ops for unresolvable usage, got {ops}"


@pytest.mark.asyncio
async def test_post_call_failure_skips_rpm_only_descriptor_in_tpm_refund(rate_limiter):
    """
    async_post_call_failure_hook skips descriptors without tokens_per_unit
    (e.g. an RPM-only api_key scope) when building the combined-TPM refund ops,
    so a key with rpm_limit but no tpm_limit doesn't receive a spurious refund
    that would drive its counter negative. Covers the continue guard at line 4250.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-rpm-only-desc")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        rpm_limit=100,
        project_id="proj-rpm-only-desc",
        project_metadata={"model_tpm_limit": {"gpt-3.5-turbo": 100000}},
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 20,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    rpm_tokens_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    api_key_tokens_after = int(
        await cache.async_get_cache(key=rpm_tokens_key, local_only=True) or 0
    )
    assert api_key_tokens_after >= 0, (
        f"RPM-only api_key scope must not receive a negative TPM refund; got {api_key_tokens_after}"
    )


@pytest.mark.asyncio
async def test_max_output_tokens_prevents_cap_injection(rate_limiter):
    """
    Regression for veria-ai comment: when a Responses API request supplies
    max_output_tokens (the canonical Responses output bound) but not max_tokens
    or max_completion_tokens, the has_explicit_max_tokens check was False, so
    the code injected data["max_tokens"] = capped_floor and silently truncated
    the response.

    With the fix, max_output_tokens is included in the explicit-cap check and
    data["max_tokens"] must NOT be injected when max_output_tokens is already
    set.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-max-output-tokens"),
        project_id="proj-responses-max-output",
        project_metadata={
            "model_otpm_limit": {"mock-model": 100},
        },
    )

    data: dict = {
        "model": "mock-model",
        "input": "Summarise the document",
        "max_output_tokens": 80,
        "litellm_call_id": "test-max-output-tokens",
        "metadata": {},
    }

    try:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="responses",
        )
    except Exception:
        pass

    assert "max_tokens" not in data, (
        "data['max_tokens'] must not be injected when max_output_tokens is already "
        "set; the cap injection was overriding the caller's explicit output bound"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_type", "request_data", "cap_field", "reserved_tokens"),
    [
        ("aresponses", {"input": "hello", "max_tokens": 1}, "max_output_tokens", 16),
        (
            "acompletion",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_output_tokens": 1,
            },
            "max_tokens",
            10,
        ),
    ],
)
async def test_output_reservation_ignores_cap_fields_from_other_endpoints(
    rate_limiter,
    call_type,
    request_data,
    cap_field,
    reserved_tokens,
):
    handler, cache = rate_limiter
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token(f"sk-{call_type}"),
        project_id=f"project-{call_type}",
        project_metadata={"model_otpm_limit": {"model": 40}},
    )
    data = {"model": "model", **request_data}

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type=call_type,
    )

    assert data[cap_field] == reserved_tokens
    stash = get_request_stash()
    assert stash is not None
    assert stash.otpm_reserved_tokens == reserved_tokens


def test_responses_input_is_counted_even_when_messages_is_present(rate_limiter):
    handler, _cache = rate_limiter
    small_estimate = handler._estimate_precise_input_tokens(
        data={"input": "short", "messages": [{"role": "user", "content": "ignored"}]},
        model="",
        call_type="aresponses",
    )
    large_estimate = handler._estimate_precise_input_tokens(
        data={"input": "large input " * 500, "messages": []},
        model="",
        call_type="aresponses",
    )

    assert large_estimate > small_estimate


def test_anthropic_messages_usage_reconciles_split_project_quota(rate_limiter):
    handler, _cache = rate_limiter

    billable_input, output_tokens, resolved = handler._resolve_io_token_reconcile_usage(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 25,
                "cache_read_input_tokens": 30,
            }
        }
    )

    assert resolved is True
    assert billable_input == 70
    assert output_tokens == 25


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["agenerate_content", "agenerate_content_stream"],
)
async def test_google_genai_native_contents_reserve_project_itpm(
    rate_limiter,
    call_type,
):
    handler, cache = rate_limiter
    model = "gemini/gemini-2.5-flash"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-google-genai-native-itpm"),
        project_id="project-google-genai-native-itpm",
        project_metadata={"model_itpm_limit": {model: 10_000}},
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data={
            "model": model,
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Gemini quota input " * 200}],
                }
            ],
        },
        call_type=call_type,
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.itpm_reserved_tokens > 100


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["rerank", "arerank"])
async def test_rerank_query_and_documents_enforce_project_itpm(
    rate_limiter,
    monkeypatch,
    call_type,
):
    handler, cache = rate_limiter
    captured = {}

    def token_counter(**kwargs):
        captured.update(kwargs)
        return 101

    monkeypatch.setattr("litellm.token_counter", token_counter)
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token(f"sk-{call_type}-itpm"),
        project_id=f"project-{call_type}-itpm",
        project_metadata={"model_itpm_limit": {"rerank-model": 100}},
    )

    with pytest.raises(Exception, match='Rate limit exceeded for model_per_project_itpm') as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={
                "model": "rerank-model",
                "query": "Which document is most relevant?",
                "documents": ["first document", {"text": "second document"}],
            },
            call_type=call_type,
        )

    assert getattr(exc_info.value, "status_code", None) == 429
    assert captured["text"] == (
        "Which document is most relevant?\n"
        "first document\n"
        "{'text': 'second document'}"
    )


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
    rerank_text = handler._rerank_input_to_text(data)

    assert handler._estimate_precise_input_tokens(
        data,
        model="custom-rerank-model",
        call_type="rerank",
    ) == len(rerank_text) // 4


@pytest.mark.parametrize(
    ("response_obj", "expected"),
    [
        (
            RerankResponse(
                meta={"tokens": {"input_tokens": 42, "output_tokens": 3}}
            ),
            (42, 3, True),
        ),
        (
            RerankResponse(
                meta={
                    "tokens": {"input_tokens": 0, "output_tokens": 0},
                    "billed_units": {"total_tokens": 57},
                }
            ),
            (57, 0, True),
        ),
        (
            RerankResponse(
                meta={
                    "tokens": {"input_tokens": 0, "output_tokens": 0},
                    "billed_units": {"total_tokens": 0},
                }
            ),
            (0, 0, False),
        ),
    ],
)
def test_rerank_usage_reconciles_project_split_token_quota(
    rate_limiter,
    response_obj,
    expected,
):
    handler, _cache = rate_limiter

    assert handler._resolve_io_token_reconcile_usage(response_obj) == expected


def test_split_quota_helpers_handle_non_mapping_inputs(rate_limiter):
    handler, _cache = rate_limiter

    assert _call_id_from_callback_kwargs(object()) is None
    assert handler._is_embedding_request(object(), None) is False
    assert handler._get_explicit_output_cap(object(), None) is None
    assert handler.get_output_candidate_count(object()) == 1
    assert handler.get_output_candidate_count({"n": 1e309}) == 1
    assert (
        handler._get_explicit_output_cap({"max_output_tokens": []}, "responses") is None
    )
    assert handler._apply_implicit_output_cap(object(), 100, "responses") is None
    assert handler._estimate_input_and_output_tokens(object()) == (0, 0)
    assert handler._build_io_token_reservation_ops(object(), object()) == ()


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
def test_get_explicit_output_cap_tolerates_unparseable_values(
    rate_limiter, data, call_type, expected
):
    """A client-supplied cap the proxy cannot parse must fall back to the
    no-cap output estimate instead of raising ValueError and 500ing the
    request before it ever reaches the provider."""
    handler, _cache = rate_limiter

    assert handler._get_explicit_output_cap(data, call_type) == expected


@pytest.mark.asyncio
async def test_project_io_counters_not_double_charged_when_reservation_disabled(
    monkeypatch,
):
    """With LITELLM_TPM_TOKEN_RESERVATION_ENABLED=false the first
    should_rate_limit pass used to +1 every ITPM/OTPM counter on top of the
    full reservation _reserve_project_io_tokens_or_raise always makes,
    permanently inflating each bucket by one token per request."""
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false")
    cache = DualCache()
    handler = RateLimitHandler(internal_usage_cache=InternalUsageCache(cache))
    assert handler.tpm_reservation_enabled is False

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-io-no-reservation"),
        project_id="proj-io-no-reservation",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 1000000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 1000000},
        },
    )
    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.itpm_reserved_tokens > 0
    assert stash.otpm_reserved_tokens > 0

    for descriptor_key, reserved in (
        ("model_per_project_itpm", stash.itpm_reserved_tokens),
        ("model_per_project_otpm", stash.otpm_reserved_tokens),
    ):
        counter_key = handler.create_rate_limit_keys(
            key=descriptor_key,
            value="proj-io-no-reservation:bedrock_mantle/claude-opus",
            rate_limit_type="tokens",
        )
        cached = await cache.async_get_cache(key=counter_key, local_only=True)
        assert int(cached or 0) == reserved, (
            f"{descriptor_key} counter {cached} != reserved {reserved}: "
            "first-pass should_rate_limit double-charged the bucket"
        )


@pytest.mark.parametrize(
    ("call_type", "data"),
    [
        (
            "text_completion",
            {
                "messages": [{"role": "user", "content": "ignored"}],
                "prompt": "abcd",
                "input": "ignored",
                "max_tokens": 1,
            },
        ),
        (None, {"prompt": "abcd", "max_tokens": 1}),
        (None, {"prompt": ["abcd", "efgh"], "max_tokens": 1}),
    ],
)
def test_split_token_estimate_selects_endpoint_input(rate_limiter, call_type, data):
    handler, _cache = rate_limiter

    estimated_input, estimated_output = handler._estimate_input_and_output_tokens(
        data=data,
        call_type=call_type,
    )

    assert estimated_input > 0
    assert estimated_output == 1


@pytest.mark.parametrize(
    "call_type,data,cheap_input,precise_input",
    [
        ("embedding", {"input": [1, 2, 3]}, 1, 3),
        ("embedding", {"input": [[1, 2], [3]]}, 2, 3),
        ("rerank", {"query": "query text", "documents": ["first document", "second document"]}, 0, None),
    ],
)
def test_nonmedia_input_keeps_distinct_cheap_and_precise_reservations(
    rate_limiter: tuple[RateLimitHandler, DualCache], call_type: str, data: dict[str, object],
    cheap_input: int, precise_input: int | None,
) -> None:
    handler, _ = rate_limiter
    estimate: Final = handler._estimate_request_input_tokens(data, "gpt-6-astra", call_type, precise=True)
    assert handler._estimate_input_and_output_tokens(
        data, min_configured_tpm_limit=100, call_type=call_type, input_estimate=estimate,
    ) == (cheap_input, 0)
    assert estimate.precise_tokens is not None and estimate.precise_tokens > cheap_input
    if precise_input is not None:
        assert estimate.precise_tokens == precise_input


def test_split_quota_multimodal_guards_handle_non_mapping_inputs(rate_limiter):
    handler, _cache = rate_limiter

    assert handler._estimate_audio_block_tokens(
        object()
    ) == handler._estimate_audio_block_tokens({})
    assert handler._responses_input_to_chat_messages(object()) == ()
    assert handler._estimate_precise_input_tokens(object(), model=None) == 0


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

    assert (
        handler._estimate_precise_input_tokens(data, model="test", call_type=call_type)
        == 7
    )
    assert captured["messages"] is None
    assert captured["text"] == expected_text


@pytest.mark.asyncio
async def test_project_io_reservation_ignores_non_mapping_request_data(rate_limiter):
    handler, _cache = rate_limiter

    await handler._reserve_project_io_tokens_or_raise(
        descriptors=[],
        data=object(),
        requested_model=None,
        user_api_key_dict=UserAPIKeyAuth(),
        tpm_reservation_scopes=(),
        tpm_reservation_amount=0,
    )


@pytest.mark.asyncio
async def test_streaming_combined_usage_reconciles_project_io_reservations(
    rate_limiter,
):
    handler, _cache = rate_limiter
    itpm_scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    otpm_scope = (PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    kwargs = {
        "combined_usage_object": Usage(
            prompt_tokens=40,
            completion_tokens=15,
            total_tokens=55,
        ),
    }
    increments = []

    async def capture_increments(increment_list, **_kwargs):
        increments.extend(increment_list)

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        capture_increments
    )

    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj={"response": "stream body"},
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_adjustments = [
        operation
        for operation in increments
        if PROJECT_ITPM_DESCRIPTOR_KEY in operation["key"]
    ]
    otpm_adjustments = [
        operation
        for operation in increments
        if PROJECT_OTPM_DESCRIPTOR_KEY in operation["key"]
    ]
    assert [operation["increment_value"] for operation in itpm_adjustments] == [-60]
    assert [operation["increment_value"] for operation in otpm_adjustments] == [-45]


def test_aggregate_only_combined_usage_reconciles_project_io_reservations(rate_limiter):
    handler, _cache = rate_limiter
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset(
        {(PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")}
    )
    stash.otpm_reserved_tokens = 80
    stash.otpm_reserved_scopes = frozenset(
        {(PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")}
    )
    kwargs = {
        "combined_usage_object": Usage(total_tokens=55),
    }

    operations = handler._build_io_token_reservation_ops(kwargs, object())

    assert [operation["increment_value"] for operation in operations] == [-45, -25]


def test_raw_split_usage_dict_reconciles_project_io_tokens(rate_limiter):
    handler, _cache = rate_limiter

    assert handler._resolve_io_token_reconcile_usage(
        {
            "input_tokens": 30,
            "output_tokens": 12,
            "input_tokens_details": {"cached_tokens": 5},
        }
    ) == (25, 12, True)


@pytest.mark.asyncio
async def test_post_call_success_hook_contains_header_merge_failures(
    rate_limiter, monkeypatch
):
    handler, _cache = rate_limiter
    response = ModelResponse()
    response._hidden_params = {}

    def raise_on_merge(**_kwargs):
        raise RuntimeError("header merge failed")

    monkeypatch.setattr(
        handler,
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
    from tests.large_text import text
    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    handler, _cache = rate_limiter
    stash = get_or_create_request_stash()
    warm_tokenizer("claude-fable-5")
    data: dict[str, object] = {
        "model": "claude-fable-5",
        "messages": [{"role": "user", "content": text * 100}],
    }
    itpm_descriptor = {
        "key": PROJECT_ITPM_DESCRIPTOR_KEY,
        "value": "proj-loop:claude-fable-5",
        "rate_limit": {"tokens_per_unit": 10_000_000, "window_size": 60},
    }

    _, took, lags = await timed_with_loop_lags(
        lambda: handler._reserve_project_io_tokens_or_raise(
            descriptors=[itpm_descriptor],
            data=data,
            requested_model="claude-fable-5",
            user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-itpm-loop"), project_id="proj-loop"),
            tpm_reservation_scopes=[],
            tpm_reservation_amount=0,
        )
    )

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

    assert handler._estimate_input_and_output_tokens(
        data=compacted_request, call_type="aresponses"
    ) == handler._estimate_input_and_output_tokens(
        data={**compacted_request, "input": _COMPACTED_INPUT[2:]}, call_type="aresponses"
    )
    assert handler._estimate_precise_input_tokens(
        compacted_request, model="gpt-4o", call_type="aresponses"
    ) == handler._estimate_precise_input_tokens(
        {**compacted_request, "input": _SUMMARY_INPUT}, model="gpt-4o", call_type="aresponses"
    )


def test_reservation_counts_a_compaction_item_by_its_summary(rate_limiter):
    handler, _cache = rate_limiter
    padded_item: Final = {
        **_COMPACTED_INPUT[2],
        "encrypted_content": json.dumps(
            {"type": "compaction", "content": _COMPACTION_SUMMARY, "opaque_state": "x" * 4000}
        ),
    }

    assert handler._estimate_input_and_output_tokens(
        data={"model": "gpt-4o", "input": [padded_item, _NEXT_TURN], "max_output_tokens": 1}, call_type="aresponses"
    ) == handler._estimate_input_and_output_tokens(
        data={"model": "gpt-4o", "input": _COMPACTED_INPUT[2:], "max_output_tokens": 1}, call_type="aresponses"
    )
