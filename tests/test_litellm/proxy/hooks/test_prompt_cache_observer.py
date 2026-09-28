import asyncio
import json
import time
from datetime import datetime

import httpx
import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.anthropic.prompt_cache_prediction import cache_scope, parse_prompt
from litellm.proxy.hooks.prompt_cache_prediction import (
    PromptCacheObserver,
    lookup,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.utils import ModelResponse

MODEL = "claude-sonnet-5"
CALLER = "a" * 64
DEPLOYMENT = "native-deployment"
KEY = "test-provider-key"


def body(ttl="5m", texts=("private cache prefix",)):
    return {
        "model": MODEL,
        "max_tokens": 2,
        "system": "private system instructions",
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text, **(
                {"cache_control": {"type": "ephemeral", "ttl": ttl}}
                if index == len(texts) - 1 else {}
            )}
            for index, text in enumerate(texts)
        ]}],
    }


def usage(ttl="5m", read=100, write=200):
    return {
        "input_tokens": 11,
        "output_tokens": 2,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
        "cache_creation": {
            "ephemeral_5m_input_tokens": write if ttl == "5m" else 0,
            "ephemeral_1h_input_tokens": write if ttl == "1h" else 0,
        },
    }


def event(request_body, started=1000.0, headers=None, **overrides):
    request = httpx.Request(
        "POST", "https://api.anthropic.com/v1/messages", json=request_body,
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", **(headers or {})},
    )
    return {
        "call_type": "anthropic_messages",
        "custom_llm_provider": "anthropic",
        "cache_hit": False,
        "httpx_response": httpx.Response(200, request=request),
        "first_api_call_start_time": datetime.fromtimestamp(started),
        "standard_logging_object": {
            "status": "success", "model_id": DEPLOYMENT,
            "metadata": {"user_api_key_hash": CALLER},
        },
        **overrides,
    }


async def observe(cache, request_body=None, native_usage=None, now=1010.0, **overrides):
    observer = PromptCacheObserver(InternalUsageCache(dual_cache=cache), clock=lambda: now)
    response = ModelResponse(
        model=MODEL,
        usage=AnthropicConfig().calculate_usage(native_usage or usage(), reasoning_content=None),
    )
    await observer.async_log_success_event(
        event(request_body or body(), **overrides), response,
        datetime.fromtimestamp(now), datetime.fromtimestamp(now),
    )


def scope(**overrides):
    return cache_scope(**{
        "caller_key_hash": CALLER, "deployment_id": DEPLOYMENT,
        "provider_key": KEY, "model": MODEL, **overrides,
    })


@pytest.mark.parametrize("ttl,expires", [("5m", 1300), ("1h", 4600)])
@pytest.mark.asyncio
async def test_observed_cache_count_and_request_start_expiry_survive_as_stale(ttl, expires):
    cache = DualCache()
    request_body = body(ttl=ttl)
    await observe(cache, request_body, usage(ttl=ttl))
    prefix = parse_prompt(request_body)
    observed = await lookup(cache, scope(), prefix, now=1200)
    assert observed.cached_tokens == 300
    assert observed.observed_at == 1010
    assert observed.expires_at == expires
    assert await lookup(cache, scope(), prefix, now=expires) == observed
    saved = json.dumps(cache.in_memory_cache.cache_dict)
    assert "private cache prefix" not in saved
    assert "private system instructions" not in saved
    assert KEY not in saved
    assert CALLER not in saved


@pytest.mark.parametrize("changed", [
    {"caller_key_hash": "b" * 64}, {"deployment_id": "other"},
    {"provider_key": "rotated"}, {"model": "claude-opus-5"},
    {"anthropic_version": "different"},
])
@pytest.mark.asyncio
async def test_cache_evidence_is_isolated_by_every_scope_dimension(changed):
    cache = DualCache()
    await observe(cache)
    assert await lookup(cache, scope(**changed), parse_prompt(body()), now=1010) is None


@pytest.mark.asyncio
async def test_append_only_prefix_finds_prior_evidence_but_edit_or_context_change_does_not():
    cache = DualCache()
    await observe(cache)
    extended = parse_prompt(body(texts=("private cache prefix", "new turn")))
    prior = await lookup(cache, scope(), extended, now=1010)
    assert prior.cached_tokens == 300
    assert prior.fingerprint != extended.fingerprint
    for changed in (
        body(texts=("edited prefix", "new turn")),
        {**body(), "system": "different system"},
        {**body(), "tools": [{"name": "other", "input_schema": {"type": "object"}}]},
        body(ttl="1h"),
    ):
        assert await lookup(cache, scope(), parse_prompt(changed), now=1010) is None
    outside_lookback = parse_prompt(body(texts=("private cache prefix", *[str(i) for i in range(20)])))
    assert await lookup(cache, scope(), outside_lookback, now=1010) is None


@pytest.mark.parametrize("change", [
    {"thinking": {"type": "enabled", "budget_tokens": 1024}},
    {"tool_choice": {"type": "auto"}},
    {"cache_control": {"type": "ephemeral"}},
    {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
    {"system": [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}]},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
    {"messages": [{"role": "user", "content": "no breakpoint"}]},
])
def test_unsupported_or_ambiguous_shapes_have_no_cache_identity(change):
    assert parse_prompt({**body(), **change}) is None
    duplicate = body()
    duplicate["messages"][0]["content"].append(duplicate["messages"][0]["content"][0])
    assert parse_prompt(duplicate) is None


@pytest.mark.parametrize("overrides", [
    {"cache_hit": True}, {"call_type": "completion"},
    {"custom_llm_provider": "bedrock"}, {"stream": True},
    {"headers": {"anthropic-beta": "unverified-feature"}},
    {"headers": {"x-custom-header": "unverified"}},
    {"standard_logging_object": {"status": "success", "model_id": DEPLOYMENT, "metadata": {}}},
])
@pytest.mark.asyncio
async def test_unverified_source_never_creates_observations(overrides):
    cache = DualCache()
    await observe(cache, **overrides)
    assert await lookup(cache, scope(), parse_prompt(body()), now=1010) is None


@pytest.mark.parametrize("native_usage", [
    usage(write=0),
    {**usage(), "cache_creation": None},
    {**usage(), "cache_creation": {"ephemeral_5m_input_tokens": 199, "ephemeral_1h_input_tokens": 0}},
    usage(ttl="1h"),
    {**usage(), "cache_creation_input_tokens": -200},
])
@pytest.mark.asyncio
async def test_missing_or_contradictory_telemetry_cannot_create_observations(native_usage):
    cache = DualCache()
    await observe(cache, native_usage=native_usage)
    assert await lookup(cache, scope(), parse_prompt(body()), now=1010) is None


@pytest.mark.asyncio
async def test_pure_read_refresh_requires_prior_matching_evidence():
    cache = DualCache()
    await observe(cache, native_usage=usage(read=300, write=0))
    assert await lookup(cache, scope(), parse_prompt(body()), now=1010) is None
    await observe(cache)
    await observe(cache, native_usage=usage(read=300, write=0), started=1100, now=1110)
    assert (await lookup(cache, scope(), parse_prompt(body()), now=1110)).expires_at == 1400


class RecordingObserver(PromptCacheObserver):
    def __init__(self, cache):
        super().__init__(InternalUsageCache(dual_cache=cache))
        self.finished = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await super().async_log_success_event(kwargs, response_obj, start_time, end_time)
        self.finished.set()


def native_response():
    return {
        "id": "msg_prediction", "type": "message", "role": "assistant", "model": MODEL,
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": usage(ttl="1h"),
    }


def stream_response(completed, provider_error=False):
    response = native_response()
    events = [
        {"type": "message_start", "message": {**response, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    ]
    if completed:
        events.append({"type": "message_stop"})
    if provider_error:
        events.append({"type": "error", "error": {"type": "overloaded_error", "message": "temporary failure"}})
    return "".join(f"event: {item['type']}\ndata: {json.dumps(item)}\n\n" for item in events)


class TransportChunks(httpx.AsyncByteStream):
    def __init__(self, payload, chunk_size, fragment_error_only=False):
        self.payload = payload.encode()
        self.chunk_size = chunk_size or len(self.payload)
        self.prefix_length = self.payload.index(b"event: error") if fragment_error_only else 0

    async def __aiter__(self):
        if self.prefix_length:
            yield self.payload[:self.prefix_length]
        for offset in range(self.prefix_length, len(self.payload), self.chunk_size):
            yield self.payload[offset:offset + self.chunk_size]


@pytest.mark.parametrize("stream,completed,provider_error,transport", [
    (False, True, False, "whole"),
    (True, True, False, "whole"),
    (True, False, False, "whole"),
    (True, True, True, "whole"),
    (True, True, False, "fragmented"),
    (True, False, False, "fragmented"),
    (True, True, True, "fragmented"),
    (True, True, True, "fragmented_error"),
    (True, True, False, "unterminated"),
])
@pytest.mark.asyncio
async def test_native_production_callback_records_only_completed_wire_requests(stream, completed, provider_error, transport):
    cache = DualCache()
    observer = RecordingObserver(cache)
    litellm.logging_callback_manager.add_litellm_callback(observer)

    def provider(request):
        if stream:
            payload = stream_response(completed, provider_error)
            if transport == "unterminated":
                payload = payload.removesuffix("\n\n")
            return httpx.Response(
                200, request=request, headers={"content-type": "text/event-stream"},
                stream=TransportChunks(
                    payload, 1 if transport.startswith("fragmented") else None,
                    fragment_error_only=transport == "fragmented_error",
                ),
            )
        return httpx.Response(200, request=request, json=native_response())

    client = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    try:
        request_body = body(ttl="1h")
        before = time.time()
        result = await litellm.anthropic_messages(
            **{**request_body, "model": f"anthropic/{MODEL}"},
            api_key=KEY, client=client, stream=stream, model_info={"id": DEPLOYMENT},
            litellm_metadata={"user_api_key_hash": CALLER, "model_info": {"id": DEPLOYMENT}},
        )
        if stream:
            async for _ in result:
                pass
        await asyncio.wait_for(observer.finished.wait(), timeout=5)
        found = await lookup(cache, scope(), parse_prompt(request_body))
        if completed and not provider_error and transport != "unterminated":
            assert found is not None
            assert found.cached_tokens == 300
            assert before + 3600 <= found.expires_at <= time.time() + 3600
        else:
            assert found is None
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(observer)
        await client.client.aclose()
