import asyncio
import json
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
import respx

import litellm

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.context_caching import vertex_ai_context_caching
from litellm.llms.vertex_ai.context_caching.known_cached_contents import KnownCachedContents, parse_expire_time
from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import ContextCachingEndpoints
from litellm.utils import get_prompt_cache_min_tokens

_REGISTRY_MODULE: Final = "litellm.llms.vertex_ai.context_caching.known_cached_contents"
_SCOPE: Final = ("https://vertex.test/v1/projects/p/locations/us/cachedContents", "")
_NOW: Final = datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc).timestamp()
_FAR_FUTURE: Final = "2100-01-01T00:00:00.123456789Z"
_PREFIX_REPEATS: Final = max(
    get_prompt_cache_min_tokens(model=f"{provider}/gemini-3.1-pro") for provider in ("vertex_ai", "gemini")
)


@pytest.fixture
def clock() -> MagicMock:
    return MagicMock(return_value=_NOW)


@pytest.fixture
def endpoints(clock: MagicMock) -> ContextCachingEndpoints:
    return ContextCachingEndpoints(known_caches=KnownCachedContents(clock=clock))


def _expire_time(seconds_from_now: float) -> str:
    expires: Final = datetime.fromtimestamp(_NOW + seconds_from_now, timezone.utc)
    return expires.strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z"


def _messages(label: str = "shared") -> list:
    text: Final = f"{label} cacheable prefix " * _PREFIX_REPEATS
    return [
        {"role": "system", "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": "question"},
    ]


def _response(body: dict) -> MagicMock:
    return MagicMock(json=lambda: body)


def _created(name: str, expire_time: str | None) -> dict:
    body: Final = {"name": name, "usageMetadata": {"totalTokenCount": 2048}}
    return body if expire_time is None else {**body, "expireTime": expire_time}


def _sync_call(
    endpoints: ContextCachingEndpoints,
    client: HTTPHandler,
    api_key: str = "k",
    custom_llm_provider: str = "vertex_ai",
    messages=None,
    extra_headers: dict | None = None,
):
    return endpoints.check_and_create_cache(
        extra_headers=extra_headers,
        messages=messages or _messages(),
        optional_params={},
        api_key=api_key,
        api_base=None,
        model="gemini-3.1-pro",
        client=client,
        timeout=30.0,
        logging_obj=MagicMock(model_call_details={}),
        custom_llm_provider=custom_llm_provider,
        vertex_project="p",
        vertex_location="us",
        vertex_auth_header="token",
    )


async def _async_call(
    endpoints: ContextCachingEndpoints, client: AsyncHTTPHandler, timeout: float | httpx.Timeout = 30.0
):
    return await endpoints.async_check_and_create_cache(
        messages=_messages(),
        optional_params={},
        api_key="k",
        api_base=None,
        model="gemini-3.1-pro",
        client=client,
        timeout=timeout,
        logging_obj=MagicMock(model_call_details={}),
        custom_llm_provider="vertex_ai",
        vertex_project="p",
        vertex_location="us",
        vertex_auth_header="token",
    )


def test_created_cache_is_reused_without_listing(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.return_value = _response(_created("cachedContents/1", _expire_time(60)))

    assert _sync_call(endpoints, client)[2] == "cachedContents/1"
    assert _sync_call(endpoints, client)[2] == "cachedContents/1"

    assert client.get.call_count == 1
    assert client.post.call_count == 1


def test_listed_cache_is_reused_without_listing_again(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=HTTPHandler)
    key: Final = vertex_ai_context_caching.local_cache_obj.get_cache_key(
        messages=_messages()[:1], tools=None, tool_choice=None, model="gemini-3.1-pro"
    )
    client.get.return_value = _response(
        {"cachedContents": [{"name": "cachedContents/listed", "displayName": key, "expireTime": _expire_time(60)}]}
    )

    assert _sync_call(endpoints, client)[2] == "cachedContents/listed"
    assert _sync_call(endpoints, client)[2] == "cachedContents/listed"

    assert client.get.call_count == 1
    client.post.assert_not_called()


def test_listed_cache_about_to_expire_is_not_used(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=HTTPHandler)
    key: Final = vertex_ai_context_caching.local_cache_obj.get_cache_key(
        messages=_messages()[:1], tools=None, tool_choice=None, model="gemini-3.1-pro"
    )
    client.get.return_value = _response(
        {"cachedContents": [{"name": "cachedContents/stale", "displayName": key, "expireTime": _expire_time(3)}]}
    )
    client.post.return_value = _response(_created("cachedContents/fresh", _expire_time(60)))

    assert _sync_call(endpoints, client)[2] == "cachedContents/fresh"


def test_cache_without_expire_time_is_not_remembered(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.return_value = _response(_created("cachedContents/1", None))

    _sync_call(endpoints, client)
    _sync_call(endpoints, client)

    assert client.get.call_count == 2


def test_remembered_cache_is_listed_again_once_its_reuse_window_closes(
    endpoints: ContextCachingEndpoints, clock: MagicMock
):
    client: Final = MagicMock(spec=HTTPHandler)
    key: Final = vertex_ai_context_caching.local_cache_obj.get_cache_key(
        messages=_messages()[:1], tools=None, tool_choice=None, model="gemini-3.1-pro"
    )
    client.get.side_effect = [
        _response({}),
        _response(
            {"cachedContents": [{"name": "cachedContents/listed", "displayName": key, "expireTime": _expire_time(600)}]}
        ),
    ]
    client.post.return_value = _response(_created("cachedContents/1", _expire_time(60)))
    assert _sync_call(endpoints, client)[2] == "cachedContents/1"

    clock.return_value = _NOW + 55
    assert _sync_call(endpoints, client)[2] == "cachedContents/listed"


def test_gemini_caches_are_not_shared_across_api_keys(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.return_value = _response(_created("cachedContents/1", _expire_time(60)))

    _sync_call(endpoints, client, api_key="key-a", custom_llm_provider="gemini")
    _sync_call(endpoints, client, api_key="key-b", custom_llm_provider="gemini")

    assert client.post.call_count == 2


def test_gemini_cache_created_with_an_overridden_api_key_is_not_reused_with_the_configured_key(
    endpoints: ContextCachingEndpoints,
):
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.side_effect = [
        _response(_created("cachedContents/caller-project", _expire_time(60))),
        _response(_created("cachedContents/deployment-project", _expire_time(60))),
    ]

    caller: Final = _sync_call(
        endpoints,
        client,
        api_key="deployment-key",
        custom_llm_provider="gemini",
        extra_headers={"X-Goog-Api-Key": "caller-key"},
    )
    configured: Final = _sync_call(endpoints, client, api_key="deployment-key", custom_llm_provider="gemini")

    assert (caller[2], configured[2]) == ("cachedContents/caller-project", "cachedContents/deployment-project")


def test_different_prefixes_do_not_share_a_cache(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.side_effect = [
        _response(_created("cachedContents/a", _expire_time(60))),
        _response(_created("cachedContents/b", _expire_time(60))),
    ]

    assert _sync_call(endpoints, client, messages=_messages("first"))[2] == "cachedContents/a"
    assert _sync_call(endpoints, client, messages=_messages("second"))[2] == "cachedContents/b"


@pytest.mark.asyncio
async def test_concurrent_requests_for_one_prefix_list_and_create_once(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=AsyncHTTPHandler)

    async def slow_list(**_):
        await asyncio.sleep(0.01)
        return _response({})

    client.get = AsyncMock(side_effect=slow_list)
    client.post = AsyncMock(return_value=_response(_created("cachedContents/1", _expire_time(60))))

    results: Final = await asyncio.gather(*(_async_call(endpoints, client) for _ in range(5)))

    assert {result[2] for result in results} == {"cachedContents/1"}
    assert client.get.await_count == 1
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_async_created_cache_is_reused_by_a_later_request(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=AsyncHTTPHandler)
    client.get = AsyncMock(return_value=_response({}))
    client.post = AsyncMock(return_value=_response(_created("cachedContents/1", _expire_time(60))))

    first: Final = await _async_call(endpoints, client)
    client.get = AsyncMock(side_effect=AssertionError("a known cache must not be listed again"))
    second: Final = await _async_call(endpoints, client)

    assert (first[2], second[2]) == ("cachedContents/1", "cachedContents/1")


@pytest.mark.asyncio
async def test_async_listed_cache_about_to_expire_is_not_used(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=AsyncHTTPHandler)
    key: Final = vertex_ai_context_caching.local_cache_obj.get_cache_key(
        messages=_messages()[:1], tools=None, tool_choice=None, model="gemini-3.1-pro"
    )
    client.get = AsyncMock(
        return_value=_response(
            {"cachedContents": [{"name": "cachedContents/stale", "displayName": key, "expireTime": _expire_time(3)}]}
        )
    )
    client.post = AsyncMock(return_value=_response(_created("cachedContents/fresh", _expire_time(60))))

    assert (await _async_call(endpoints, client))[2] == "cachedContents/fresh"


@pytest.mark.asyncio
async def test_async_create_http_error_surfaces_its_status_code(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=AsyncHTTPHandler)
    client.get = AsyncMock(return_value=_response({}))
    request: Final = httpx.Request("POST", "https://vertex.test/v1/projects/p/locations/us/cachedContents")
    failed: Final = httpx.Response(429, request=request, text="RESOURCE_EXHAUSTED")
    client.post = AsyncMock(return_value=failed)

    with pytest.raises(VertexAIError) as raised:
        await _async_call(endpoints, client)

    assert raised.value.status_code == 429


@pytest.mark.asyncio
async def test_waiters_resolve_on_their_own_when_the_first_request_fails(endpoints: ContextCachingEndpoints):
    client: Final = MagicMock(spec=AsyncHTTPHandler)

    async def slow_list(**_):
        await asyncio.sleep(0.01)
        return _response({})

    client.get = AsyncMock(side_effect=slow_list)
    client.post = AsyncMock(
        side_effect=[
            VertexAIError(status_code=500, message="create failed"),
            _response(_created("cachedContents/2", _expire_time(60))),
        ]
    )

    results: Final = await asyncio.gather(
        _async_call(endpoints, client), _async_call(endpoints, client), return_exceptions=True
    )

    assert isinstance(results[0], VertexAIError)
    assert results[1][2] == "cachedContents/2"


@pytest.mark.parametrize("waiter_timeout", [0.05, httpx.Timeout(30.0, read=0.05)])
@pytest.mark.asyncio
async def test_waiter_stops_waiting_for_the_first_request_at_its_own_timeout(
    endpoints: ContextCachingEndpoints, waiter_timeout: float | httpx.Timeout
):
    client: Final = MagicMock(spec=AsyncHTTPHandler)

    async def slow_list(**_):
        await asyncio.sleep(0.5)
        return _response({})

    client.get = AsyncMock(side_effect=slow_list)
    client.post = AsyncMock(return_value=_response(_created("cachedContents/1", _expire_time(60))))

    results: Final = await asyncio.gather(
        _async_call(endpoints, client), _async_call(endpoints, client, waiter_timeout), return_exceptions=True
    )

    assert results[0][2] == "cachedContents/1"
    assert isinstance(results[1], VertexAIError)
    assert results[1].status_code == 408


def test_parse_expire_time_accepts_nanosecond_precision():
    assert parse_expire_time("2026-10-14T18:06:21.123456789Z") == pytest.approx(
        datetime(2026, 10, 14, 18, 6, 21, 123456, tzinfo=timezone.utc).timestamp()
    )
    assert (
        parse_expire_time("2026-10-14T18:06:21Z") == datetime(2026, 10, 14, 18, 6, 21, tzinfo=timezone.utc).timestamp()
    )
    assert parse_expire_time("not a time") is None
    assert parse_expire_time(None) is None


def test_parse_expire_time_reads_a_missing_offset_as_utc():
    assert (
        parse_expire_time("2026-10-14T18:06:21") == datetime(2026, 10, 14, 18, 6, 21, tzinfo=timezone.utc).timestamp()
    )


def test_cache_expiring_within_the_margin_is_not_remembered():
    known: Final = KnownCachedContents(clock=lambda: _NOW)
    known.remember(_SCOPE, "key", "cachedContents/soon", _expire_time(3))

    assert known.get(_SCOPE, "key") is None


def test_registry_is_bounded():
    known: Final = KnownCachedContents(clock=lambda: _NOW)
    with patch(f"{_REGISTRY_MODULE}.MAX_KNOWN_CACHED_CONTENTS", 3):
        for index in range(5):
            known.remember(_SCOPE, f"key-{index}", f"cachedContents/{index}", _expire_time(60))

    assert known.get(_SCOPE, "key-0") is None
    assert known.get(_SCOPE, "key-4") == "cachedContents/4"


def _generated(stream: bool) -> httpx.Response:
    body: Final = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1, "totalTokenCount": 11},
    }
    if stream:
        return httpx.Response(200, text=f"data: {json.dumps(body)}\n\n", headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=body)


def _rejected(status_code: int, message: str) -> httpx.Response:
    return httpx.Response(status_code, json={"error": {"code": status_code, "message": message}})


async def _complete(sync_mode: bool, stream: bool, **kwargs) -> str:
    request: Final = {"model": "gemini/gemini-3.1-pro", "messages": _messages(), "stream": stream, **kwargs}
    if sync_mode:
        response = litellm.completion(**request)
        if stream:
            return "".join(chunk.choices[0].delta.content or "" for chunk in response)
        return response.choices[0].message.content
    response = await litellm.acompletion(**request)
    if stream:
        return "".join([chunk.choices[0].delta.content or "" async for chunk in response])
    return response.choices[0].message.content


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("status_code", "message", "replaced"),
    [
        (400, "Invalid resource state for cache content 4484391867902853120.", True),
        (403, "CachedContent not found (or permission denied)", True),
        (404, "CachedContent not found", True),
        (400, "Invalid value at 'generation_config.temperature'", False),
        (500, "CachedContent not found", False),
    ],
)
@pytest.mark.asyncio
async def test_remembered_cache_rejected_by_generate_content_is_replaced_within_the_same_call(
    respx_mock: respx.MockRouter,
    fake_provider_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
    stream: bool,
    status_code: int,
    message: str,
    replaced: bool,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    list_route: Final = respx_mock.get(url__regex=r".*/cachedContents.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx_mock.post(url__regex=r".*/cachedContents$").mock(
        side_effect=[
            httpx.Response(200, json=_created("cachedContents/deleted", _FAR_FUTURE)),
            httpx.Response(200, json=_created("cachedContents/fresh", _FAR_FUTURE)),
        ]
    )
    generate_route: Final = respx_mock.post(url__regex=r".*:(stream)?[gG]enerateContent.*").mock(
        side_effect=[_generated(stream), _rejected(status_code, message), _generated(stream)]
    )
    logged: Final[list[dict]] = []

    assert await _complete(sync_mode, stream) == "ok"
    if replaced:
        assert await _complete(sync_mode, stream, logger_fn=logged.append) == "ok"
    else:
        with pytest.raises(openai.APIStatusError):
            await _complete(sync_mode, stream, logger_fn=logged.append)

    sent: Final = [json.loads(call.request.content)["cachedContent"] for call in generate_route.calls]
    assert sent == ["cachedContents/deleted", "cachedContents/deleted"] + (["cachedContents/fresh"] if replaced else [])
    assert list_route.call_count == (2 if replaced else 1)
    assert "enerateContent" in logged[-1]["litellm_params"]["api_base"]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "message",
    [
        "Invalid value at 'generation_config.temperature'",
        "Tool config, tools and system instruction should not be set in therequest when using cached content.",
        "CachedContent can not be used with GenerateContent request setting system_instruction, tools or tool_config.",
    ],
)
@pytest.mark.asyncio
async def test_error_that_is_not_about_a_missing_cache_keeps_the_remembered_cache(
    respx_mock: respx.MockRouter,
    fake_provider_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
    stream: bool,
    message: str,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    list_route: Final = respx_mock.get(url__regex=r".*/cachedContents.*").mock(
        return_value=httpx.Response(200, json={})
    )
    respx_mock.post(url__regex=r".*/cachedContents$").mock(
        return_value=httpx.Response(200, json=_created("cachedContents/kept", _FAR_FUTURE))
    )
    generate_route: Final = respx_mock.post(url__regex=r".*:(stream)?[gG]enerateContent.*").mock(
        side_effect=[
            _generated(stream),
            _rejected(400, message),
            _generated(stream),
        ]
    )

    assert await _complete(sync_mode, stream) == "ok"
    with pytest.raises(openai.BadRequestError):
        await _complete(sync_mode, stream)
    assert await _complete(sync_mode, stream) == "ok"

    assert [json.loads(call.request.content)["cachedContent"] for call in generate_route.calls] == [
        "cachedContents/kept"
    ] * 3
    assert list_route.call_count == 1


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_rejected_caller_supplied_cached_content_is_not_sent_again(
    respx_mock: respx.MockRouter,
    fake_provider_credentials: None,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
    stream: bool,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    list_route: Final = respx_mock.get(url__regex=r".*/cachedContents.*").mock(
        return_value=httpx.Response(200, json={})
    )
    generate_route: Final = respx_mock.post(url__regex=r".*:(stream)?[gG]enerateContent.*").mock(
        side_effect=[_rejected(404, "CachedContent not found"), _generated(stream)]
    )

    with pytest.raises(openai.APIStatusError):
        await _complete(sync_mode, stream, cached_content="cachedContents/callers")

    assert [json.loads(call.request.content)["cachedContent"] for call in generate_route.calls] == [
        "cachedContents/callers"
    ]
    assert list_route.call_count == 0
