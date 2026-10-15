import asyncio
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.context_caching import vertex_ai_context_caching
from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import (
    ContextCachingEndpoints,
    _parse_expire_time,
    known_cached_contents,
)

_MODULE: Final = "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching"


@pytest.fixture(autouse=True)
def _cacheable_prompt_and_fixed_url():
    with (
        patch(f"{_MODULE}.is_prompt_caching_valid_prompt", return_value=True),
        patch.object(
            ContextCachingEndpoints,
            "_get_token_and_url_context_caching",
            return_value=("token", "https://vertex.test/v1/projects/p/locations/us/cachedContents"),
        ),
    ):
        yield


def _expire_time(seconds_from_now: float) -> str:
    expires: Final = datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now)
    return expires.strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z"


def _messages(prefix: str = "cacheable prefix ") -> list:
    return [
        {"role": "system", "content": [{"type": "text", "text": prefix * 50, "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": "question"},
    ]


def _response(body: dict) -> MagicMock:
    return MagicMock(json=lambda: body)


def _created(name: str, expire_time: str | None) -> dict:
    body: Final = {"name": name, "usageMetadata": {"totalTokenCount": 2048}}
    return body if expire_time is None else {**body, "expireTime": expire_time}


def _sync_call(client: HTTPHandler, api_key: str = "k", custom_llm_provider: str = "vertex_ai", messages=None):
    return ContextCachingEndpoints().check_and_create_cache(
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
        vertex_auth_header=None,
    )


async def _async_call(client: AsyncHTTPHandler):
    return await ContextCachingEndpoints().async_check_and_create_cache(
        messages=_messages(),
        optional_params={},
        api_key="k",
        api_base=None,
        model="gemini-3.1-pro",
        client=client,
        timeout=30.0,
        logging_obj=MagicMock(model_call_details={}),
        custom_llm_provider="vertex_ai",
        vertex_project="p",
        vertex_location="us",
        vertex_auth_header=None,
    )


def test_created_cache_is_reused_without_listing():
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.return_value = _response(_created("cachedContents/1", _expire_time(60)))

    assert _sync_call(client)[2] == "cachedContents/1"
    assert _sync_call(client)[2] == "cachedContents/1"

    assert client.get.call_count == 1
    assert client.post.call_count == 1


def test_listed_cache_is_reused_without_listing_again():
    client: Final = MagicMock(spec=HTTPHandler)
    key: Final = vertex_ai_context_caching.local_cache_obj.get_cache_key(
        messages=_messages()[:1], tools=None, tool_choice=None, model="gemini-3.1-pro"
    )
    client.get.return_value = _response(
        {"cachedContents": [{"name": "cachedContents/listed", "displayName": key, "expireTime": _expire_time(60)}]}
    )

    assert _sync_call(client)[2] == "cachedContents/listed"
    assert _sync_call(client)[2] == "cachedContents/listed"

    assert client.get.call_count == 1
    client.post.assert_not_called()


def test_listed_cache_about_to_expire_is_not_used():
    client: Final = MagicMock(spec=HTTPHandler)
    key: Final = vertex_ai_context_caching.local_cache_obj.get_cache_key(
        messages=_messages()[:1], tools=None, tool_choice=None, model="gemini-3.1-pro"
    )
    client.get.return_value = _response(
        {"cachedContents": [{"name": "cachedContents/stale", "displayName": key, "expireTime": _expire_time(3)}]}
    )
    client.post.return_value = _response(_created("cachedContents/fresh", _expire_time(60)))

    assert _sync_call(client)[2] == "cachedContents/fresh"


def test_cache_without_expire_time_is_not_remembered():
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.return_value = _response(_created("cachedContents/1", None))

    _sync_call(client)
    _sync_call(client)

    assert client.get.call_count == 2


def test_remembered_cache_is_listed_again_once_its_reuse_window_closes():
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
    assert _sync_call(client)[2] == "cachedContents/1"

    with patch(f"{_MODULE}.time.time", return_value=datetime.now(timezone.utc).timestamp() + 55):
        assert _sync_call(client)[2] == "cachedContents/listed"


def test_gemini_caches_are_not_shared_across_api_keys():
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.return_value = _response(_created("cachedContents/1", _expire_time(60)))

    _sync_call(client, api_key="key-a", custom_llm_provider="gemini")
    _sync_call(client, api_key="key-b", custom_llm_provider="gemini")

    assert client.post.call_count == 2


def test_different_prefixes_do_not_share_a_cache():
    client: Final = MagicMock(spec=HTTPHandler)
    client.get.return_value = _response({})
    client.post.side_effect = [
        _response(_created("cachedContents/a", _expire_time(60))),
        _response(_created("cachedContents/b", _expire_time(60))),
    ]

    assert _sync_call(client, messages=_messages("first "))[2] == "cachedContents/a"
    assert _sync_call(client, messages=_messages("second "))[2] == "cachedContents/b"


@pytest.mark.asyncio
async def test_concurrent_requests_for_one_prefix_list_and_create_once():
    client: Final = MagicMock(spec=AsyncHTTPHandler)

    async def slow_list(**_):
        await asyncio.sleep(0.01)
        return _response({})

    client.get = AsyncMock(side_effect=slow_list)
    client.post = AsyncMock(return_value=_response(_created("cachedContents/1", _expire_time(60))))

    results: Final = await asyncio.gather(*(_async_call(client) for _ in range(5)))

    assert {result[2] for result in results} == {"cachedContents/1"}
    assert client.get.await_count == 1
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_waiters_resolve_on_their_own_when_the_first_request_fails():
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

    results: Final = await asyncio.gather(_async_call(client), _async_call(client), return_exceptions=True)

    assert isinstance(results[0], VertexAIError)
    assert results[1][2] == "cachedContents/2"


def test_parse_expire_time_accepts_nanosecond_precision():
    assert _parse_expire_time("2026-10-14T18:06:21.123456789Z") == pytest.approx(
        datetime(2026, 10, 14, 18, 6, 21, 123456, tzinfo=timezone.utc).timestamp()
    )
    assert (
        _parse_expire_time("2026-10-14T18:06:21Z") == datetime(2026, 10, 14, 18, 6, 21, tzinfo=timezone.utc).timestamp()
    )
    assert _parse_expire_time("not a time") is None
    assert _parse_expire_time(None) is None


def test_registry_is_bounded():
    with patch(f"{_MODULE}.MAX_KNOWN_CACHED_CONTENTS", 3):
        for index in range(5):
            known_cached_contents.remember("scope", f"key-{index}", f"cachedContents/{index}", _expire_time(60))

    assert known_cached_contents.get("scope", "key-0") is None
    assert known_cached_contents.get("scope", "key-4") == "cachedContents/4"
