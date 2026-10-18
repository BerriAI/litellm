import re
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@pytest.mark.asyncio
async def test_second_cleanup_pass_does_not_resurrect_owned_client():
    handler = AsyncHTTPHandler()
    original_client = handler._client
    cache_key = "test-cleanup-no-resurrect"
    litellm.in_memory_llm_clients_cache.cache_dict[cache_key] = handler
    try:
        await close_litellm_async_clients()
        assert original_client.is_closed
        await close_litellm_async_clients()
    finally:
        litellm.in_memory_llm_clients_cache.cache_dict.pop(cache_key, None)

    assert handler._client is original_client


GEMINI_GENERATE_URL: Final = re.compile(r"https://generativelanguage\.googleapis\.com/.*:generateContent.*")


def _gemini_reply(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        },
    )


def _cached_async_handlers() -> list[AsyncHTTPHandler]:
    return [
        handler
        for handler in litellm.in_memory_llm_clients_cache.cache_dict.values()
        if isinstance(handler, AsyncHTTPHandler)
    ]


@pytest.fixture
def gemini_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-cleanup-test")
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
@pytest.mark.usefixtures("gemini_httpx_transport")
async def test_acompletion_client_is_closed_by_cleanup() -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.post(GEMINI_GENERATE_URL).mock(return_value=_gemini_reply("Hi there!"))
        response: Final = await litellm.acompletion(
            model="gemini/gemini-2.0-flash-lite-001",
            messages=[{"role": "user", "content": "Hello"}],
        )
    assert response.choices[0].message.content == "Hi there!"
    clients: Final = [handler._client for handler in _cached_async_handlers()]
    assert clients
    assert not any(client.is_closed for client in clients)

    await close_litellm_async_clients()

    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.usefixtures("gemini_httpx_transport")
async def test_repeated_acompletion_calls_reuse_one_client_that_cleanup_closes() -> None:
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post(GEMINI_GENERATE_URL).mock(
            side_effect=[_gemini_reply(f"Response {index}") for index in range(3)]
        )
        for index in range(3):
            response = await litellm.acompletion(
                model="gemini/gemini-2.0-flash-lite-001",
                messages=[{"role": "user", "content": f"Hello {index}"}],
            )
            assert response.choices[0].message.content == f"Response {index}"
    assert route.call_count == 3
    handlers: Final = _cached_async_handlers()
    assert len(handlers) == 1
    client: Final = handlers[0]._client

    await close_litellm_async_clients()

    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.usefixtures("gemini_httpx_transport")
async def test_cleanup_is_idempotent_and_acompletion_works_afterwards() -> None:
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post(GEMINI_GENERATE_URL).mock(side_effect=[_gemini_reply("Hello!"), _gemini_reply("Hi!")])
        await litellm.acompletion(
            model="gemini/gemini-2.0-flash-lite-001",
            messages=[{"role": "user", "content": "Hello"}],
        )
        for _ in range(3):
            await close_litellm_async_clients()
        response: Final = await litellm.acompletion(
            model="gemini/gemini-2.0-flash-lite-001",
            messages=[{"role": "user", "content": "Hello"}],
        )
    assert response.choices[0].message.content == "Hi!"
    assert route.call_count == 2
    await close_litellm_async_clients()
