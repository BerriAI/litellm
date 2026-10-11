import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("diskcache")

from litellm.caching.disk_cache import DiskCache


class _SlowInt(int):
    def __add__(self, value: int) -> "_SlowInt":
        time.sleep(0.05)
        return _SlowInt(int(self) + value)


@pytest.fixture
def cache(tmp_path):
    return DiskCache(disk_cache_dir=str(tmp_path))


def test_increment_cache_starts_from_zero_when_key_missing(cache):
    assert cache.increment_cache("counter", 3) == 3
    assert cache.get_cache("counter") == 3


def test_increment_cache_adds_to_existing_int(cache):
    cache.set_cache("counter", 7)
    assert cache.increment_cache("counter", 5) == 12
    assert cache.get_cache("counter") == 12


def test_increment_cache_treats_non_int_cached_value_as_zero(cache):
    cache.set_cache("counter", "not-a-number")
    assert cache.increment_cache("counter", 4) == 4
    assert cache.get_cache("counter") == 4


def test_increment_cache_is_atomic_under_thread_concurrency(cache):
    seed = 1000
    cache.set_cache("counter", _SlowInt(seed))
    thread_count = 8
    barrier = threading.Barrier(thread_count)

    def increment(_: int) -> int:
        barrier.wait()
        return cache.increment_cache("counter", 1)

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        tuple(executor.map(increment, range(thread_count)))

    assert cache.get_cache("counter") == seed + thread_count


async def test_async_increment_starts_from_zero_when_key_missing(cache):
    assert await cache.async_increment("counter", 2) == 2


async def test_async_increment_adds_to_existing_int(cache):
    await cache.async_set_cache("counter", 10)
    assert await cache.async_increment("counter", 5) == 15


async def test_async_increment_treats_non_int_cached_value_as_zero(cache):
    await cache.async_set_cache("counter", "corrupt")
    assert await cache.async_increment("counter", 9) == 9


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cache_completion_hit_and_param_sensitivity(tmp_path, sync_mode):
    """Same params hit the disk cache; different params do not."""
    import litellm
    from litellm.caching import Cache

    messages = [{"role": "user", "content": "write a one sentence poem about: disk-cache-unit-key"}]
    litellm.cache = Cache(type="disk", disk_cache_dir=str(tmp_path))
    try:
        call = litellm.completion if sync_mode else litellm.acompletion
        response1 = await call(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            max_tokens=20,
            mock_response="This number is so great!",
        ) if not sync_mode else call(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            max_tokens=20,
            mock_response="This number is so great!",
        )
        response2 = await call(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            max_tokens=20,
            mock_response="This number is great!",
        ) if not sync_mode else call(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            max_tokens=20,
            mock_response="This number is great!",
        )
        assert response1.choices[0].message.content == response2.choices[0].message.content
        assert response1.id == response2.id
        assert response1.created == response2.created

        response3 = await call(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            temperature=0.5,
            mock_response="This number is awful!",
        ) if not sync_mode else call(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            temperature=0.5,
            mock_response="This number is awful!",
        )
        assert response1.choices[0].message.content != response3.choices[0].message.content
    finally:
        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_disk_cache_embedding(tmp_path, respx_mock, monkeypatch):
    """Second identical embedding call is served from the disk cache."""
    import json as _json

    import httpx

    import litellm
    from litellm.caching import Cache

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    input_text = ["hello disk-embedding-unit-key"]
    vector = [0.1, 0.2, 0.3]
    embeddings_route = respx_mock.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )
    litellm.cache = Cache(type="disk", disk_cache_dir=str(tmp_path))
    try:
        response1 = await litellm.aembedding(
            "openai/text-embedding-ada-002", input=input_text, caching=True, api_key="sk-unit"
        )
        response2 = await litellm.aembedding(
            "openai/text-embedding-ada-002", input=input_text, caching=True, api_key="sk-unit"
        )
        assert response2._hidden_params["cache_hit"] is True
        assert embeddings_route.call_count == 1

        response3 = await litellm.aembedding(
            "openai/text-embedding-ada-002",
            input=input_text,
            caching=True,
            user="charlie",
            api_key="sk-unit",
        )
        assert response3._hidden_params.get("cache_hit") is not True
    finally:
        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []
