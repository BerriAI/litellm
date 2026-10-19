import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("diskcache")

from litellm.caching.disk_cache import DiskCache
from tests._wait_helpers import await_until


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


@pytest.fixture
def disk_cache(tmp_path):
    import litellm
    from litellm.caching import Cache

    litellm.cache = Cache(type="disk", disk_cache_dir=str(tmp_path))
    yield litellm.cache
    litellm.cache = None


COMPLETION_MESSAGES = [{"role": "user", "content": "write a one sentence poem about: disk-cache-unit-key"}]


def test_completion_is_served_from_disk_cache_until_params_change(disk_cache):
    import litellm

    first = litellm.completion(
        "gpt-3.5-turbo", messages=COMPLETION_MESSAGES, caching=True, max_tokens=20, mock_response="first"
    )
    repeat = litellm.completion(
        "gpt-3.5-turbo", messages=COMPLETION_MESSAGES, caching=True, max_tokens=20, mock_response="second"
    )
    changed = litellm.completion(
        "gpt-3.5-turbo", messages=COMPLETION_MESSAGES, caching=True, temperature=0.5, mock_response="third"
    )

    assert (repeat.choices[0].message.content, repeat.id, repeat.created) == (
        first.choices[0].message.content,
        first.id,
        first.created,
    )
    assert changed.choices[0].message.content == "third"


async def test_async_completion_is_served_from_disk_cache_until_params_change(disk_cache):
    import litellm

    first = await litellm.acompletion(
        "gpt-3.5-turbo", messages=COMPLETION_MESSAGES, caching=True, max_tokens=20, mock_response="first"
    )
    await await_until(lambda: len(disk_cache.cache.disk_cache) == 1, message="the first completion is cached")
    repeat = await litellm.acompletion(
        "gpt-3.5-turbo", messages=COMPLETION_MESSAGES, caching=True, max_tokens=20, mock_response="second"
    )
    changed = await litellm.acompletion(
        "gpt-3.5-turbo", messages=COMPLETION_MESSAGES, caching=True, temperature=0.5, mock_response="third"
    )

    assert (repeat.choices[0].message.content, repeat.id, repeat.created) == (
        first.choices[0].message.content,
        first.id,
        first.created,
    )
    assert changed.choices[0].message.content == "third"


async def test_embedding_is_served_from_disk_cache_until_params_change(disk_cache, respx_mock, monkeypatch):
    import httpx

    import litellm

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    input_text = ["hello disk-embedding-unit-key"]
    embeddings_route = respx_mock.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )

    await litellm.aembedding("openai/text-embedding-ada-002", input=input_text, caching=True, api_key="sk-unit")
    await await_until(lambda: len(disk_cache.cache.disk_cache) == 1, message="the first embedding is cached")
    cached = await litellm.aembedding(
        "openai/text-embedding-ada-002", input=input_text, caching=True, api_key="sk-unit"
    )
    assert cached._hidden_params["cache_hit"] is True
    assert cached.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert embeddings_route.call_count == 1

    await litellm.aembedding(
        "openai/text-embedding-ada-002", input=input_text, caching=True, user="charlie", api_key="sk-unit"
    )
    assert embeddings_route.call_count == 2
