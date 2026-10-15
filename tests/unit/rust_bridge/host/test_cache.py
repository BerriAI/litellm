import asyncio
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await background cache writes
)
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import RESPONSE_CACHE_EXCLUDED_PROVIDERS
from litellm.rust_bridge.host.cache import CacheAccess, access, alookup, astore, lookup, store
from litellm.types.caching import CachingSupportedCallTypes, LiteLLMCacheType

REQUEST: Final[dict[str, object]] = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


def configure(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Cache:
    monkeypatch.setattr(litellm, "cache", cache)
    return cache


def _cache_access(cache: Cache, memory_tier: DualCache | None = None, tiered_writes: bool = True) -> CacheAccess:
    return CacheAccess(
        cache=cache,
        read=True,
        write=True,
        memory_tier=memory_tier,
        tiered_writes=tiered_writes,
    )


def grants(kwargs: dict[str, object], call_type: str) -> tuple[bool, bool] | None:
    granted: Final = access(kwargs, call_type)
    return None if granted is None else (granted.read, granted.write)


def test_no_configured_cache_grants_no_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", None)

    assert access(REQUEST, "acompletion") is None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({}, (True, True)),
        ({"cache": {"no-cache": True}}, (False, True)),
        ({"cache": {"no-store": True}}, (True, False)),
        ({"cache": {"no-cache": "true"}}, (True, True)),
        ({"caching": False}, (False, True)),
        ({"cache": {"no-cache": True, "no-store": True}}, None),
    ),
    ids=("default", "no-cache", "no-store", "non-bool-control", "caching-off", "both-off"),
)
def test_access_follows_the_python_handler_controls(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], expected: tuple[bool, bool] | None
) -> None:
    configure(monkeypatch, Cache(type=LiteLLMCacheType.LOCAL))

    assert grants({**REQUEST, **overrides}, "acompletion") == expected


def test_an_excluded_provider_grants_no_access(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch, Cache(type=LiteLLMCacheType.LOCAL))
    provider: Final = next(iter(RESPONSE_CACHE_EXCLUDED_PROVIDERS))

    assert access({**REQUEST, "custom_llm_provider": provider}, "acompletion") is None
    assert access({**REQUEST, "model": f"{provider}/gpt-4o"}, "acompletion") is None


@pytest.mark.parametrize(
    ("supported", "call_type", "granted"),
    (
        (["acompletion"], "acompletion", True),
        (["embedding"], "acompletion", False),
        (["responses"], "aresponses", True),
        (["aresponses"], "responses", False),
    ),
)
def test_supported_call_types_gate_access(
    monkeypatch: pytest.MonkeyPatch, supported: list[CachingSupportedCallTypes], call_type: str, granted: bool
) -> None:
    configure(monkeypatch, Cache(type=LiteLLMCacheType.LOCAL, supported_call_types=supported))

    assert (access(REQUEST, call_type) is not None) == granted


def test_default_off_mode_leaves_the_opt_in_to_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = configure(monkeypatch, Cache(type=LiteLLMCacheType.LOCAL, mode=CacheMode.default_off))
    granted: Final = access(REQUEST, "acompletion")

    assert granted is not None
    assert granted.cache is cache


@pytest.mark.parametrize(("call_type", "tiered"), (("acompletion", True), ("anthropic_messages", False)))
def test_only_a_redis_cache_gets_a_worker_memory_tier(
    monkeypatch: pytest.MonkeyPatch, call_type: str, tiered: bool
) -> None:
    configure(monkeypatch, Cache(type=LiteLLMCacheType.LOCAL))
    local: Final = access(REQUEST, call_type)
    configure(monkeypatch, Cache(type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="1"))
    redis: Final = access(REQUEST, call_type)

    assert local is not None and local.memory_tier is None
    assert redis is not None and isinstance(redis.memory_tier, DualCache)
    assert redis.tiered_writes is tiered


@pytest.mark.parametrize("cache_key", (None, "preset-key"), ids=("generated-key", "preset-key"))
def test_lookup_returns_cached_response_and_key(cache_key: str | None) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache_access: Final = _cache_access(cache)
    response: Final = {"id": "cached"}
    request: Final = REQUEST if cache_key is None else {**REQUEST, "cache_key": cache_key}
    expected_key: Final = cache.get_cache_key(**REQUEST) if cache_key is None else cache_key

    store(cache_access, response, request)
    hit: Final = lookup(cache_access, request)

    assert hit is not None
    assert hit.response == response
    assert hit.key == expected_key


def test_lookup_skips_responses_without_output() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache_access: Final = _cache_access(cache)
    response: Final = {"choices": []}

    store(cache_access, response, REQUEST)
    assert lookup(cache_access, REQUEST) is None

    cache.add_cache(response, **REQUEST)
    assert lookup(cache_access, REQUEST) is None


@pytest.mark.asyncio
async def test_async_lookup_uses_memory_tier_and_discards_empty_responses() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    memory_tier: Final = DualCache(in_memory_cache=InMemoryCache())
    cache_access: Final = _cache_access(cache, memory_tier)

    assert await alookup(cache_access, REQUEST) is None

    response: Final = {"id": "cached"}
    await cache.async_add_cache(response, dynamic_cache_object=memory_tier, **REQUEST)
    hit: Final = await alookup(cache_access, REQUEST)

    assert hit is not None
    assert hit.response == response
    assert hit.key == cache.get_cache_key(**REQUEST)

    empty_key: Final = "empty-response"
    empty_request: Final = {**REQUEST, "cache_key": empty_key}
    await cache.async_add_cache({"choices": []}, dynamic_cache_object=memory_tier, **empty_request)

    assert await alookup(cache_access, empty_request) is None
    assert await memory_tier.in_memory_cache.async_get_cache(empty_key) is None


@pytest.mark.asyncio
async def test_async_store_routes_writes_and_skips_responses_without_output() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    memory_tier: Final = DualCache(in_memory_cache=InMemoryCache())
    cache_access: Final = _cache_access(cache, memory_tier)
    response: Final = {"id": "cached"}

    await astore(cache_access, response, REQUEST)
    pending_writes: Final = tuple(_PENDING_CACHE_WRITES)
    await asyncio.gather(*pending_writes)
    hit: Final = await alookup(cache_access, REQUEST)

    assert hit is not None
    assert hit.response == response
    assert hit.key == cache.get_cache_key(**REQUEST)

    empty_key: Final = "empty-response"
    empty_request: Final = {**REQUEST, "cache_key": empty_key}
    await astore(cache_access, {"choices": []}, empty_request)

    assert await alookup(cache_access, empty_request) is None
    assert await memory_tier.in_memory_cache.async_get_cache(empty_key) is None


@pytest.mark.asyncio
async def test_async_store_omits_the_memory_tier_when_tiered_writes_are_disabled() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    memory_tier: Final = DualCache(in_memory_cache=InMemoryCache())
    cache_access: Final = _cache_access(cache, memory_tier, tiered_writes=False)
    response: Final = {"id": "cached"}

    await astore(cache_access, response, REQUEST)
    pending_writes: Final = tuple(_PENDING_CACHE_WRITES)
    await asyncio.gather(*pending_writes)

    assert await cache.async_get_cache(**REQUEST) == response
    assert await memory_tier.async_get_cache(cache.get_cache_key(**REQUEST)) is None
