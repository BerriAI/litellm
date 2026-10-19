import asyncio
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await background cache writes
    cache_key_kwargs,
    response_cache_key,
)
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import RESPONSE_CACHE_EXCLUDED_PROVIDERS
from litellm.llms.anthropic.pass_through.messages.response_cache import get_cached_stream_events
from litellm.rust_bridge.host.cache import (
    MESSAGES_ENTRY,
    RESPONSE_ENTRY,
    CacheAccess,
    EntryFormat,
    ResponseOutput,
    StreamOutput,
    access,
    alookup,
    astore,
    lookup,
    store,
)
from litellm.types.caching import CachingSupportedCallTypes, LiteLLMCacheType

REQUEST: Final[dict[str, object]] = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
STREAM_REQUEST: Final[dict[str, object]] = {**REQUEST, "stream": True}
EMPTY: Final[dict[str, object]] = {"choices": []}
_ENTRY: Final = TypeAdapter(dict[str, object])
MESSAGES_EVENTS: Final = (
    'event: message_start\ndata: {"type":"message_start"}\n\n',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0}\n\n',
    'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
)
RAW_ENTRY: Final = EntryFormat(
    entry=RESPONSE_ENTRY.entry,
    output=lambda entry, _: ResponseOutput(value=entry),
    tiers_responses=True,
    tiers_streams=False,
)


class KeyCountingCache(Cache):
    key_builds: int = 0

    def get_cache_key(self, **kwargs: object) -> str:
        self.key_builds += 1
        return super().get_cache_key(**kwargs)


def response_output(value: object) -> dict[str, object]:
    return ResponseOutput(value=value).model_dump()


def configure(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Cache:
    monkeypatch.setattr(litellm, "cache", cache)
    return cache


def _cache_access(
    cache: Cache,
    request: dict[str, object] = REQUEST,
    memory_tier: DualCache | None = None,
    entry_format: EntryFormat = RESPONSE_ENTRY,
) -> CacheAccess:
    return CacheAccess(
        cache=cache,
        request=cache_key_kwargs(request),
        key=response_cache_key(cache, request),
        read=True,
        write=True,
        memory_tier=memory_tier,
        format=entry_format,
    )


def stored_entry(cache: Cache, request: dict[str, object]) -> object:
    hit: Final = lookup(_cache_access(cache, request, entry_format=RAW_ENTRY))
    return None if hit is None else hit.response["value"]


async def memory_entry(memory_tier: DualCache, key: object) -> object:
    return await memory_tier.in_memory_cache.async_get_cache(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # InMemoryCache.async_get_cache is unannotated


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


@pytest.mark.parametrize(
    ("overrides", "granted"),
    (({}, False), ({"cache": {"use-cache": True}}, True)),
    ids=("plain", "opt-in"),
)
def test_default_off_mode_grants_only_calls_that_opt_in_and_hashes_nothing_otherwise(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object], granted: bool
) -> None:
    cache: Final = KeyCountingCache(type=LiteLLMCacheType.LOCAL, mode=CacheMode.default_off)
    configure(monkeypatch, cache)

    assert (access({**REQUEST, **overrides}, "acompletion") is not None) is granted
    assert cache.key_builds == int(granted)


def test_a_miss_then_store_builds_the_key_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = KeyCountingCache(type=LiteLLMCacheType.LOCAL)
    configure(monkeypatch, cache)
    granted: Final = access(REQUEST, "acompletion")
    assert granted is not None

    assert lookup(granted) is None
    store(granted, response_output({"id": "cached"}))
    hit: Final = lookup(granted)

    assert hit is not None
    assert hit.response == response_output({"id": "cached"})
    assert cache.key_builds == 1


def test_a_cache_that_fails_to_decide_grants_no_access(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingCache(Cache):
        def should_use_cache(self, **kwargs: object) -> bool:
            raise RuntimeError("cache unavailable")

    configure(monkeypatch, FailingCache(type=LiteLLMCacheType.LOCAL))

    assert access(REQUEST, "acompletion") is None


@pytest.mark.parametrize("cache_key", (None, "preset-key"), ids=("generated-key", "preset-key"))
def test_lookup_returns_cached_response_and_key(cache_key: str | None) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    request: Final = REQUEST if cache_key is None else {**REQUEST, "cache_key": cache_key}
    cache_access: Final = _cache_access(cache, request)
    expected_key: Final = cache.get_cache_key(**REQUEST) if cache_key is None else cache_key

    store(cache_access, response_output({"id": "cached"}))
    hit: Final = lookup(cache_access)

    assert hit is not None
    assert hit.response == response_output({"id": "cached"})
    assert hit.key == expected_key


def test_lookup_skips_responses_without_output() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache_access: Final = _cache_access(cache)

    store(cache_access, response_output(EMPTY))
    assert lookup(cache_access) is None

    cache.add_cache(EMPTY, **REQUEST)
    assert lookup(cache_access) is None


@pytest.mark.asyncio
async def test_async_lookup_uses_memory_tier_and_discards_empty_responses() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    memory_tier: Final = DualCache(in_memory_cache=InMemoryCache())
    cache_access: Final = _cache_access(cache, memory_tier=memory_tier)

    assert await alookup(cache_access) is None

    await cache.async_add_cache({"id": "cached"}, dynamic_cache_object=memory_tier, **REQUEST)
    hit: Final = await alookup(cache_access)

    assert hit is not None
    assert hit.response == response_output({"id": "cached"})
    assert hit.key == cache.get_cache_key(**REQUEST)

    empty_request: Final = {**REQUEST, "cache_key": "empty-response"}
    await cache.async_add_cache(EMPTY, dynamic_cache_object=memory_tier, **empty_request)

    assert await alookup(_cache_access(cache, empty_request, memory_tier)) is None
    assert await memory_entry(memory_tier, "empty-response") is None


@pytest.mark.asyncio
async def test_async_store_routes_writes_and_skips_responses_without_output() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    memory_tier: Final = DualCache(in_memory_cache=InMemoryCache())
    cache_access: Final = _cache_access(cache, memory_tier=memory_tier)

    await astore(cache_access, response_output({"id": "cached"}))
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    hit: Final = await alookup(cache_access)

    assert hit is not None
    assert hit.response == response_output({"id": "cached"})
    assert hit.key == cache.get_cache_key(**REQUEST)

    empty_request: Final = {**REQUEST, "cache_key": "empty-response"}
    empty_access: Final = _cache_access(cache, empty_request, memory_tier)
    await astore(empty_access, response_output(EMPTY))

    assert await alookup(empty_access) is None
    assert await memory_entry(memory_tier, "empty-response") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry_format", "request_kwargs", "output", "tiered"),
    (
        (RESPONSE_ENTRY, REQUEST, ResponseOutput(value={"id": "cached"}), True),
        (MESSAGES_ENTRY, REQUEST, ResponseOutput(value={"id": "cached"}), False),
        (MESSAGES_ENTRY, STREAM_REQUEST, StreamOutput(value="".join(MESSAGES_EVENTS)), True),
    ),
    ids=("response", "messages-response", "messages-stream"),
)
async def test_an_async_store_reaches_the_worker_memory_tier_as_the_python_route_does(
    entry_format: EntryFormat,
    request_kwargs: dict[str, object],
    output: ResponseOutput | StreamOutput,
    tiered: bool,
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    memory_tier: Final = DualCache(in_memory_cache=InMemoryCache())
    cache_access: Final = _cache_access(cache, request_kwargs, memory_tier, entry_format)

    await astore(cache_access, output.model_dump())
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))

    assert (await memory_entry(memory_tier, cache_access.key) is not None) is tiered
    assert (stored_entry(cache, request_kwargs) is not None) is not tiered


@pytest.mark.asyncio
async def test_only_a_redis_backed_cache_answers_from_the_worker_memory_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    request: Final = {**REQUEST, "messages": [{"role": "user", "content": "worker tier"}]}
    configure(monkeypatch, Cache(type=LiteLLMCacheType.REDIS, host="127.0.0.1", port="1"))
    redis_access: Final = access(request, "acompletion")
    configure(monkeypatch, Cache(type=LiteLLMCacheType.LOCAL))
    local_access: Final = access(request, "acompletion")
    assert redis_access is not None and local_access is not None

    await astore(redis_access, response_output({"id": "from-the-worker"}))
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    hit: Final = await alookup(redis_access)

    assert hit is not None
    assert hit.response == response_output({"id": "from-the-worker"})
    assert local_access.memory_tier is None


def test_cache_controls_that_cannot_be_read_grant_no_access(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(monkeypatch, Cache(type=LiteLLMCacheType.LOCAL))

    assert access({**REQUEST, "cache": {1: True}}, "acompletion") is None


def test_a_messages_stream_is_stored_as_the_legacy_event_list_and_replayed() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache_access: Final = _cache_access(cache, STREAM_REQUEST, entry_format=MESSAGES_ENTRY)
    stream: Final = StreamOutput(value="".join(MESSAGES_EVENTS)).model_dump()

    store(cache_access, stream)
    hit: Final = lookup(cache_access)

    assert get_cached_stream_events(_ENTRY.validate_python(stored_entry(cache, STREAM_REQUEST))) == MESSAGES_EVENTS
    assert hit is not None
    assert hit.response == stream


def test_a_stream_on_a_route_without_a_stream_entry_is_not_stored() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)

    store(_cache_access(cache, STREAM_REQUEST), StreamOutput(value="".join(MESSAGES_EVENTS)).model_dump())

    assert stored_entry(cache, STREAM_REQUEST) is None


@pytest.mark.parametrize(("request_kwargs", "hit"), ((REQUEST, True), (STREAM_REQUEST, False)), ids=("unary", "stream"))
def test_a_stored_response_answers_only_a_unary_request(request_kwargs: dict[str, object], hit: bool) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    cache.add_cache({"id": "cached"}, **request_kwargs)

    assert (lookup(_cache_access(cache, request_kwargs)) is not None) is hit
