from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.dual_cache import DualCache
from litellm.constants import RESPONSE_CACHE_EXCLUDED_PROVIDERS
from litellm.rust_bridge.host.cache import access
from litellm.types.caching import CachingSupportedCallTypes, LiteLLMCacheType

REQUEST: Final[dict[str, object]] = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


def configure(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Cache:
    monkeypatch.setattr(litellm, "cache", cache)
    return cache


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
