from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache, CacheMode
from litellm.constants import RESPONSE_CACHE_EXCLUDED_PROVIDERS
from litellm.rust_bridge.host.cache import CacheAccess, access
from litellm.types.caching import LiteLLMCacheType

REQUEST: Final[dict[str, object]] = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> Cache:
    configured: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", configured)
    return configured


def test_no_configured_cache_grants_no_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", None)

    assert access(REQUEST, "acompletion") is None


@pytest.mark.parametrize(
    ("overrides", "read", "write"),
    (
        ({}, True, True),
        ({"cache": {"no-cache": True}}, False, True),
        ({"cache": {"no-store": True}}, True, False),
        ({"cache": {"no-cache": "true"}}, True, True),
        ({"caching": False}, False, True),
    ),
    ids=("default", "no-cache", "no-store", "non-bool-control", "caching-off"),
)
def test_access_follows_the_python_handler_controls(
    cache: Cache, overrides: dict[str, object], read: bool, write: bool
) -> None:
    assert access({**REQUEST, **overrides}, "acompletion") == CacheAccess(cache=cache, read=read, write=write)


def test_both_controls_off_grants_no_access(cache: Cache) -> None:
    assert access({**REQUEST, "cache": {"no-cache": True, "no-store": True}}, "acompletion") is None


def test_an_excluded_provider_grants_no_access(cache: Cache) -> None:
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
def test_supported_call_types_gate_access(cache: Cache, supported: list[str], call_type: str, granted: bool) -> None:
    cache.supported_call_types = supported  # pyright: ignore[reportAttributeAccessIssue]  # test narrows the configured list

    assert (access(REQUEST, call_type) is not None) == granted


def test_default_off_mode_grants_access_only_on_opt_in(cache: Cache) -> None:
    cache.mode = CacheMode.default_off

    assert access(REQUEST, "acompletion") is None
    assert access({**REQUEST, "cache": {"use-cache": True}}, "acompletion") == CacheAccess(
        cache=cache, read=True, write=True
    )
