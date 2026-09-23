import asyncio
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # same pending-writes drain the Python path awaits
)
from litellm.rust_bridge import call_cache
from litellm.types.caching import LiteLLMCacheType

_CALL_TYPE: Final = "anthropic_messages"
# cache hits are re-parsed with ast.literal_eval, so the stored value must be a real dict
_RESPONSE: Final[Mapping[str, object]] = TypeAdapter(dict[str, object]).validate_json(
    '{"id": "msg_proof", "role": "assistant", "content": [{"type": "text", "text": "after"}]}'
)


def _kwargs(*, cache: Mapping[str, object] | None = None, caching: bool | None = None) -> Mapping[str, object]:
    return MappingProxyType(
        {
            name: value
            for name, value in (
                ("model", "anthropic/claude-haiku-4-5"),
                ("messages", json.loads('[{"role": "user", "content": "Say the word after in one word"}]')),
                ("cache_key", "proof-key"),
                ("cache", cache),
                ("caching", caching),
            )
            if value is not None
        }
    )


def _local_cache(monkeypatch: pytest.MonkeyPatch) -> Cache:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", cache)
    return cache


def test_lookup_sync_returns_none_without_a_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", None)

    assert call_cache.lookup_sync(_CALL_TYPE, _kwargs()) is None


def test_store_and_lookup_sync_roundtrip_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_cache(monkeypatch)

    call_cache.store_sync(_CALL_TYPE, _kwargs(), _RESPONSE)

    assert call_cache.lookup_sync(_CALL_TYPE, _kwargs()) == _RESPONSE


def test_lookup_sync_uses_the_generated_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_cache(monkeypatch)
    kwargs: Final = MappingProxyType({name: value for name, value in _kwargs().items() if name != "cache_key"})

    call_cache.store_sync(_CALL_TYPE, kwargs, _RESPONSE)

    assert call_cache.lookup_sync(_CALL_TYPE, kwargs) == _RESPONSE


def test_lookup_sync_honors_no_cache_and_the_caching_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_cache(monkeypatch)
    call_cache.store_sync(_CALL_TYPE, _kwargs(), _RESPONSE)

    assert call_cache.lookup_sync(_CALL_TYPE, _kwargs(cache=MappingProxyType({"no-cache": True}))) is None
    assert call_cache.lookup_sync(_CALL_TYPE, _kwargs(caching=False)) is None


def test_unsupported_call_types_skip_lookup_and_store(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = _local_cache(monkeypatch)
    monkeypatch.setattr(cache, "supported_call_types", ("acompletion",))

    call_cache.store_sync(_CALL_TYPE, _kwargs(), _RESPONSE)

    assert call_cache.lookup_sync(_CALL_TYPE, _kwargs()) is None


def test_no_store_blocks_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_cache(monkeypatch)

    call_cache.store_sync(_CALL_TYPE, _kwargs(cache=MappingProxyType({"no-store": True})), _RESPONSE)

    assert call_cache.lookup_sync(_CALL_TYPE, _kwargs()) is None


@pytest.mark.asyncio
async def test_lookup_uses_the_sync_backend_for_local_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_cache(monkeypatch)
    call_cache.store_sync(_CALL_TYPE, _kwargs(), _RESPONSE)

    assert await call_cache.lookup("aanthropic_messages", _kwargs()) == _RESPONSE


@pytest.mark.asyncio
async def test_store_writes_through_the_pending_task(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_cache(monkeypatch)

    call_cache.store(_CALL_TYPE, _kwargs(), _RESPONSE)
    while _PENDING_CACHE_WRITES:
        await asyncio.gather(*_PENDING_CACHE_WRITES)

    assert call_cache.lookup_sync(_CALL_TYPE, _kwargs()) == _RESPONSE


@pytest.mark.asyncio
async def test_lookup_returns_none_without_a_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", None)

    assert await call_cache.lookup("aanthropic_messages", _kwargs()) is None


def test_store_sync_never_raises_when_the_cache_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cache", None)

    call_cache.store_sync(_CALL_TYPE, _kwargs(), _RESPONSE)
    call_cache.store(_CALL_TYPE, _kwargs(), _RESPONSE)
