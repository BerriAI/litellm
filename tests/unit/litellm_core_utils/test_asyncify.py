import asyncio
from typing import Final

import pytest

from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.asyncify import run_async_function


class _AsyncClient:
    def __init__(self, fails: bool = False) -> None:
        self.closes = 0
        self._fails = fails

    async def aclose(self) -> None:
        self.closes += 1
        if self._fails:
            raise RuntimeError("close failed")


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> LLMClientCache:
    fresh: Final = LLMClientCache(evicted_client_closer=EvictedClientCloser())
    monkeypatch.setattr("litellm.in_memory_llm_clients_cache", fresh)
    return fresh


def _cache_on_throwaway_loop(cache: LLMClientCache, client: _AsyncClient, owned: bool) -> str:
    async def store() -> str:
        cache.set_cache("client", client, litellm_owned_client=owned)
        return "result"

    return run_async_function(store)


@pytest.mark.parametrize("owned, closes", [(True, 1), (False, 0)])
def test_throwaway_loop_drops_its_clients_and_closes_only_owned_ones(
    cache: LLMClientCache, owned: bool, closes: int
) -> None:
    client: Final = _AsyncClient()

    assert _cache_on_throwaway_loop(cache, client, owned) == "result"

    assert client.closes == closes
    assert cache.cache_dict == {}


def test_failing_close_keeps_the_call_result(cache: LLMClientCache) -> None:
    client: Final = _AsyncClient(fails=True)

    assert _cache_on_throwaway_loop(cache, client, owned=True) == "result"
    assert client.closes == 1


def test_call_error_survives_release(cache: LLMClientCache) -> None:
    client: Final = _AsyncClient()

    async def store_then_fail() -> None:
        cache.set_cache("client", client, litellm_owned_client=True)
        raise ValueError("call failed")

    with pytest.raises(ValueError, match="call failed"):
        run_async_function(store_then_fail)
    assert client.closes == 1


def test_release_from_running_loop_runs_on_its_own_thread(cache: LLMClientCache) -> None:
    client: Final = _AsyncClient()

    async def caller() -> str:
        return _cache_on_throwaway_loop(cache, client, owned=True)

    assert asyncio.run(caller()) == "result"
    assert client.closes == 1


def test_release_leaves_other_loops_clients_alone(cache: LLMClientCache) -> None:
    async def scenario() -> tuple[object, int]:
        kept: Final = _AsyncClient()
        cache.set_cache("client", kept, litellm_owned_client=True)
        _ = await asyncio.to_thread(_cache_on_throwaway_loop, cache, _AsyncClient(), True)
        return cache.get_cache("client"), kept.closes

    cached, closes = asyncio.run(scenario())

    assert isinstance(cached, _AsyncClient)
    assert closes == 0
