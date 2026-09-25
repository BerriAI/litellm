import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import diskcache
import pytest

from litellm.caching.caching import Cache
from litellm.caching.disk_cache import DiskCache
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import CacheTestHandle, CacheTestResolver, request
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


async def test_disk_reads_python_entries_and_python_reads_native_entries(tmp_path: Path) -> None:
    disk_cache: Final = DiskCache(disk_cache_dir=str(tmp_path))
    response: Final = {"choices": [{"text": "cached"}], "usage": {"total_tokens": 3}}
    disk_cache.disk_cache.set(
        "sync",
        {"timestamp": time.time(), "response": json.dumps(response)},
    )
    disk_cache.disk_cache.set("async", json.dumps({"timestamp": time.time(), "response": response}))
    disk_cache.disk_cache.set("raw", json.dumps(response))
    disk_cache.disk_cache.set("invalid", "not a cache entry")
    disk_cache.disk_cache.set(
        "large",
        {"timestamp": time.time(), "response": {"text": "x" * 70_000}},
    )
    binding: Final = CacheTestResolver(SimpleNamespace(cache=CacheTestHandle.disk(str(tmp_path)))).resolve()

    assert binding.lookup(request("sync")) == response
    assert await binding.async_lookup(request("async")) == response
    assert binding.lookup(request("raw")) == response
    assert await binding.async_lookup(request("invalid")) is None
    assert binding.lookup(request("large")) == {"text": "x" * 70_000}

    await binding.async_store({**request("native"), "ttl_seconds": 12.0}, response)
    stored_response: Final = disk_cache.get_cache("native")
    assert isinstance(stored_response, dict)
    assert stored_response["response"] == response
    stored, expire_time = disk_cache.disk_cache.get("native", expire_time=True)
    assert stored is not None
    assert time.time() < expire_time <= time.time() + 12.0
    await binding.async_store(request("no-ttl"), response)
    _, no_expiry = disk_cache.disk_cache.get("no-ttl", expire_time=True)
    assert no_expiry is None


async def test_disk_entries_survive_a_fresh_handle_and_expire_on_time(tmp_path: Path) -> None:
    first: Final = CacheTestResolver(SimpleNamespace(cache=CacheTestHandle.disk(str(tmp_path)))).resolve()
    await first.async_store(request("persistent"), {"value": "persistent"})
    await first.async_store({**request("expiring"), "ttl_seconds": 0.3}, {"value": "expiring"})
    fresh: Final = CacheTestResolver(SimpleNamespace(cache=CacheTestHandle.disk(str(tmp_path)))).resolve()
    assert fresh.lookup(request("persistent")) == {"value": "persistent"}
    assert fresh.lookup(request("expiring")) == {"value": "expiring"}
    await asyncio.sleep(0.4)
    assert fresh.lookup(request("expiring")) is None
    assert fresh.lookup(request("persistent")) == {"value": "persistent"}


def test_disk_facade_registers_and_store_changes_fall_back(tmp_path: Path) -> None:
    facade: Final = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    with pytest.raises(TypeError, match="directories must match"):
        CacheTestHandle.disk(str(tmp_path / "other"))._bind_facade(facade)
    handle: Final = CacheTestHandle.disk(str(tmp_path))
    handle._bind_facade(facade)
    resolver: Final = CacheTestResolver(SimpleNamespace(cache=facade))
    binding: Final = resolver.resolve()
    assert binding.kind == "native"
    binding.store(request("native"), {"value": "native"})
    assert facade.get_cache(cache_key="native") == {"value": "native"}

    with rebound(facade.cache, "disk_cache", diskcache.Cache(str(tmp_path))):
        assert resolver.resolve().kind == "python_callback"
    assert resolver.resolve().kind == "native"

    class CustomDiskCache(DiskCache):
        pass

    with rebound(facade, "cache", CustomDiskCache(disk_cache_dir=str(tmp_path))):
        assert resolver.resolve().kind == "python_callback"

    class CustomStore(diskcache.Cache):
        pass

    custom_facade: Final = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    custom_facade.cache.disk_cache = CustomStore(str(tmp_path))
    with pytest.raises(TypeError, match="built-in diskcache store"):
        CacheTestHandle.disk(str(tmp_path))._bind_facade(custom_facade)


async def test_disk_native_batch_lookup_and_store_report_partial_hits(tmp_path: Path) -> None:
    binding: Final = CacheTestResolver(SimpleNamespace(cache=CacheTestHandle.disk(str(tmp_path)))).resolve()
    requests: Final = [request("hit"), request("miss"), request("disabled")]
    requests[2]["controls"] = {
        "supported_call_type": True,
        "configured": True,
        "native_backend": True,
        "default_on": True,
        "caching": False,
        "no_cache": False,
        "no_store": False,
        "use_cache": False,
    }
    await binding.async_store_batch(requests, [{"value": 1}, {"value": 2}, {"value": 3}])

    partial: Final = await binding.async_lookup_batch(requests)

    assert partial == {
        "values": [{"value": 1}, {"value": 2}, None],
        "missing_indices": [2],
    }
