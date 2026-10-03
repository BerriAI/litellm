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
from litellm.rust_bridge import _native
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import CacheTestResolver, activate_native, native_runtime, request
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
    binding: Final = native_runtime(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))

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
    first: Final = native_runtime(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))
    await first.async_store(request("persistent"), {"value": "persistent"})
    await first.async_store({**request("expiring"), "ttl_seconds": 0.3}, {"value": "expiring"})
    fresh: Final = native_runtime(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))
    assert fresh.lookup(request("persistent")) == {"value": "persistent"}
    assert fresh.lookup(request("expiring")) == {"value": "expiring"}
    await asyncio.sleep(0.4)
    assert fresh.lookup(request("expiring")) is None
    assert fresh.lookup(request("persistent")) == {"value": "persistent"}


def test_selected_disk_runtime_declines_store_changes(tmp_path: Path) -> None:
    facade: Final = activate_native(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))
    selected: Final = CacheTestResolver(SimpleNamespace(cache=facade))
    native: Final = selected.resolve()
    native.store(request("native"), {"value": "native"})
    assert facade.get_cache(cache_key="native") == {"value": "native"}
    replacement: Final = diskcache.Cache(str(tmp_path))
    try:
        with rebound(facade.cache, "disk_cache", replacement):
            with pytest.raises(_native.RustBridgeDeclined):
                selected.resolve()
        assert selected.resolve().kind == "native"
    finally:
        replacement.close()

    class CustomStore(diskcache.Cache):
        pass

    unsupported: Final = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    store: Final = CustomStore(str(tmp_path))
    try:
        unsupported.cache.disk_cache = store
        with pytest.raises(_native.RustBridgeDeclined, match="built-in diskcache store"):
            native_runtime(unsupported)
    finally:
        store.close()


async def test_disk_native_batch_lookup_and_store_report_partial_hits(tmp_path: Path) -> None:
    binding: Final = native_runtime(Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))
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
