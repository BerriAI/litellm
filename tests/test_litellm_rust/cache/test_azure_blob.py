import asyncio
import json
import os
import time
import uuid
from collections.abc import Generator
from types import SimpleNamespace
from typing import Final, cast

import pytest
from azure.storage.blob import ContainerClient

from litellm.caching.azure_blob_cache import AzureBlobCache
from litellm.caching.caching import Cache
from litellm.rust_bridge import _native
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import (
    CacheLookup,
    CacheTestHandle,
    CacheTestResolver,
    assert_native_runtime,
    completion_kwargs,
    request,
    require_rust,
)
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


@pytest.fixture
def azure_blob_facade() -> Generator[Cache]:
    account_url: Final = os.environ.get("AZURE_BLOB_CACHE_ACCOUNT_URL")
    if account_url is None:
        pytest.skip(
            "live Azure Blob parity needs AZURE_BLOB_CACHE_ACCOUNT_URL plus DefaultAzureCredential inputs in the environment"
        )
    facade: Final = Cache(
        type=LiteLLMCacheType.AZURE_BLOB,
        azure_account_url=account_url,
        azure_blob_container=f"litellm-parity-{uuid.uuid4().hex[:12]}",
    )
    backend: Final = facade.cache
    assert isinstance(backend, AzureBlobCache)
    try:
        yield facade
    finally:
        backend.container_client.delete_container()
        asyncio.run(backend.disconnect())


def azure_blob_handle(facade: Cache) -> _native._CacheTestHandle:
    backend: Final = facade.cache
    assert isinstance(backend, AzureBlobCache)
    return CacheTestHandle.azure_blob(
        backend.container_client.url.removesuffix(f"/{backend.container_client.container_name}"),
        backend.container_client.container_name,
    )


def test_azure_blob_facade_serves_natively_and_python_reads_the_same_blobs(azure_blob_facade: Cache) -> None:
    backend: Final = azure_blob_facade.cache
    assert isinstance(backend, AzureBlobCache)
    handle: Final = azure_blob_handle(azure_blob_facade)
    assert handle.backend == "azure-blob"
    account_url: Final = backend.container_client.url.removesuffix(f"/{backend.container_client.container_name}")
    with pytest.raises(TypeError, match="containers must match"):
        CacheTestHandle.azure_blob(account_url, f"{backend.container_client.container_name}-other")._bind_facade(
            azure_blob_facade
        )
    handle._bind_facade(azure_blob_facade)
    resolver: Final = CacheTestResolver(SimpleNamespace(cache=azure_blob_facade))
    native: Final = resolver.resolve()
    assert native.kind == "native"

    response: Final = {
        "choices": [{"text": "caf\u00e9 \u2603"}],
        "usage": {"total_tokens": 3},
        "flag": True,
        "empty": None,
    }
    native.store({**request("sync"), "ttl_seconds": 0.001}, response)
    native.store(request("sync"), {"choices": [{"text": "second"}]})
    time.sleep(0.01)
    stored: Final = json.loads(backend.container_client.download_blob("sync").readall())
    assert stored["response"] == response
    assert isinstance(stored["timestamp"], float)
    assert native.lookup(request("sync")) == response
    assert cast(CacheLookup, azure_blob_facade).get_cache(cache_key="sync") == response

    backend.set_cache("python", {"timestamp": time.time(), "response": response})
    backend.set_cache("legacy", "bare legacy value")
    backend.container_client.upload_blob("invalid", b"{not json", overwrite=True)
    assert native.lookup(request("python")) == response
    assert native.lookup(request("legacy")) == cast(CacheLookup, azure_blob_facade).get_cache(cache_key="legacy")
    assert native.lookup_batch([request("python"), request("missing"), request("invalid"), request("sync")]) == {
        "values": [response, None, None, response],
        "missing_indices": [1, 2],
    }

    with rebound(azure_blob_facade, "ttl", 12):
        assert resolver.resolve().kind == "python_callback"
    with rebound(backend, "container_client", ContainerClient.from_container_url(backend.container_client.url)):
        assert resolver.resolve().kind == "python_callback"

    def custom_get(*_args: object, **_kwargs: object) -> None:
        return None

    with rebound(backend, "get_cache", custom_get):
        assert resolver.resolve().kind == "python_callback"
    assert resolver.resolve().kind == "python_callback"
    assert cast(CacheLookup, azure_blob_facade).get_cache(cache_key="sync") == response

    class CustomBlobCache(AzureBlobCache):
        pass

    with rebound(azure_blob_facade, "cache", CustomBlobCache(account_url, backend.container_client.container_name)):
        assert resolver.resolve().kind == "python_callback"
        with pytest.raises(TypeError):
            azure_blob_handle(azure_blob_facade)._bind_facade(azure_blob_facade)


async def test_azure_blob_native_async_writes_overwrite_batch_and_flush_like_python(azure_blob_facade: Cache) -> None:
    backend: Final = azure_blob_facade.cache
    assert isinstance(backend, AzureBlobCache)
    azure_blob_handle(azure_blob_facade)._bind_facade(azure_blob_facade)
    binding: Final = CacheTestResolver(SimpleNamespace(cache=azure_blob_facade)).resolve()
    assert binding.kind == "native"
    ping: Final = cast(dict[str, object], await binding.ping())
    assert ping["status"] == "success", ping

    await binding.async_store(request("async"), {"value": 1})
    await binding.async_store({**request("async"), "ttl_seconds": 0.001}, {"value": 2})
    time.sleep(0.01)
    assert await binding.async_lookup(request("async")) == {"value": 2}
    assert await backend.async_get_cache("async") == json.loads(
        backend.container_client.download_blob("async").readall()
    )
    assert cast(CacheLookup, azure_blob_facade).get_cache(cache_key="async") == {"value": 2}

    await binding.async_store_batch([request("first"), request("second")], [{"value": 3}, {"value": 4}])
    assert await binding.async_lookup_batch([request("second"), request("missing"), request("first")]) == {
        "values": [{"value": 4}, None, {"value": 3}],
        "missing_indices": [1],
    }
    await binding.async_flush()
    assert [blob.name for blob in backend.container_client.list_blobs()] == []
    assert await binding.async_lookup(request("async")) is None


async def test_azure_blob_rust_required_rule_activates_natively(monkeypatch: pytest.MonkeyPatch) -> None:
    account_url: Final = os.environ.get("AZURE_BLOB_CACHE_ACCOUNT_URL")
    if account_url is None:
        pytest.skip(
            "live Azure Blob parity needs AZURE_BLOB_CACHE_ACCOUNT_URL plus DefaultAzureCredential inputs in the environment"
        )
    require_rust(monkeypatch, LiteLLMCacheType.AZURE_BLOB)
    facade: Final = Cache(
        type=LiteLLMCacheType.AZURE_BLOB,
        azure_account_url=account_url,
        azure_blob_container=f"litellm-parity-{uuid.uuid4().hex[:12]}",
    )
    backend: Final = facade.cache
    assert isinstance(backend, AzureBlobCache)
    try:
        assert_native_runtime(facade)
        kwargs: Final = completion_kwargs("azure")
        await facade.async_add_cache({"answer": "azure"}, **kwargs)
        assert await facade.async_get_cache(**kwargs) == {"answer": "azure"}
        assert backend.get_cache(facade.get_cache_key(**kwargs))["response"] == {"answer": "azure"}
    finally:
        backend.container_client.delete_container()
        await backend.disconnect()
