import json
import time
from collections.abc import Generator
from types import SimpleNamespace
from typing import Final, cast

import pytest

from litellm.caching.caching import Cache
from litellm.caching.gcs_cache import GCSCache
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import CacheLookup, CacheTestHandle, CacheTestResolver, request
from tests.test_litellm_rust.support.fake_gcs import FakeGcs
from tests.test_litellm_rust.support.isolation import rebound

pytestmark: Final = pytest.mark.requires_rust_extension


@pytest.fixture
def fake_gcs() -> Generator[FakeGcs]:
    server: Final = FakeGcs()
    try:
        yield server
    finally:
        server.close()


async def test_gcs_reads_python_entries_and_writes_python_compatible_objects(
    fake_gcs: FakeGcs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    response: Final = {"choices": [{"text": "cached"}], "usage": {"total_tokens": 3}, "flag": True, "empty": None}
    fake_gcs.put(
        "bucket",
        "cache/sync",
        json.dumps({"timestamp": time.time(), "response": json.dumps(response)}).encode(),
    )
    fake_gcs.put("bucket", "cache/async", json.dumps({"timestamp": time.time(), "response": response}).encode())
    fake_gcs.put("bucket", "cache/raw", json.dumps(response).encode())
    fake_gcs.put("bucket", "cache/invalid", b"not a cache entry")
    binding: Final = CacheTestResolver(
        SimpleNamespace(
            cache=CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token=fake_gcs.token,
            )
        )
    ).resolve()

    assert binding.lookup(request("sync")) == response
    assert await binding.async_lookup(request("async")) == response
    assert binding.lookup(request("raw")) == response
    assert await binding.async_lookup(request("invalid")) is None
    assert binding.lookup(request("missing")) is None

    await binding.async_store({**request("native"), "ttl_seconds": 12.0}, response)
    stored: Final = fake_gcs.objects[("bucket", "cache/native")]
    stored_value: Final = cast(dict[str, object], json.loads(stored))
    assert stored_value["response"] == response
    assert isinstance(stored_value["timestamp"], float)
    upload: Final = next(item for item in fake_gcs.requests if item.method == "POST")
    assert upload.path == "/upload/storage/v1/b/bucket/o"
    assert upload.query == "uploadType=media&name=cache%2Fnative"
    assert upload.headers["Authorization"] == f"Bearer {fake_gcs.token}"
    assert upload.headers["Content-Type"] == "application/json"
    upload_text: Final = f"{upload.path}?{upload.query}{upload.headers}"
    assert "ttl" not in upload_text.lower()
    assert "expiry" not in upload_text.lower()
    download: Final = next(item for item in fake_gcs.requests if item.path.endswith("/cache%2Fsync"))
    assert download.path == "/storage/v1/b/bucket/o/cache%2Fsync"
    assert download.query == "alt=media"

    binding.store(request("sync2"), response)
    assert binding.lookup(request("sync2")) == response
    assert GCSCache(bucket_name="bucket", gcs_path="cache").key_prefix == "cache/"
    assert GCSCache(bucket_name="bucket", gcs_path="cache/").key_prefix == "cache/"
    assert GCSCache(bucket_name="bucket").key_prefix == ""


async def test_gcs_batch_lookup_preserves_order_and_treats_malformed_entries_as_misses(fake_gcs: FakeGcs) -> None:
    fake_gcs.put("bucket", "cache/hit", json.dumps({"timestamp": time.time(), "response": {"value": 1}}).encode())
    fake_gcs.put("bucket", "cache/invalid", b"not a cache entry")
    binding: Final = CacheTestResolver(
        SimpleNamespace(
            cache=CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token=fake_gcs.token,
            )
        )
    ).resolve()
    requests: Final = [request("hit"), request("missing"), request("invalid")]
    expected: Final = {"values": [{"value": 1}, None, None], "missing_indices": [1, 2]}

    assert await binding.async_lookup_batch(requests) == expected
    assert binding.lookup_batch(requests) == expected
    await binding.async_store_batch([request("first"), request("second")], [{"value": 1}, {"value": 2}])
    assert ("bucket", "cache/first") in fake_gcs.objects
    assert ("bucket", "cache/second") in fake_gcs.objects


async def test_gcs_facade_binds_only_exact_matching_configuration(
    fake_gcs: FakeGcs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent")
    facade: Final = Cache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket", gcs_path="cache/")
    assert type(facade.cache) is GCSCache

    mismatched_bucket: Final = CacheTestHandle.gcs(
        "other",
        gcs_path="cache",
        endpoint=fake_gcs.url,
        token=fake_gcs.token,
    )
    with pytest.raises(TypeError, match="buckets must match"):
        mismatched_bucket._bind_facade(facade)
    mismatched_prefix: Final = CacheTestHandle.gcs(
        "bucket",
        gcs_path="x",
        endpoint=fake_gcs.url,
        token=fake_gcs.token,
    )
    with pytest.raises(TypeError, match="key prefixes must match"):
        mismatched_prefix._bind_facade(facade)
    mismatched_credentials: Final = CacheTestHandle.gcs(
        "bucket",
        gcs_path="cache",
        path_service_account="sa.json",
        endpoint=fake_gcs.url,
        token=fake_gcs.token,
    )
    with pytest.raises(TypeError, match="credentials must match"):
        mismatched_credentials._bind_facade(facade)
    with pytest.raises(TypeError, match="types must match"):
        CacheTestHandle.memory()._bind_facade(facade)

    matching: Final = CacheTestHandle.gcs(
        "bucket",
        gcs_path="cache",
        endpoint=fake_gcs.url,
        token=fake_gcs.token,
    )
    matching._bind_facade(facade)
    resolver: Final = CacheTestResolver(SimpleNamespace(cache=facade))
    binding: Final = resolver.resolve()
    assert binding.kind == "native"
    await binding.async_store(request("native"), {"value": "native"})
    assert await binding.async_lookup(request("native")) == {"value": "native"}
    assert cast(CacheLookup, facade).get_cache(cache_key="native") is None

    with rebound(facade.cache, "bucket_name", "other"):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade.cache, "key_prefix", "x/"):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade.cache, "path_service_account", "sa.json"):
        assert resolver.resolve().kind == "python_callback"

    def no_get_cache(*args: object, **kwargs: object) -> None:
        return None

    with rebound(facade.cache, "get_cache", no_get_cache):
        assert resolver.resolve().kind == "python_callback"
    with rebound(facade, "ttl", 12):
        assert resolver.resolve().kind == "python_callback"

    class CustomGcs(GCSCache):
        pass

    with rebound(facade, "cache", CustomGcs(bucket_name="bucket", gcs_path="cache/")):
        assert resolver.resolve().kind == "python_callback"
    custom_facade: Final = Cache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket", gcs_path="cache/")
    with rebound(custom_facade, "cache", CustomGcs(bucket_name="bucket", gcs_path="cache/")):
        with pytest.raises(TypeError, match="types must match"):
            matching._bind_facade(custom_facade)

    missing_bucket: Final = Cache(type=LiteLLMCacheType.GCS)
    with pytest.raises(TypeError, match="requires a configured bucket name"):
        matching._bind_facade(missing_bucket)


async def test_gcs_flush_is_a_no_op_and_ping_is_not_implemented(
    fake_gcs: FakeGcs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GCS_PATH_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    binding: Final = CacheTestResolver(
        SimpleNamespace(
            cache=CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token=fake_gcs.token,
            )
        )
    ).resolve()
    await binding.async_store(request("key"), {"value": "stored"})
    await binding.async_flush()
    assert ("bucket", "cache/key") in fake_gcs.objects
    assert await binding.async_lookup(request("key")) == {"value": "stored"}
    with pytest.raises(NotImplementedError):
        await binding.ping()

    facade: Final = Cache(type=LiteLLMCacheType.GCS, gcs_bucket_name="bucket", gcs_path="cache/")
    with pytest.raises(AttributeError):
        await facade.ping()
    assert cast(CacheLookup, facade.cache).flush_cache() is None


async def test_gcs_unauthorized_and_server_errors_surface_as_runtime_errors(fake_gcs: FakeGcs) -> None:
    wrong_token: Final = CacheTestResolver(
        SimpleNamespace(
            cache=CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token="wrong-token",
            )
        )
    ).resolve()
    with pytest.raises(RuntimeError):
        wrong_token.lookup(request("missing"))
    assert not fake_gcs.objects

    binding: Final = CacheTestResolver(
        SimpleNamespace(
            cache=CacheTestHandle.gcs(
                "bucket",
                gcs_path="cache",
                endpoint=fake_gcs.url,
                token=fake_gcs.token,
            )
        )
    ).resolve()
    with pytest.raises(RuntimeError):
        binding.lookup(request("server-error"))
    assert binding.lookup(request("missing")) is None
