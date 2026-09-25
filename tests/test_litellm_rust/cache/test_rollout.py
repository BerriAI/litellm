import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Final, TypeAlias, cast
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from litellm.caching.caching import Cache
from litellm.rust_bridge.response_cache import NativeResponseCacheRuntime, ResponseCacheRuntime, resolve_response_cache
from litellm.types.caching import LiteLLMCacheType
from litellm.types.utils import EmbeddingResponse
from tests.test_litellm_rust.support.cache import assert_native_runtime, completion_kwargs, require_rust
from tests.test_litellm_rust.support.s3_stub import S3Stub

pytestmark: Final = pytest.mark.requires_rust_extension


CacheFactory: TypeAlias = Callable[[], Cache]


@pytest.fixture
def cache_factory(request: pytest.FixtureRequest, tmp_path: Path) -> CacheFactory:
    backend: Final = cast(LiteLLMCacheType, request.param)
    match backend:
        case LiteLLMCacheType.LOCAL:
            return lambda: Cache(type=backend)
        case LiteLLMCacheType.DISK:
            return lambda: Cache(type=backend, disk_cache_dir=str(tmp_path))
        case LiteLLMCacheType.REDIS:
            parsed: Final = urlparse(cast(str, request.getfixturevalue("redis_url")))
            return lambda: Cache(type=backend, host=parsed.hostname, port=str(parsed.port))
        case LiteLLMCacheType.S3:
            stub: Final = cast(S3Stub, request.getfixturevalue("s3_stub"))
            return lambda: Cache(
                type=backend,
                s3_bucket_name="cache-bucket",
                s3_region_name="us-east-1",
                s3_endpoint_url=stub.url,
                s3_aws_access_key_id="key",
                s3_aws_secret_access_key="secret",
                s3_path="team",
            )
        case LiteLLMCacheType.GCS:
            return lambda: Cache(type=backend, gcs_bucket_name="bucket", gcs_path="cache/")
        case LiteLLMCacheType.REDIS_SEMANTIC:
            return lambda: Cache(
                type=backend,
                redis_url="redis://127.0.0.1:6379",
                similarity_threshold=0.8,
                redis_semantic_cache_embedding_model="text-embedding-3-small",
            )
        case LiteLLMCacheType.VALKEY_SEMANTIC:
            return lambda: Cache(type=backend, redis_url="redis://127.0.0.1:6390/0", similarity_threshold=0.8)
        case _:
            raise AssertionError(f"no local factory for {backend}")


ROUND_TRIP_BACKENDS: Final = (
    LiteLLMCacheType.LOCAL,
    LiteLLMCacheType.DISK,
    LiteLLMCacheType.REDIS,
    LiteLLMCacheType.S3,
)


SHARED_STORE_BACKENDS: Final = (LiteLLMCacheType.DISK, LiteLLMCacheType.REDIS, LiteLLMCacheType.S3)


@pytest.mark.parametrize("backend", list(LiteLLMCacheType))
def test_shipped_rules_keep_every_backend_on_python(backend: LiteLLMCacheType) -> None:
    assert resolve_response_cache(cast(Cache, SimpleNamespace(type=backend))) is None


@pytest.mark.parametrize(
    "cache_factory",
    [
        LiteLLMCacheType.LOCAL,
        LiteLLMCacheType.DISK,
        LiteLLMCacheType.REDIS,
        LiteLLMCacheType.S3,
        LiteLLMCacheType.GCS,
        LiteLLMCacheType.REDIS_SEMANTIC,
        LiteLLMCacheType.VALKEY_SEMANTIC,
    ],
    indirect=True,
)
def test_shipped_rules_construct_python_backed_facades(cache_factory: CacheFactory) -> None:
    assert cache_factory()._native_cache is None  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor


@pytest.mark.parametrize(
    "cache_factory",
    [
        LiteLLMCacheType.LOCAL,
        LiteLLMCacheType.DISK,
        LiteLLMCacheType.REDIS,
        LiteLLMCacheType.S3,
        LiteLLMCacheType.GCS,
        LiteLLMCacheType.REDIS_SEMANTIC,
        LiteLLMCacheType.VALKEY_SEMANTIC,
    ],
    indirect=True,
)
def test_rust_required_rule_activates_the_native_backend(
    cache_factory: CacheFactory, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    require_rust(monkeypatch, cast(LiteLLMCacheType, request.node.callspec.params["cache_factory"]))
    assert_native_runtime(cache_factory())


@pytest.mark.parametrize("cache_factory", ROUND_TRIP_BACKENDS, indirect=True)
async def test_facade_storage_calls_round_trip_through_the_native_backend(
    cache_factory: CacheFactory, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    require_rust(monkeypatch, cast(LiteLLMCacheType, request.node.callspec.params["cache_factory"]))
    facade: Final = cache_factory()
    assert_native_runtime(facade)

    sync_kwargs: Final = completion_kwargs("sync")
    facade.add_cache({"answer": 1}, **sync_kwargs)
    assert facade.get_cache(**sync_kwargs) == {"answer": 1}

    async_kwargs: Final = completion_kwargs("async")
    await facade.async_add_cache({"answer": 2}, **async_kwargs)
    assert await facade.async_get_cache(**async_kwargs) == {"answer": 2}
    assert facade.get_cache(**completion_kwargs("absent")) is None


async def test_memory_facade_writes_bypass_the_python_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    require_rust(monkeypatch, LiteLLMCacheType.LOCAL)
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    assert_native_runtime(facade)
    kwargs: Final = completion_kwargs("memory")
    facade.add_cache({"answer": 1}, **kwargs)
    assert facade.cache.get_cache(facade.get_cache_key(**kwargs)) is None
    assert facade.get_cache(**kwargs) == {"answer": 1}


@pytest.mark.parametrize("cache_factory", SHARED_STORE_BACKENDS, indirect=True)
async def test_native_and_python_facades_share_one_wire_format(
    cache_factory: CacheFactory, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    python_facade: Final = cache_factory()
    assert python_facade._native_cache is None  # pyright: ignore[reportPrivateUsage]  # the activation under test has no public accessor
    require_rust(monkeypatch, cast(LiteLLMCacheType, request.node.callspec.params["cache_factory"]))
    native_facade: Final = cache_factory()
    assert_native_runtime(native_facade)

    native_written: Final = completion_kwargs("native")
    native_facade.add_cache({"writer": "native"}, **native_written)
    assert python_facade.get_cache(**native_written) == {"writer": "native"}

    python_written: Final = completion_kwargs("python")
    python_facade.add_cache({"writer": "python"}, **python_written)
    assert native_facade.get_cache(**python_written) == {"writer": "python"}

    async_native: Final = completion_kwargs("async-native")
    await native_facade.async_add_cache({"writer": "async-native"}, **async_native)
    assert await python_facade.async_get_cache(**async_native) == {"writer": "async-native"}

    async_python: Final = completion_kwargs("async-python")
    await python_facade.async_add_cache({"writer": "async-python"}, **async_python)
    assert await native_facade.async_get_cache(**async_python) == {"writer": "async-python"}


@pytest.mark.parametrize("cache_factory", ROUND_TRIP_BACKENDS, indirect=True)
async def test_embedding_pipeline_stores_one_native_entry_per_input(
    cache_factory: CacheFactory, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    require_rust(monkeypatch, cast(LiteLLMCacheType, request.node.callspec.params["cache_factory"]))
    facade: Final = cache_factory()
    assert_native_runtime(facade)
    inputs: Final = [f"alpha {uuid4().hex}", f"beta {uuid4().hex}"]
    result: Final = EmbeddingResponse(
        model="text-embedding-3-small",
        data=[
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
    )
    await facade.async_add_cache_pipeline(result, model="text-embedding-3-small", input=inputs)

    keys: Final = [facade.get_cache_key(model="text-embedding-3-small", input=text) for text in inputs]
    assert len(set(keys)) == len(inputs)
    for text, expected in zip(inputs, ([0.1, 0.2], [0.3, 0.4]), strict=True):
        cached = await facade.async_get_cache(model="text-embedding-3-small", input=text)
        assert isinstance(cached, dict)
        assert cached["embedding"] == expected
    assert await facade.async_get_cache(model="text-embedding-3-small", input=inputs) is None


@pytest.mark.parametrize(
    ("backend", "settings", "message"),
    [
        pytest.param(
            LiteLLMCacheType.VALKEY_SEMANTIC,
            {"redis_url": "rediss://127.0.0.1:6390/0", "similarity_threshold": 0.8},
            "native Valkey semantic cache does not support TLS connections",
            id="valkey-tls",
        ),
        pytest.param(
            LiteLLMCacheType.VALKEY_SEMANTIC,
            {"redis_url": "redis://127.0.0.1:6390/0?socket_timeout=1", "similarity_threshold": 0.8},
            "native Redis uses fixed socket timeouts; socket_timeout and socket_connect_timeout require Python",
            id="valkey-socket-timeout",
        ),
        pytest.param(
            LiteLLMCacheType.REDIS_SEMANTIC,
            {"redis_url": "rediss://127.0.0.1:6380", "similarity_threshold": 0.8},
            "native Redis semantic cache does not support TLS or query options in redis_url",
            id="redis-semantic-tls",
        ),
        pytest.param(
            LiteLLMCacheType.REDIS_SEMANTIC,
            {"redis_url": "redis://127.0.0.1:6379?socket_timeout=1", "similarity_threshold": 0.8},
            "native Redis semantic cache does not support TLS or query options in redis_url",
            id="redis-semantic-query",
        ),
    ],
)
def test_semantic_settings_the_native_client_cannot_honor_decline(
    monkeypatch: pytest.MonkeyPatch, backend: LiteLLMCacheType, settings: dict[str, object], message: str
) -> None:
    require_rust(monkeypatch, backend)
    with pytest.raises(RuntimeError, match=f"declined the cache: {message}"):
        Cache(type=backend, **settings)


class _SemanticHit:
    """A native semantic runtime that answers every lookup with one cached response."""

    kind: Final = "native"

    def lookup_semantic(self, request: object) -> tuple[object, float | None]:
        return {"answer": 42}, 0.97

    async def async_lookup_semantic(self, request: object) -> tuple[object, float | None]:
        return {"answer": 42}, 0.97


@pytest.mark.parametrize("semantic_type", [LiteLLMCacheType.QDRANT_SEMANTIC, LiteLLMCacheType.REDIS_SEMANTIC])
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_native_semantic_hit_stamps_similarity_on_request_metadata(
    semantic_type: LiteLLMCacheType, use_async: bool
) -> None:
    """Python semantic backends write `metadata["semantic-similarity"]` on every lookup, and the
    facade copies it to the caller's metadata; the native path must report it the same way."""
    facade: Final = Cache()
    facade.type = semantic_type
    facade._native_cache = ResponseCacheRuntime(cast(NativeResponseCacheRuntime, _SemanticHit()))  # pyright: ignore[reportPrivateUsage]  # the native path under test has no public setter
    metadata: Final[dict[str, object]] = {}
    kwargs: Final = {
        "cache_key": "semantic-key",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": metadata,
    }

    result: Final = asyncio.run(facade.async_get_cache(**kwargs)) if use_async else facade.get_cache(**kwargs)

    assert result == {"answer": 42}
    assert metadata["semantic-similarity"] == 0.97
