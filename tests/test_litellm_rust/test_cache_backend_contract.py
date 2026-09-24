import itertools
import os
import subprocess
import sys
import threading
from collections.abc import Awaitable, Callable, Generator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from unittest.mock import patch
from uuid import uuid4

import fakeredis
import pytest

import litellm
from litellm.caching.base_cache import BaseCache
from litellm.caching.caching import Cache
from litellm.caching.disk_cache import DiskCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.caching.s3_cache import S3Cache
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.response_cache import resolve_response_cache
from litellm.types.caching import LiteLLMCacheType
from litellm.types.utils import Choices, ModelResponse
from tests.test_litellm_rust.support.s3_stub import S3Stub

pytestmark: Final = pytest.mark.requires_rust_extension

Writer = Callable[[str, object], Awaitable[None]]
Reader = Callable[[str], Awaitable[object]]
Settings = Callable[[pytest.FixtureRequest], Mapping[str, object]]
OVERRIDDEN: Final = {"source": "override"}


@dataclass(frozen=True, slots=True)
class Backend:
    type: LiteLLMCacheType
    cls: type[BaseCache]
    settings: Settings
    backend_settings: Settings


def redis_settings(request: pytest.FixtureRequest) -> Mapping[str, object]:
    host, port = request.getfixturevalue("redis_address")
    return {"host": host, "port": port}


def s3_settings(request: pytest.FixtureRequest) -> Mapping[str, object]:
    stub: Final[S3Stub] = request.getfixturevalue("s3_stub")
    return {
        "s3_bucket_name": "cache-bucket",
        "s3_region_name": "us-east-1",
        "s3_endpoint_url": stub.url,
        "s3_aws_access_key_id": "key",
        "s3_aws_secret_access_key": "secret",
        "s3_path": "team",
    }


def disk_settings(request: pytest.FixtureRequest) -> Mapping[str, object]:
    return {"disk_cache_dir": str(request.getfixturevalue("tmp_path"))}


BACKENDS: Final = (
    Backend(LiteLLMCacheType.LOCAL, InMemoryCache, lambda _: {}, lambda _: {}),
    Backend(LiteLLMCacheType.DISK, DiskCache, disk_settings, disk_settings),
    Backend(LiteLLMCacheType.REDIS, RedisCache, redis_settings, redis_settings),
    Backend(LiteLLMCacheType.S3, S3Cache, s3_settings, s3_settings),
)
EXCLUDED_BACKENDS: Final = MappingProxyType(
    {
        LiteLLMCacheType.GCS: "Python GCSCache has no configurable endpoint, so it cannot reach an in-process fake",
        LiteLLMCacheType.AZURE_BLOB: "no in-process Azure Blob fake yet",
        LiteLLMCacheType.REDIS_SEMANTIC: "fakeredis lacks the vector search commands the backend sends",
        LiteLLMCacheType.VALKEY_SEMANTIC: "fakeredis lacks the vector search commands the backend sends",
        LiteLLMCacheType.QDRANT_SEMANTIC: "no in-process Qdrant fake yet",
    }
)
BACKEND_PARAMS: Final = tuple(pytest.param(backend, id=backend.type.value) for backend in BACKENDS)

SEPARATE_STORES: Final = "Cache and the response-cache runtime keep separate local stores"
OVERRIDES_BYPASSED: Final = (
    "the response-cache runtime reads storage directly instead of calling overridden Cache methods"
)
STORAGE_OBJECT_IGNORED: Final = "native storage ignores an object assigned to Cache.cache"
PER_CALL_OBJECT_IGNORED: Final = "native writes ignore the per-call storage object"
FORKED_CHILD_LOSES_ENTRIES: Final = "a forked child loses entries written through the native cache"
BACKEND_IDS: Final = tuple(backend.type.value for backend in BACKENDS)
NATIVE_GAPS: Final = MappingProxyType(
    {
        ("test_every_writer_is_visible_to_every_reader", "local"): SEPARATE_STORES,
        **{
            ("test_cache_subclass_overrides_run_on_every_read_and_super_reaches_the_store", backend): OVERRIDES_BYPASSED
            for backend in BACKEND_IDS
        },
        **{
            ("test_backend_subclass_overrides_win_on_every_reader", backend): STORAGE_OBJECT_IGNORED
            for backend in BACKEND_IDS
        },
        **{
            ("test_backend_subclass_super_calls_reach_the_shared_store", backend): STORAGE_OBJECT_IGNORED
            for backend in BACKEND_IDS
        },
        ("test_custom_storage_backend_serves_every_path", None): STORAGE_OBJECT_IGNORED,
        ("test_patched_public_methods_win_on_every_reader", None): OVERRIDES_BYPASSED,
        ("test_per_call_storage_object_replaces_the_configured_backend", None): PER_CALL_OBJECT_IGNORED,
        ("test_a_forked_child_reads_inherited_entries_and_writes_its_own", None): FORKED_CHILD_LOSES_ENTRIES,
    }
)


@pytest.fixture
def redis_address() -> Generator[tuple[str, int]]:
    server: Final = fakeredis.TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    worker: Final = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield ("127.0.0.1", server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.fixture
def s3_stub() -> Generator[S3Stub]:
    stub: Final = S3Stub()
    try:
        yield stub
    finally:
        stub.close()


@pytest.fixture(params=(Rollout.PYTHON_ONLY, Rollout.RUST_REQUIRED), ids=("python", "native"))
def rollout(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Rollout:
    selected: Final[Rollout] = request.param
    monkeypatch.setattr(catalog, "RULES", (CacheRule(selected),))
    return selected


@pytest.fixture(autouse=True)
def expected_native_gap(request: pytest.FixtureRequest) -> None:
    params: Final[Mapping[str, object]] = getattr(getattr(request.node, "callspec", None), "params", {})
    if params.get("rollout") is not Rollout.RUST_REQUIRED:
        return
    backend: Final = params.get("backend")
    backend_id: Final = backend.type.value if isinstance(backend, Backend) else None
    reason: Final = NATIVE_GAPS.get((request.node.originalname, backend_id))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(strict=True, reason=f"native: {reason}"))


@dataclass(frozen=True, slots=True)
class Paths:
    writers: Mapping[str, Writer]
    readers: Mapping[str, Reader]


def paths(facade: Cache) -> Paths:
    async def facade_add(key: str, value: object) -> None:
        facade.add_cache(value, cache_key=key)

    async def facade_async_add(key: str, value: object) -> None:
        await facade.async_add_cache(value, cache_key=key)

    async def facade_get(key: str) -> object:
        return facade.get_cache(cache_key=key)

    async def facade_async_get(key: str) -> object:
        return await facade.async_get_cache(cache_key=key)

    facade_writers: Final = {"facade.add_cache": facade_add, "facade.async_add_cache": facade_async_add}
    facade_readers: Final = {"facade.get_cache": facade_get, "facade.async_get_cache": facade_async_get}
    runtime: Final = resolve_response_cache(facade)
    if runtime is None:
        return Paths(facade_writers, facade_readers)

    async def native_store(key: str, value: object) -> None:
        runtime.store(runtime.request(facade, {"cache_key": key}), value)

    async def native_async_store(key: str, value: object) -> None:
        await runtime.async_store(runtime.request(facade, {"cache_key": key}), value)

    async def native_lookup(key: str) -> object:
        return runtime.lookup(runtime.request(facade, {"cache_key": key}))

    async def native_async_lookup(key: str) -> object:
        return await runtime.async_lookup(runtime.request(facade, {"cache_key": key}))

    return Paths(
        {**facade_writers, "runtime.store": native_store, "runtime.async_store": native_async_store},
        {**facade_readers, "runtime.lookup": native_lookup, "runtime.async_lookup": native_async_lookup},
    )


async def disagreements(paths: Paths, expected: Callable[[object], object]) -> tuple[str, ...]:
    async def check(index: int, writer: str, reader: str) -> str | None:
        key: Final = f"{writer}->{reader}"
        value: Final = {"written_by": writer, "index": index}
        await paths.writers[writer](key, value)
        seen: Final = await paths.readers[reader](key)
        return None if seen == expected(value) else f"{key}: saw {seen!r}"

    results: Final = [
        await check(index, writer, reader)
        for index, (writer, reader) in enumerate(itertools.product(paths.writers, paths.readers))
    ]
    return tuple(result for result in results if result is not None)


def serve_override(*_args: object, **_kwargs: object) -> object:
    return OVERRIDDEN


async def async_serve_override(*_args: object, **_kwargs: object) -> object:
    return OVERRIDDEN


def is_wrapped(response: object) -> bool:
    return isinstance(response, dict) and response.keys() == {"wrapped"}


def wrap_response(entry: object) -> object:
    if not isinstance(entry, dict) or "response" not in entry or is_wrapped(entry["response"]):
        return entry
    return {**entry, "response": {"wrapped": entry["response"]}}


def test_every_cache_type_has_a_contract_row_or_a_recorded_exclusion() -> None:
    covered: Final = {backend.type for backend in BACKENDS}
    assert covered.isdisjoint(EXCLUDED_BACKENDS)
    assert covered | EXCLUDED_BACKENDS.keys() == set(LiteLLMCacheType)


@pytest.mark.usefixtures("rollout")
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
async def test_every_writer_is_visible_to_every_reader(backend: Backend, request: pytest.FixtureRequest) -> None:
    facade: Final = Cache(type=backend.type, **backend.settings(request))
    assert await disagreements(paths(facade), lambda value: value) == ()


@pytest.mark.usefixtures("rollout")
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
async def test_cache_subclass_overrides_run_on_every_read_and_super_reaches_the_store(
    backend: Backend, request: pytest.FixtureRequest
) -> None:
    reads: Final[list[str]] = []  # mutable-ok: the override records each read it serves

    class AuditedCache(Cache):
        def get_cache(self, *args: object, **kwargs: object) -> object:
            reads.append("sync")
            return super().get_cache(*args, **kwargs)

        async def async_get_cache(self, *args: object, **kwargs: object) -> object:
            reads.append("async")
            return await super().async_get_cache(*args, **kwargs)

    facade: Final = AuditedCache(type=backend.type, **backend.settings(request))
    contract: Final = paths(facade)
    assert await disagreements(contract, lambda value: value) == ()
    assert len(reads) == len(contract.writers) * len(contract.readers)


@pytest.mark.usefixtures("rollout")
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
async def test_backend_subclass_overrides_win_on_every_reader(backend: Backend, request: pytest.FixtureRequest) -> None:
    class Overriding(backend.cls):
        def get_cache(self, key: str, **kwargs: object) -> object:
            return {"timestamp": 0, "response": OVERRIDDEN}

        async def async_get_cache(self, key: str, **kwargs: object) -> object:
            return {"timestamp": 0, "response": OVERRIDDEN}

    facade: Final = Cache(type=backend.type, **backend.settings(request))
    facade.cache = Overriding(**backend.backend_settings(request))
    assert await disagreements(paths(facade), lambda _value: OVERRIDDEN) == ()


@pytest.mark.usefixtures("rollout")
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
async def test_backend_subclass_super_calls_reach_the_shared_store(
    backend: Backend, request: pytest.FixtureRequest
) -> None:
    class Wrapping(backend.cls):
        def set_cache(self, key: str, value: object, **kwargs: object) -> None:
            super().set_cache(key, wrap_response(value), **kwargs)

        async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
            await super().async_set_cache(key, wrap_response(value), **kwargs)

    facade: Final = Cache(type=backend.type, **backend.settings(request))
    facade.cache = Wrapping(**backend.backend_settings(request))
    assert await disagreements(paths(facade), lambda value: {"wrapped": value}) == ()


@pytest.mark.usefixtures("rollout")
async def test_custom_storage_backend_serves_every_path() -> None:
    class DictCache(BaseCache):
        def __init__(self) -> None:
            super().__init__()
            self.entries: dict[str, object] = {}  # mutable-ok: a user-defined cache is a mutable store by design

        def set_cache(self, key: str, value: object, **kwargs: object) -> None:
            self.entries[key] = value

        async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
            self.set_cache(key, value)

        def get_cache(self, key: str, **kwargs: object) -> object:
            return self.entries.get(key)

        async def async_get_cache(self, key: str, **kwargs: object) -> object:
            return self.get_cache(key)

        async def async_set_cache_pipeline(self, cache_list: object, **kwargs: object) -> None:
            raise NotImplementedError

    storage: Final = DictCache()
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    facade.cache = storage
    contract: Final = paths(facade)
    assert await disagreements(contract, lambda value: value) == ()
    assert len(storage.entries) == len(contract.writers) * len(contract.readers)


@pytest.mark.usefixtures("rollout")
async def test_patched_public_methods_win_on_every_reader() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    with (
        patch.object(Cache, "get_cache", serve_override),
        patch.object(Cache, "async_get_cache", async_serve_override),
    ):
        assert await disagreements(paths(facade), lambda _value: OVERRIDDEN) == ()


@pytest.mark.usefixtures("rollout")
async def test_per_call_storage_object_replaces_the_configured_backend() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    override: Final = InMemoryCache()

    await facade.async_add_cache({"source": "override"}, cache_key="key", dynamic_cache_object=override)

    assert facade.get_cache(cache_key="key", dynamic_cache_object=override) == {"source": "override"}
    assert await facade.async_get_cache(cache_key="key", dynamic_cache_object=override) == {"source": "override"}
    assert facade.get_cache(cache_key="key") is None
    assert await facade.async_get_cache(cache_key="key") is None


def reply(response: object) -> object:
    assert isinstance(response, ModelResponse)
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices)
    return choice.message.content


@pytest.mark.usefixtures("rollout")
def test_reassigning_the_global_cache_takes_effect_on_the_next_call() -> None:
    messages: Final = [{"role": "user", "content": f"contract {uuid4().hex}"}]

    def call(mock: str) -> object:
        return reply(litellm.completion(model="gpt-4o-mini", messages=messages, mock_response=mock, caching=True))

    litellm.enable_cache(type=LiteLLMCacheType.LOCAL)
    original: Final = litellm.cache
    replacement: Final = Cache(type=LiteLLMCacheType.LOCAL)
    stored: Final = (call("first"), call("second"))
    litellm.cache = replacement  # test-quality-ok: reassignment is the behavior under test; conftest restores it
    replaced: Final = (call("third"), call("fourth"))
    litellm.disable_cache()
    disabled: Final = (call("fifth"), call("sixth"))
    litellm.cache = original  # test-quality-ok: reassignment is the behavior under test; conftest restores it
    restored: Final = call("seventh")

    assert stored == ("first", "first")
    assert replaced == ("third", "third")
    assert disabled == ("fifth", "sixth")
    assert restored == "first"


FORK_SCRIPT: Final = """
import os
import sys
from litellm.caching.caching import Cache
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout

catalog.RULES = (CacheRule(Rollout[sys.argv[1]]),)
cache = Cache(type="local")
cache.add_cache({"answer": "parent"}, cache_key="parent-key")
read_fd, write_fd = os.pipe()
child = os.fork()
if child == 0:
    os.close(read_fd)
    cache.add_cache({"answer": "child"}, cache_key="child-key")
    seen = (cache.get_cache(cache_key="parent-key"), cache.get_cache(cache_key="child-key"))
    os.write(write_fd, repr(seen).encode())
    os._exit(0)
os.close(write_fd)
print(os.read(read_fd, 4096).decode())
os.waitpid(child, 0)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_a_forked_child_reads_inherited_entries_and_writes_its_own(rollout: Rollout) -> None:
    completed: Final = subprocess.run(
        [sys.executable, "-P", "-c", FORK_SCRIPT, rollout.name],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        env={**os.environ, "LITELLM_RUST": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    assert completed.stdout.strip() == "({'answer': 'parent'}, {'answer': 'child'})"


@pytest.mark.usefixtures("rollout")
async def test_memory_backend_returns_the_stored_object_itself() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    stored: Final = object()
    facade.cache.set_cache("identity", stored)
    await facade.cache.async_set_cache("async identity", stored)
    assert facade.cache.get_cache("identity") is stored
    assert await facade.cache.async_get_cache("async identity") is stored
