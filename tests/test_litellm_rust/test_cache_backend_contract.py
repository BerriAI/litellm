import itertools
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from litellm.caching.base_cache import BaseCache
from litellm.caching.caching import Cache
from litellm.caching.disk_cache import DiskCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.response_cache import resolve_response_cache
from litellm.types.caching import LiteLLMCacheType

pytestmark: Final = pytest.mark.requires_rust_extension

Writer = Callable[[str, object], Awaitable[None]]
Reader = Callable[[str], Awaitable[object]]
OVERRIDDEN: Final = {"source": "subclass override"}


@dataclass(frozen=True, slots=True)
class Backend:
    type: LiteLLMCacheType
    cls: type[BaseCache]
    config: Callable[[Path], Mapping[str, str]]
    facade_config: Callable[[Path], Mapping[str, str]]


BACKENDS: Final = (
    pytest.param(Backend(LiteLLMCacheType.LOCAL, InMemoryCache, lambda _: {}, lambda _: {}), id="local"),
    pytest.param(
        Backend(
            LiteLLMCacheType.DISK,
            DiskCache,
            lambda root: {"disk_cache_dir": str(root)},
            lambda root: {"disk_cache_dir": str(root)},
        ),
        id="disk",
    ),
)


@pytest.fixture(params=(Rollout.PYTHON_ONLY, Rollout.RUST_REQUIRED), ids=("python", "native"))
def rollout(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Rollout:
    selected: Final[Rollout] = request.param
    monkeypatch.setattr(catalog, "RULES", (CacheRule(selected),))
    return selected


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


@pytest.mark.usefixtures("rollout")
@pytest.mark.parametrize("backend", BACKENDS)
async def test_every_writer_is_visible_to_every_reader(backend: Backend, tmp_path: Path) -> None:
    facade: Final = Cache(type=backend.type, **backend.facade_config(tmp_path))
    assert await disagreements(paths(facade), lambda value: value) == ()


@pytest.mark.usefixtures("rollout")
@pytest.mark.parametrize("backend", BACKENDS)
async def test_backend_subclass_overrides_win_on_every_reader(backend: Backend, tmp_path: Path) -> None:
    class Overriding(backend.cls):
        def get_cache(self, key: str, **kwargs: object) -> object:
            return OVERRIDDEN

        async def async_get_cache(self, key: str, **kwargs: object) -> object:
            return OVERRIDDEN

    facade: Final = Cache(type=backend.type, **backend.facade_config(tmp_path))
    facade.cache = Overriding(**backend.config(tmp_path))
    assert await disagreements(paths(facade), lambda _value: OVERRIDDEN) == ()


def is_wrapped(response: object) -> bool:
    return isinstance(response, dict) and response.keys() == {"wrapped"}


def wrap_response(entry: object) -> object:
    if not isinstance(entry, dict) or "response" not in entry or is_wrapped(entry["response"]):
        return entry
    return {**entry, "response": {"wrapped": entry["response"]}}


@pytest.mark.usefixtures("rollout")
@pytest.mark.parametrize("backend", BACKENDS)
async def test_backend_subclass_super_calls_reach_the_shared_store(backend: Backend, tmp_path: Path) -> None:
    class Wrapping(backend.cls):
        def set_cache(self, key: str, value: object, **kwargs: object) -> None:
            super().set_cache(key, wrap_response(value), **kwargs)

        async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
            await super().async_set_cache(key, wrap_response(value), **kwargs)

    facade: Final = Cache(type=backend.type, **backend.facade_config(tmp_path))
    facade.cache = Wrapping(**backend.config(tmp_path))
    assert await disagreements(paths(facade), lambda value: {"wrapped": value}) == ()


@pytest.mark.usefixtures("rollout")
async def test_custom_cache_objects_serve_every_path() -> None:
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

    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    facade.cache = DictCache()
    assert await disagreements(paths(facade), lambda value: value) == ()


@pytest.mark.usefixtures("rollout")
async def test_memory_backend_returns_the_stored_object_itself() -> None:
    facade: Final = Cache(type=LiteLLMCacheType.LOCAL)
    stored: Final = object()
    facade.cache.set_cache("identity", stored)
    await facade.cache.async_set_cache("async identity", stored)
    assert facade.cache.get_cache("identity") is stored
    assert await facade.cache.async_get_cache("async identity") is stored
