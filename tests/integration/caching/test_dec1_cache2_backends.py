"""Cache backend settings: removed and unknown types, ignored disk settings, the Redis ttl default and batched writes.

``Cache.__init__`` (``litellm/caching/caching.py``) accepts ``local``, ``redis``, ``redis-semantic``,
``valkey-semantic``, ``s3``, ``qdrant-semantic``, ``azure-blob`` and ``gcs`` (``LiteLLMCacheType``) and raises
``ValueError("Unsupported cache type ...")`` for anything else; the Python ``disk`` backend was removed, and a
``disk_cache_dir`` given to another type is ignored with a warning (``caching.py:340``). The Rust side has a
``cache-disk`` crate, so a Rust config loader may be tempted to accept ``type: disk`` again; these cells pin
what the Python proxy does today so that change is deliberate.

``default_in_redis_ttl`` becomes the default entry ttl of a Redis cache, and ``redis_flush_size`` turns the
write into a per-worker buffer flushed to Redis once it holds that many entries (``RedisCache.batch_cache_write``,
``redis_cache.py:1366``). Every cell boots its own single-worker proxy.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, held
from tests.integration._support.redis_process import owned_redis
from tests.integration.caching.dec1_cache2_support import (
    CHAT_ROUTE,
    DEFAULT_ENTRY_TTL_SECONDS,
    entry_ttl,
    route_probe,
)
from tests.integration.caching.dec1_cache_support import (
    CacheRig,
    await_named_entry,
    cache_config,
    cache_rig,
    entry_keys_naming,
    redis_params,
)
from tests.integration.caching.response_cache_case import prompt

pytestmark: Final = [pytest.mark.shares_worker_state, pytest.mark.timeout(300)]

REDIS_TTL: Final = 90
FLUSH_SIZE: Final = 2


@pytest.fixture(scope="module")
def redis_defaults_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CacheRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-cache2-redis-defaults")
    config: Final = cache_config(
        directory,
        cache_params=redis_params(default_in_redis_ttl=REDIS_TTL, disk_cache_dir=str(directory / "unused-disk")),
    )
    with owned_redis(directory) as redis, cache_rig(session_gateway, directory, config, redis=redis) as rig:
        yield rig


@pytest.fixture(scope="module")
def batched_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CacheRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-cache2-batched")
    config: Final = cache_config(directory, cache_params=redis_params(redis_flush_size=FLUSH_SIZE))
    with owned_redis(directory) as redis, cache_rig(session_gateway, directory, config, redis=redis) as rig:
        yield rig


def test_a_disk_cache_dir_on_a_redis_cache_is_ignored_and_redis_still_caches(redis_defaults_rig: CacheRig) -> None:
    """The proxy runs a Redis cache with ``disk_cache_dir`` set. Two identical chat requests must give one fill and
    one hit and the entry must be in the owned Redis: the setting is ignored with a warning, never a reason to
    fail the start or to switch backends.
    """
    with redis_defaults_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        first, _ = target.fill(redis_defaults_rig.store(), text)
        target.assert_hit(first, target.ask(text))


def test_default_in_redis_ttl_sets_the_entry_expiry(redis_defaults_rig: CacheRig) -> None:
    """On the proxy with ``default_in_redis_ttl: 90`` a plain chat request fills an entry. Its Redis TTL must be
    above the 60 second fallback and at most 90 (``caching.py:368``).
    """
    with redis_defaults_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        _, key = target.fill(redis_defaults_rig.store(), prompt())
        assert DEFAULT_ENTRY_TTL_SECONDS < entry_ttl(redis_defaults_rig.store(), key) <= REDIS_TTL


def test_a_request_ttl_still_overrides_default_in_redis_ttl(redis_defaults_rig: CacheRig) -> None:
    """On the same proxy a request with ``cache: {"ttl": 30}`` fills an entry. Its TTL must be at most 30: the
    request control wins over the configured default.
    """
    with redis_defaults_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        _, key = target.fill(redis_defaults_rig.store(), prompt(), cache={"ttl": 30})
        assert 0 < entry_ttl(redis_defaults_rig.store(), key) <= 30


def test_a_batched_write_stays_out_of_redis_until_the_buffer_is_full(batched_rig: CacheRig) -> None:
    """On the proxy with ``redis_flush_size: 2`` one chat request reaches the upstream. For the next five seconds no
    Redis key may name its answer: the write sits in the worker's buffer until a second write fills it.
    """
    with batched_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        first: Final = target.ask(prompt())
        target.assert_miss(first)
        held(
            lambda: entry_keys_naming(batched_rig.store(), first.id()),
            lambda found: found == frozenset(),
            holding=5,
        )


def test_a_batched_entry_is_flushed_and_served_once_the_buffer_is_full(batched_rig: CacheRig) -> None:
    """On the same proxy prompt A is asked once (buffered), then prompt B once (the second write fills the buffer
    of size 2 and flushes both). A then must be a hit with the first answer's id, read back from Redis.
    """
    with batched_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text_a: Final = prompt()
        first: Final = target.ask(text_a)
        target.assert_miss(first)
        target.assert_miss(target.ask(prompt()))
        await_named_entry(batched_rig.store(), target.needle(first))
        target.assert_hit(first, target.ask(text_a))


@pytest.mark.parametrize(
    "cache_type", (pytest.param("disk", id="removed-disk"), pytest.param("dec1-unknown", id="unknown"))
)
def test_an_unsupported_cache_type_fails_the_proxy_start(
    session_gateway: Gateway, tmp_path: Path, cache_type: str
) -> None:
    """A proxy configured with ``cache_params.type`` ``disk`` (removed) or an unknown name must not come up
    serving requests: ``Cache.__init__`` raises ``ValueError("Unsupported cache type ...")`` while the config
    loads. A Rust config loader must refuse the same configs instead of silently running without a cache.
    UNVERIFIED: how the start failure surfaces through ``owned_proxy_process`` (an exception when the health
    check never passes is assumed).
    """
    config: Final = cache_config(tmp_path, cache_params={"type": cache_type})
    with pytest.raises(Exception), cache_rig(session_gateway, tmp_path, config):  # noqa: B017
        pass
