from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

from tests.integration._support.database import scratch_database
from tests.integration._support.redis_process import owned_redis


@contextmanager
def worker_services(directory: Path) -> Iterator[Mapping[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    with (
        scratch_database() as database,
        owned_redis(directory) as cache,
    ):
        yield MappingProxyType(
            {
                "DATABASE_URL": database,
                "INTEGRATION_PROXY_DATABASE_URL": database,
                "INTEGRATION_PROXY_READ_REPLICA_URL": "",
                "REDIS_HOST": cache.host,
                "REDIS_PORT": str(cache.port),
            }
        )
