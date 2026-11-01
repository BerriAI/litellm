import os
import subprocess
import sys
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Final

import pytest
import pytest_asyncio
from redis import Redis

from litellm.caching.redis_cache import RedisCache
from litellm.proxy import proxy_server
from litellm.proxy.utils import PrismaClient
from tests.integration._support.database import scratch_database
from tests.integration._support.prisma_clients import connected_proxy_prisma
from tests.integration._support.redis_process import owned_redis


@pytest.fixture(scope="module")
def key_spend_database() -> Iterator[str]:
    with scratch_database("key_spend") as database_url:
        schema: Final = Path(__file__).resolve().parents[3] / "litellm/proxy/schema.prisma"
        result: Final = subprocess.run(
            [sys.executable, "-I", "-m", "prisma", "db", "push", "--skip-generate", "--schema", str(schema)],
            env={**os.environ, "DATABASE_URL": database_url, "DIRECT_URL": database_url},
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        yield database_url


@pytest_asyncio.fixture(loop_scope="function")
async def buffered_spend_worker(
    key_spend_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[tuple[PrismaClient, "Redis[str]"]]:
    monkeypatch.setattr(proxy_server, "general_settings", {"use_redis_transaction_buffer": True})
    monkeypatch.setenv("DATABASE_URL", key_spend_database)
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    with owned_redis(tmp_path) as redis_server:
        cache: Final = RedisCache(host=redis_server.host, port=redis_server.port)
        try:
            async with connected_proxy_prisma(key_spend_database) as database:
                database.proxy_logging_obj.update_values(  # pyright: ignore[reportUnknownMemberType]  # Untyped alerts
                    redis_cache=cache
                )
                observer_client: Final[Redis[str]] = Redis(
                    host=redis_server.host, port=redis_server.port, decode_responses=True
                )
                with observer_client as observer:
                    yield database, observer
        finally:
            await cache.disconnect()
