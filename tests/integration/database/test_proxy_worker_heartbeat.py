from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import pytest
import pytest_asyncio

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.proxy_worker_heartbeat import ProxyWorkerHeartbeat, count_live_proxy_workers
from litellm.proxy.utils import PrismaClient, ProxyLogging
from tests.integration._support.database import read_rows, scratch_database, write_rows


@pytest_asyncio.fixture(loop_scope="function")
async def heartbeat_database(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[PrismaClient, str]]:
    with scratch_database() as url:
        write_rows(
            'CREATE TABLE "LiteLLM_ProxyWorkerHeartbeat" (worker_id TEXT PRIMARY KEY, hostname TEXT NOT NULL, '
            "started_at TIMESTAMP NOT NULL DEFAULT NOW(), last_heartbeat_at TIMESTAMP NOT NULL DEFAULT NOW())",
            (),
            database_url=url,
        )
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
        client: Final = PrismaClient(url, ProxyLogging(UserApiKeyCache()))
        await client.connect()
        try:
            migration: Final = (
                Path(__file__).resolve().parents[3]
                / "litellm-proxy-extras/litellm_proxy_extras/migrations/20261009120000_proxy_worker_expiry/migration.sql"
            )
            await client.db.execute_raw(migration.read_text())
            await client.db.execute_raw(migration.read_text())
            yield client, url
        finally:
            await client.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(("slow_interval", "elapsed"), ((1800, 600), (7200, 3700)))
async def test_fast_worker_neither_hides_nor_prunes_a_live_slow_worker(
    heartbeat_database: tuple[PrismaClient, str], monkeypatch: pytest.MonkeyPatch, slow_interval: int, elapsed: int
) -> None:
    client, url = heartbeat_database
    monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", str(slow_interval))
    slow: Final = ProxyWorkerHeartbeat(client, worker_id="slow")
    await slow.beat()
    write_rows(
        'UPDATE "LiteLLM_ProxyWorkerHeartbeat" '
        "SET last_heartbeat_at = last_heartbeat_at - make_interval(secs => %s::int), "
        "expires_at = expires_at - make_interval(secs => %s::int)",
        (str(elapsed), str(elapsed)),
        database_url=url,
    )
    monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", "60")
    await ProxyWorkerHeartbeat(client, worker_id="fast").beat()

    assert await count_live_proxy_workers(client) == 2
    assert read_rows(
        'SELECT worker_id FROM "LiteLLM_ProxyWorkerHeartbeat" ORDER BY worker_id', (), database_url=url
    ) == [{"worker_id": "fast"}, {"worker_id": "slow"}]
    monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", str(slow_interval))
    assert await count_live_proxy_workers(client) == 2


@pytest.mark.asyncio
async def test_expired_workers_are_excluded_and_legacy_rows_keep_the_default_window(
    heartbeat_database: tuple[PrismaClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, url = heartbeat_database
    monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", "15")
    await ProxyWorkerHeartbeat(client, worker_id="expired").beat()
    write_rows(
        'UPDATE "LiteLLM_ProxyWorkerHeartbeat" '
        "SET last_heartbeat_at = NOW() - INTERVAL '50 seconds', expires_at = NOW() - INTERVAL '5 seconds'",
        (),
        database_url=url,
    )
    write_rows(
        'INSERT INTO "LiteLLM_ProxyWorkerHeartbeat" (worker_id, hostname, last_heartbeat_at) '
        "VALUES ('legacy-live', 'old-worker', NOW() - INTERVAL '120 seconds'), "
        "('legacy-expired', 'old-worker', NOW() - INTERVAL '181 seconds'), "
        "('legacy-stale', 'old-worker', NOW() - INTERVAL '2 hours')",
        (),
        database_url=url,
    )
    monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", "1800")
    await ProxyWorkerHeartbeat(client, worker_id="current").beat()

    assert await count_live_proxy_workers(client) == 2
    assert read_rows(
        'SELECT worker_id FROM "LiteLLM_ProxyWorkerHeartbeat" ORDER BY worker_id', (), database_url=url
    ) == [
        {"worker_id": "current"},
        {"worker_id": "expired"},
        {"worker_id": "legacy-expired"},
        {"worker_id": "legacy-live"},
    ]
