from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.roi_calculator.sample import sample_report
from litellm.proxy.roi_calculator.sync_store import SyncStore
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.types.roi_calculator import ROIPullRecord, ROIReport, ROISyncStatus
from tests.integration._support.database import read_rows, scratch_database, write_rows


@pytest.mark.asyncio
async def test_roi_cache_survives_scope_changes_and_uses_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    with scratch_database() as writer_url, scratch_database() as reader_url:
        write_rows(
            'CREATE TABLE "LiteLLM_Config" (param_name TEXT PRIMARY KEY, param_value JSONB NOT NULL, '
            "last_run_at TIMESTAMP NOT NULL DEFAULT NOW())",
            (),
            database_url=writer_url,
        )
        monkeypatch.setenv("DATABASE_URL", writer_url)
        # The reader deliberately has no table: any accidental replica read fails
        monkeypatch.setenv("DATABASE_URL_READ_REPLICA", reader_url)
        client: Final = PrismaClient(writer_url, ProxyLogging(UserApiKeyCache()))
        await client.connect()
        try:
            store: Final = SyncStore(client)
            report: Final = sample_report(datetime(2026, 9, 30, tzinfo=timezone.utc))
            pull: Final[ROIPullRecord] = {
                **report["pulls"][0],
                "url": "https://github.com/example/repo/pull/1",
                "cache_key": "new",
            }
            for key, url in (("old", pull["url"]), ("new", pull["url"]), ("outside-window", "other-pr")):
                value: ROIPullRecord = {**pull, "url": url, "cache_key": key}
                write_rows(
                    'INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES (%s, %s::jsonb)',
                    (f"roi_calculator_pull_{key}", TypeAdapter(ROIPullRecord).dump_json(value).decode()),
                    database_url=writer_url,
                )
            running: Final = ROISyncStatus(
                running=True,
                phase="estimates",
                stage="Estimating",
                done=0,
                total=1,
                estimated=0,
                reused=0,
                needs_attention=0,
                error=None,
            )
            complete: Final = running.model_copy(update=MappingProxyType({"running": False, "phase": "complete"}))
            narrowed: Final[ROIReport] = {**report, "pulls": (pull,)}
            empty: Final[ROIReport] = {**report, "pulls": ()}
            assert await store.acquire("worker", running)
            assert not await store.acquire("other-worker", running)
            observed: Final = await store.status()
            assert observed is not None and observed.running
            assert await store.heartbeat("worker", running)
            assert await store.finish("worker", complete, narrowed)
            assert tuple(
                row["param_name"]
                for row in read_rows(
                    'SELECT param_name FROM "LiteLLM_Config" WHERE starts_with(param_name, %s) ORDER BY param_name',
                    ("roi_calculator_pull_",),
                    database_url=writer_url,
                )
            ) == ("roi_calculator_pull_new", "roi_calculator_pull_outside-window")
            assert not await store.acquire("scheduled", running, 1440)
            assert await store.acquire("manual", running)
            write_rows(
                "UPDATE \"LiteLLM_Config\" SET last_run_at = NOW() - INTERVAL '2 minutes' WHERE param_name = %s",
                ("roi_calculator_sync",),
                database_url=writer_url,
            )
            expired: Final = await store.status()
            assert expired is not None and expired.phase == "error" and expired.finished_at is not None
            assert datetime.fromisoformat(expired.finished_at).tzinfo == timezone.utc
            assert not await store.heartbeat("manual", running)
            assert await store.acquire("replacement", running)
            assert not await store.finish("manual", complete, empty)
            assert await store.finish("replacement", complete, empty)
            assert (
                len(
                    read_rows(
                        'SELECT param_name FROM "LiteLLM_Config" WHERE starts_with(param_name, %s)',
                        ("roi_calculator_pull_",),
                        database_url=writer_url,
                    )
                )
                == 2
            )
        finally:
            await client.disconnect()
