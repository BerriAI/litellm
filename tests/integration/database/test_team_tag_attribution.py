import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

import psycopg
import pytest
from integration._support.database import read_rows, scratch_database
from prisma import Prisma

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.utils import PrismaClient

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
PRISMA_DIR: Final = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
NEW_MIGRATIONS: Final = (
    "20261009060000_add_team_id_to_daily_tag_spend",
    "20261009060100_daily_tag_spend_team_unique_index",
    "20261009060200_daily_tag_spend_team_date_tag_index",
    "20261009060300_drop_daily_tag_spend_old_unique_index",
)
TEAM_A: Final = "team-a"
TEAM_B: Final = "team-b"
TAG: Final = "shared-tag"
TODAY: Final = "2026-10-08"


class _Client:
    """Real generated Prisma engine plus the real request-status derivation."""

    def __init__(self, db: Prisma):
        self.db = db

    def get_request_status(self, payload):
        return PrismaClient.get_request_status(self, payload)


def _deploy_migrations(database_url: str, *, exclude: tuple[str, ...] = ()) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target: Final = Path(tmp)
        (target / "migrations").mkdir(parents=True)
        shutil.copy(PRISMA_DIR / "schema.prisma", target / "schema.prisma")
        shutil.copy(PRISMA_DIR / "migrations" / "migration_lock.toml", target / "migrations" / "migration_lock.toml")
        for path in (PRISMA_DIR / "migrations").iterdir():
            if path.is_dir() and path.name not in exclude:
                shutil.copytree(path, target / "migrations" / path.name)
        subprocess.run(
            [sys.executable, "-m", "prisma", "migrate", "deploy", "--schema", str(target / "schema.prisma")],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "DATABASE_URL": database_url},
        )


def _payload(*, team_id: str, tags: list[str], api_key: str, spend: float, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "request_tags": tags,
        "team_id": team_id,
        "user": "user-1",
        "startTime": f"{TODAY}T12:00:00",
        "api_key": api_key,
        "model": "model-a",
        "model_group": "model-a",
        "custom_llm_provider": "openai",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "spend": spend,
        "metadata": '{"usage_object": {}}',
    }


def _tag_rows(url: str) -> list[dict]:
    return read_rows(
        "SELECT team_id, tag, api_key, spend, api_requests, prompt_tokens, completion_tokens "
        'FROM "LiteLLM_DailyTagSpend" ORDER BY team_id, tag',
        (),
        database_url=url,
    )


@pytest.fixture
def writer_db():
    with scratch_database() as url:
        _deploy_migrations(url)
        yield url


@asynccontextmanager
async def _writer_client(url: str):
    prisma: Final = Prisma(datasource={"url": url})
    await prisma.connect()
    try:
        yield _Client(prisma)
    finally:
        await prisma.disconnect()


class _LoggingStub:
    async def failure_handler(self, *args, **kwargs):
        return None


async def _commit_in_memory(client: _Client, writer: DBSpendUpdateWriter) -> None:
    await writer.commit_daily_tag_spend_to_db(prisma_client=client, n_retry_times=3, proxy_logging_obj=_LoggingStub())


@pytest.mark.asyncio
async def test_two_teams_same_tag_make_two_team_rows(writer_db) -> None:
    url: Final = writer_db
    async with _writer_client(url) as client:
        writer: Final = DBSpendUpdateWriter()
        for team_id in (TEAM_A, TEAM_B):
            await writer.add_spend_log_transaction_to_daily_tag_transaction(
                payload=_payload(
                    team_id=team_id, tags=[TAG], api_key=f"key-{team_id}", spend=1.0, request_id=f"r-{team_id}"
                ),
                prisma_client=client,
            )
        await _commit_in_memory(client, writer)
    rows: Final = _tag_rows(url)
    assert {(row["team_id"], row["tag"]) for row in rows} == {(TEAM_A, TAG), (TEAM_B, TAG)}, rows
    assert all(float(row["spend"]) == 1.0 for row in rows), rows


@pytest.mark.asyncio
async def test_redis_buffer_path_writes_two_team_rows(writer_db) -> None:
    url: Final = writer_db
    redis_cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    async with _writer_client(url) as client:
        writer: Final = DBSpendUpdateWriter()
        for team_id in (TEAM_A, TEAM_B):
            await writer.add_spend_log_transaction_to_daily_tag_transaction(
                payload=_payload(
                    team_id=team_id, tags=[TAG], api_key=f"key-{team_id}", spend=1.0, request_id=f"r-{team_id}"
                ),
                prisma_client=client,
            )
        await buffer.store_in_memory_daily_tag_spend_updates_in_redis(
            daily_tag_spend_update_queue=writer.daily_tag_spend_update_queue
        )
        drained: Final = await buffer.get_all_daily_tag_spend_update_transactions_from_redis_buffer()
        assert drained, "redis buffer returned no tag transactions"
        await DBSpendUpdateWriter.update_daily_tag_spend(
            n_retry_times=3,
            prisma_client=client,
            proxy_logging_obj=_LoggingStub(),
            daily_spend_transactions=drained,
        )
    rows: Final = _tag_rows(url)
    assert {(row["team_id"], row["tag"]) for row in rows} == {(TEAM_A, TAG), (TEAM_B, TAG)}, rows


@pytest.mark.asyncio
async def test_legacy_transaction_without_team_id_writes_empty(writer_db) -> None:
    url: Final = writer_db
    legacy_transaction: Final = {
        "tag": TAG,
        "date": TODAY,
        "api_key": "key-legacy",
        "model": "model-a",
        "model_group": "model-a",
        "custom_llm_provider": "openai",
        "mcp_namespaced_tool_name": None,
        "endpoint": None,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "spend": 0.5,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "total_response_time_ms": 0,
        "timed_requests": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "compression_saved_tokens": 0,
        "compression_savings_spend": 0.0,
        "prompt_caching_savings_spend": 0.0,
        "gateway_injected_caching_savings_spend": 0.0,
        "autorouter_savings_spend": 0.0,
        "request_id": "r-legacy",
    }
    assert "team_id" not in legacy_transaction
    async with _writer_client(url) as client:
        await DBSpendUpdateWriter.update_daily_tag_spend(
            n_retry_times=3,
            prisma_client=client,
            proxy_logging_obj=_LoggingStub(),
            daily_spend_transactions={"legacy": legacy_transaction},
        )
    rows: Final = _tag_rows(url)
    assert len(rows) == 1 and rows[0]["team_id"] == "", rows
    assert float(rows[0]["spend"]) == 0.5, rows


@pytest.mark.asyncio
async def test_duplicate_tags_on_one_request_count_once(writer_db) -> None:
    url: Final = writer_db
    async with _writer_client(url) as client:
        writer: Final = DBSpendUpdateWriter()
        await writer.add_spend_log_transaction_to_daily_tag_transaction(
            payload=_payload(team_id=TEAM_A, tags=[TAG, TAG], api_key="key-a", spend=1.0, request_id="r-dup"),
            prisma_client=client,
        )
        await _commit_in_memory(client, writer)
    rows: Final = _tag_rows(url)
    assert len(rows) == 1, rows
    assert int(rows[0]["api_requests"]) == 1, rows


@pytest.mark.asyncio
async def test_concurrent_updates_to_same_bucket_sum(writer_db) -> None:
    url: Final = writer_db
    async with _writer_client(url) as client:
        writers: Final = (DBSpendUpdateWriter(), DBSpendUpdateWriter())
        for writer, spend in zip(writers, (1.0, 2.0)):
            await writer.add_spend_log_transaction_to_daily_tag_transaction(
                payload=_payload(team_id=TEAM_A, tags=[TAG], api_key="key-a", spend=spend, request_id=f"r-{spend}"),
                prisma_client=client,
            )
        await asyncio.gather(*(_commit_in_memory(client, writer) for writer in writers))
    rows: Final = _tag_rows(url)
    assert len(rows) == 1, rows
    assert float(rows[0]["spend"]) == 3.0 and int(rows[0]["api_requests"]) == 2, rows


def test_migrations_on_populated_old_schema_preserve_rows() -> None:
    with scratch_database() as url:
        _deploy_migrations(url, exclude=NEW_MIGRATIONS)
        seeded: Final = (
            ("seed-1", "seed-1", TAG, TODAY, "key-1", "model-a", "openai", None, None, 10, 5, 0.5, 1, 1, 0, 0, 0),
            ("seed-2", "seed-2", TAG, TODAY, "key-2", "model-a", "openai", None, None, 20, 6, 0.7, 1, 1, 0, 0, 0),
        )
        with psycopg.connect(url, autocommit=True) as connection:
            for row in seeded:
                connection.execute(
                    'INSERT INTO "LiteLLM_DailyTagSpend" (id, request_id, tag, date, api_key, model, '
                    "custom_llm_provider, mcp_namespaced_tool_name, endpoint, prompt_tokens, completion_tokens, "
                    "spend, api_requests, successful_requests, failed_requests, total_response_time_ms, timed_requests, "
                    "created_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now())",
                    row,
                )
        before: Final = read_rows(
            "SELECT count(*)::float8 AS cnt, sum(spend)::float8 AS spend, "
            'sum(prompt_tokens)::float8 AS pt FROM "LiteLLM_DailyTagSpend"',
            (),
            database_url=url,
        )[0]
        _deploy_migrations(url)
        rows: Final = read_rows(
            "SELECT count(*)::float8 AS cnt, sum(spend)::float8 AS spend, "
            "sum(prompt_tokens)::float8 AS pt, "
            "count(*) FILTER (WHERE team_id <> '')::float8 AS non_empty, "
            "count(*) FILTER (WHERE team_id = '')::float8 AS empty_team "
            'FROM "LiteLLM_DailyTagSpend"',
            (),
            database_url=url,
        )[0]
        assert int(rows["cnt"]) == int(before["cnt"]) == 2
        assert float(rows["spend"]) == float(before["spend"]) == 1.2
        assert int(rows["pt"]) == int(before["pt"]) == 30
        assert int(rows["empty_team"]) == 2 and int(rows["non_empty"]) == 0, rows
        indexes: Final = read_rows(
            "SELECT i.indexname, x.indisvalid FROM pg_indexes i "
            "JOIN pg_class c ON c.relname = i.indexname JOIN pg_index x ON x.indexrelid = c.oid "
            "WHERE i.tablename = 'LiteLLM_DailyTagSpend' AND position('team' in i.indexname) > 0",
            (),
            database_url=url,
        )
        assert {(row["indexname"], row["indisvalid"]) for row in indexes} == {
            ("LiteLLM_DailyTagSpend_tag_team_dimensions_key", True),
            ("LiteLLM_DailyTagSpend_team_id_date_tag_idx", True),
        }, indexes
        old: Final = read_rows(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'LiteLLM_DailyTagSpend' "
            "AND indexname = 'LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key'",
            (),
            database_url=url,
        )
        assert old == [], old
