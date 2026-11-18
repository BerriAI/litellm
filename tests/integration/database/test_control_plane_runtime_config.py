import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from pydantic import SecretStr
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateSchema, DropSchema

from litellm.proxy.control_plane.app import create_app
from litellm.proxy.control_plane.container import build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from litellm.proxy.control_plane.store.migrate import (
    ADVISORY_LOCK,
    VERSION_TABLE,
    database_url,
    engine_for,
    migrate,
    migration_config,
)
from litellm.proxy.control_plane.store.models import Change, Entry
from litellm.proxy.control_plane.store.postgres import PostgresStore
from litellm.proxy.control_plane.store.schema import dataplanes, metadata
from litellm.proxy.control_plane.store.stream import ConfigStream, RedisStream

MASTER_KEY: Final = "01234567890123456789012345678901"


class Licensed:
    def is_premium(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class Services:
    url: SecretStr
    redis_url: SecretStr
    stream_name: str


@pytest_asyncio.fixture
async def services() -> AsyncIterator[Services]:
    url: Final = os.environ.get("CONTROL_PLANE_TEST_DATABASE_URL")
    if url is None:
        pytest.skip("Set CONTROL_PLANE_TEST_DATABASE_URL to a dedicated disposable Postgres database.")
    redis_url: Final = os.environ.get("CONTROL_PLANE_TEST_REDIS_URL", "redis://127.0.0.1:6379/0")
    engine: Final = engine_for(SecretStr(url))
    redis: Final = Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
    schema: Final = "cp_test_" + uuid4().hex
    stream_name: Final = "litellm:control_plane:test:" + uuid4().hex
    try:
        async with engine.connect() as probe:
            await probe.execute(text("SELECT 1"))
        await redis.ping()
    except (SQLAlchemyError, RedisError, OSError) as error:
        await redis.aclose()
        await engine.dispose()
        pytest.skip(f"Local control-plane services unavailable ({type(error).__name__}).")
    try:
        async with engine.begin() as connection:
            await connection.execute(CreateSchema(schema))
        isolated: Final = make_url(database_url(SecretStr(url))).update_query_dict(
            {"options": "-csearch_path=" + schema}
        )
        yield Services(SecretStr(isolated.render_as_string(hide_password=False)), SecretStr(redis_url), stream_name)
    finally:
        await redis.delete(stream_name)
        await redis.aclose()
        async with engine.begin() as cleanup:
            await cleanup.execute(DropSchema(schema, cascade=True, if_exists=True))
        await engine.dispose()


@pytest.mark.asyncio
async def test_migrations_are_idempotent_schema_version_matches_and_metadata_has_no_autogenerate_diff(
    services: Services,
) -> None:
    store: Final = PostgresStore.build(services.url)
    try:
        assert not await store.current()
        async with store.engine.connect() as empty:
            assert await empty.run_sync(lambda sync: inspect(sync).get_table_names()) == []
        await asyncio.to_thread(migrate, services.url)
        assert await store.current()
        await asyncio.to_thread(migrate, services.url)
        assert await store.current()
        async with store.engine.connect() as connection:
            differences: Final = await connection.run_sync(
                lambda sync: compare_metadata(
                    MigrationContext.configure(sync, opts={"version_table": VERSION_TABLE}), metadata
                )
            )
            for _ in range(8):
                assert await connection.scalar(text("SELECT CAST(:number AS integer)"), {"number": 1}) == 1
            assert await connection.scalar(text("SELECT COUNT(*) FROM pg_prepared_statements")) == 0
        assert differences == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pending_revisions_run_in_order_and_each_commits_independently(
    services: Services, tmp_path: Path
) -> None:
    config: Final = migration_config()
    source: Final = config.get_main_option("script_location")
    assert source is not None
    scripts: Final = tmp_path / "migrations"
    shutil.copytree(source, scripts)
    (scripts / "versions" / "0002_probe.py").write_text(
        'from alembic import op\nrevision = "0002"\ndown_revision = "0001"\n'
        'def upgrade():\n    op.execute("CREATE TABLE ordering_probe AS SELECT COUNT(*) AS count FROM control_plane_roles")\n'
    )
    (scripts / "versions" / "0003_failure.py").write_text(
        'from alembic import op\nrevision = "0003"\ndown_revision = "0002"\n'
        'def upgrade():\n    op.execute("CREATE TABLE rolled_back (value TEXT)")\n    raise RuntimeError("planned migration failure")\n'
    )
    config.set_main_option("script_location", str(scripts))
    config.attributes["database_url"] = services.url
    with pytest.raises(RuntimeError, match="planned migration failure"):
        await asyncio.to_thread(command.upgrade, config, "head")
    store: Final = PostgresStore.build(services.url)
    try:
        async with store.engine.connect() as connection:
            assert await connection.scalar(text("SELECT count FROM ordering_probe")) == 0
            assert await connection.scalar(text("SELECT version_num FROM control_plane_alembic_version")) == "0002"
            assert not await connection.run_sync(lambda sync: inspect(sync).has_table("rolled_back"))
        assert not await store.current()
    finally:
        await store.close()


async def eventually(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(30):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_migration_job_waits_for_the_postgres_advisory_lock(services: Services) -> None:
    engine: Final = engine_for(services.url)
    try:
        async with engine.begin() as holder:
            await holder.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": ADVISORY_LOCK})
            migrating: Final = asyncio.create_task(asyncio.to_thread(migrate, services.url))
            async with asyncio.timeout(30):
                async with engine.connect() as observer:
                    while True:
                        waiting: Final = await observer.scalar(
                            text(
                                "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                                "AND classid = :high AND objid = :low AND NOT granted)"
                            ),
                            {"high": ADVISORY_LOCK >> 32, "low": ADVISORY_LOCK & 0xFFFFFFFF},
                        )
                        if waiting:
                            break
            assert not migrating.done()
        await migrating
        assert await PostgresStore(engine).current()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_real_postgres_redis_workers_propagate_encrypted_dataplanes_roles_and_deletes(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "database_url": services.url,
            "redis_url": services.redis_url,
            "CONTROL_PLANE_SALT_KEY": "integration-encryption-key",
            "config_stream": services.stream_name,
            "audit_sinks": (),
        }
    )
    first: Final = build_container(settings)
    second: Final = build_container(settings)
    redis: Final = Redis.from_url(services.redis_url.get_secret_value(), decode_responses=True)
    store: Final = PostgresStore.build(services.url)
    app_first: Final = create_app(first, Licensed())
    app_second: Final = create_app(second, Licensed())
    try:
        async with app_first.router.lifespan_context(app_first), app_second.router.lifespan_context(app_second):
            async with (
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app_first), base_url="http://writer") as writer,
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app_second), base_url="http://reader") as reader,
            ):
                headers: Final = {"Authorization": f"Bearer {MASTER_KEY}"}
                changed: Final = await writer.put(
                    "/api/v1/dataplanes/live",
                    headers=headers,
                    json={
                        "url": "https://runtime.test",
                        "service_token": {"token": "must-not-persist-in-plaintext"},
                    },
                )
                assert changed.status_code == 200, changed.text
                await eventually(
                    lambda: first.registry.get("live") is not None and second.registry.get("live") is not None
                )
                seen: Final = await reader.get("/api/v1/dataplanes/live", headers=headers)
                assert seen.json() == changed.json()
                async with store.engine.connect() as connection:
                    persisted: Final = await connection.scalar(
                        select(dataplanes.c.spec).where(dataplanes.c.id == "live")
                    )
                published: Final = await redis.xrange(services.stream_name)
                assert "must-not-persist-in-plaintext" not in json.dumps(persisted) + str(published)
                role: Final = await writer.put(
                    "/api/v1/roles/live-reader", headers=headers, json={"scopes": ["usage:read"]}
                )
                assert role.status_code == 200, role.text
                await eventually(lambda: "live-reader" in second.policy.names)
                assert tuple(scope.value for scope in second.policy.scopes_for("live-reader")) == ("usage:read",)
                deleted: Final = await writer.delete("/api/v1/dataplanes/live", headers=headers)
                assert deleted.status_code == 204, deleted.text
                await eventually(lambda: first.registry.get("live") is None and second.registry.get("live") is None)
                assert (await reader.get("/api/v1/dataplanes/live", headers=headers)).status_code == 404
                async with store.transaction() as transaction:
                    rows: Final = await transaction.load()
                assert tuple(row.change.name for row in rows) == ("live-reader",)
    finally:
        await redis.aclose()
        await store.close()


@dataclass(frozen=True, slots=True)
class PausedDeleteStream:
    delegate: ConfigStream
    deleting: asyncio.Event
    release: asyncio.Event

    async def append(self, change: Change) -> str:
        if change.op == "delete":
            self.deleting.set()
            await self.release.wait()
        return await self.delegate.append(change)

    async def read(self, after: str) -> tuple[Entry, ...]:
        return await self.delegate.read(after)

    async def close(self) -> None:
        await self.delegate.close()


@pytest.mark.asyncio
async def test_concurrent_role_delete_and_inherit_cannot_commit_an_orphan(services: Services) -> None:
    await asyncio.to_thread(migrate, services.url)
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "database_url": services.url,
            "redis_url": services.redis_url,
            "CONTROL_PLANE_SALT_KEY": "integration-key",
            "audit_sinks": (),
        }
    )
    stream: Final = PausedDeleteStream(
        RedisStream.build(services.redis_url, services.stream_name, 100),
        asyncio.Event(),
        asyncio.Event(),
    )
    first: Final = build_container(settings, config_stream=stream)
    second: Final = build_container(settings, config_stream=stream)
    first.runtime_config.ready = second.runtime_config.ready = True
    store: Final = PostgresStore.build(services.url)
    headers: Final = {"Authorization": "Bearer " + MASTER_KEY}
    try:
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(first, Licensed())), base_url="http://delete"
            ) as deleting,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(second, Licensed())), base_url="http://inherit"
            ) as inheriting,
        ):
            seeded: Final = await deleting.put("/api/v1/roles/parent", headers=headers, json={"scopes": []})
            assert seeded.status_code == 200, seeded.text
            delete: Final = asyncio.create_task(deleting.delete("/api/v1/roles/parent", headers=headers))
            async with asyncio.timeout(30):
                await stream.deleting.wait()
            child: Final = asyncio.create_task(
                inheriting.put(
                    "/api/v1/roles/child",
                    headers=headers,
                    json={"inherits": ["parent"]},
                )
            )
            try:
                async with asyncio.timeout(30):
                    async with store.engine.connect() as observer:
                        while not child.done():
                            waiting: Final = await observer.scalar(
                                text(
                                    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
                                    "AND classid = :high AND objid = :low AND NOT granted)"
                                ),
                                {"high": ADVISORY_LOCK >> 32, "low": ADVISORY_LOCK & 0xFFFFFFFF},
                            )
                            if waiting:
                                break
            finally:
                stream.release.set()
            removed, refused = await asyncio.gather(delete, child)
            assert removed.status_code == 204, removed.text
            assert refused.status_code == 400, refused.text
            async with store.transaction() as transaction:
                assert await transaction.load() == ()
    finally:
        stream.release.set()
        await first.runtime_config.close()
        await second.runtime_config.close()
        await store.close()
