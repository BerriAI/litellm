import asyncio
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

import httpx
import psutil
import pytest

from litellm.proxy.db.create_views import SupportsRawQueries
from tests.integration._support.client import eventually
from tests.integration._support.database import read_rows, scratch_database

pytestmark: Final = pytest.mark.timeout(180)
ROOT: Final = Path(__file__).resolve().parents[3]
PUBLICLY_KNOWN_KEY: Final = "sk-" + "1234"
Mode: TypeAlias = Literal["late", "unsafe", "gateway"]
CONFIG: Final = (
    "model_list: []\n"
    "general_settings:\n"
    "  disable_prisma_schema_update: true\n"
    "  disable_spend_logs: true\n"
    "  disable_model_info_refresh: true\n"
)


def _assert_database_idle(database_url: str | None = None) -> None:
    assert (
        eventually(
            lambda: read_rows(
                "SELECT pid FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()",
                (),
                database_url=database_url,
            ),
            lambda rows: not rows,
            seconds=5,
        )
        == []
    )
    assert psutil.Process().children(recursive=True) == []
    assert tuple(thread.name for thread in threading.enumerate() if thread.name.startswith("prisma-engine-")) == ()


async def _assert_lifespan_closed(database_url: str | None = None) -> None:
    from litellm.proxy import proxy_server

    await asyncio.sleep(0)
    assert proxy_server.prisma_client is None
    assert (
        tuple(
            task.get_coro().__qualname__
            for task in asyncio.all_tasks()
            if task.get_coro().__qualname__.startswith(
                ("PrismaClient.", "PrismaWrapper.", "_proxy_lifespan.<locals>._run_")
            )
        )
        == ()
    )
    await asyncio.to_thread(_assert_database_idle, database_url)


async def _assert_one_prisma_session(db: SupportsRawQueries) -> None:
    sessions: Final = await asyncio.gather(
        *(db.query_raw("SELECT pg_backend_pid() AS pid FROM pg_sleep(0.2)") for _ in range(4))
    )
    assert len({batch[0]["pid"] for batch in sessions}) == 1, sessions


async def _assert_gateway_pool_is_capped(database_url: str) -> None:
    from gateway.main import app
    from litellm.proxy import proxy_server

    async with app.router.lifespan_context(app):
        prisma_client: Final = proxy_server.prisma_client
        assert prisma_client is not None
        for db in (prisma_client.writer_db, prisma_client.read_db):
            await _assert_one_prisma_session(cast(SupportsRawQueries, db))  # cast-ok: PrismaWrapper forwards query_raw
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            response: Final = await client.post("/v1/vector_stores/example/search", json={"query": "example"})
            assert response.status_code == 401, response.text
    await _assert_lifespan_closed(database_url)


async def _exercise_lifespans(mode: Mode) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.master_key_boot_check import UnsafeMasterKeyError

    if mode == "gateway":
        await _assert_gateway_pool_is_capped(os.environ["DATABASE_URL"])
        return

    expected: Final = UnsafeMasterKeyError if mode == "unsafe" else ValueError
    match: Final = "master key" if mode == "unsafe" else "use_redis_transaction_buffer"
    with pytest.raises(expected, match=match):
        async with proxy_server.proxy_startup_event(proxy_server.app):
            pytest.fail("Invalid configuration started the proxy")
    await _assert_lifespan_closed()

    Path(os.environ["CONFIG_FILE_PATH"]).write_text(CONFIG)
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("LITELLM_MASTER_KEY", f"sk-{uuid.uuid4().hex}")
        for cancel in (False, True, False):
            try:
                async with proxy_server.proxy_startup_event(proxy_server.app) as state:
                    database: Final = state["database"]
                    assert database is not None
                    assert database.native is not None
                    assert await database.prisma.writer_db.query_raw("SELECT 42 AS answer") == [{"answer": 42}]
                    assert await database.prisma.read_db.query_raw("SELECT 43 AS answer") == [{"answer": 43}]
                    assert psutil.Process().children(recursive=True)
                    if cancel:
                        raise asyncio.CancelledError
            except asyncio.CancelledError:
                if not cancel:
                    raise
            await _assert_lifespan_closed()


@pytest.mark.parametrize("mode", ("late", "unsafe", "gateway"))
@pytest.mark.parametrize("replica", (False, True), ids=("writer", "writer-and-reader"))
def test_proxy_lifespan_releases_database_resources(tmp_path: Path, mode: Mode, replica: bool) -> None:
    config: Final = tmp_path / "config.yaml"
    extras: Final = {
        "late": "  use_redis_transaction_buffer: true\n",
        "gateway": "  database_connection_pool_limit: 2\n",
    }
    config.write_text(CONFIG + extras.get(mode, ""))
    inherited: Final = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("REDIS_")
        and key
        not in (
            "WORKER_CONFIG",
            "DATABASE_URL_READ_REPLICA",
            "LITELLM_SALT_KEY",
            "LITELLM_MIGRATE_FROM_MASTER_KEY",
            "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY",
            "INTEGRATION_UPSTREAM_URL",
        )
    }
    with scratch_database("lifespan") as database_url:
        environment: Final = {
            **inherited,
            "DATABASE_URL": database_url,
            "DIRECT_URL": database_url,
            "CONFIG_FILE_PATH": str(config),
            "LITELLM_MASTER_KEY": PUBLICLY_KNOWN_KEY if mode == "unsafe" else f"sk-{uuid.uuid4().hex}",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_TELEMETRY_ENABLED": "false",
            "PRISMA_HEALTH_WATCHDOG_ENABLED": "true",
            **({"LITELLM_DISABLE_LAZY_ROUTES": "true"} if mode == "gateway" else {}),
            "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "litellm/enterprise"), str(ROOT / "litellm/runtime"))),
            **({"DATABASE_URL_READ_REPLICA": database_url} if replica else {}),
        }
        schema: Final = subprocess.run(
            (
                sys.executable,
                "-P",
                "-m",
                "prisma",
                "db",
                "push",
                "--skip-generate",
                "--schema",
                str(ROOT / "litellm/proxy/schema.prisma"),
            ),
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert schema.returncode == 0, schema.stderr
        probe: Final = subprocess.run(
            (
                sys.executable,
                "-P",
                "-c",
                "import asyncio; from tests.integration.database.test_proxy_database_lifespan import _exercise_lifespans; "
                f"asyncio.run(_exercise_lifespans({mode!r}))",
            ),
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert probe.returncode == 0, probe.stdout + probe.stderr
