"""Session-scoped async ASGI client for behavior-pinning tests.

The proxy app is initialised once per pytest session against the real Postgres
pointed at by ``DATABASE_URL``. No mocks: auth runs, prisma runs, integrations
run. Tests assert at the HTTP boundary.
"""

import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
import pytest_asyncio
import yaml


MASTER_KEY = "sk-1234"


def _write_minimal_proxy_config() -> str:
    config = {
        "general_settings": {"master_key": MASTER_KEY},
        "litellm_settings": {},
    }
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        config["general_settings"]["database_url"] = database_url

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, f)
    f.close()
    return f.name


@pytest_asyncio.fixture(scope="session")
async def proxy_app():
    """Boot the proxy app once per session with the real FastAPI lifespan.

    httpx 0.28's ASGITransport does not run the lifespan handler, so we enter
    ``proxy_startup_event`` (the @asynccontextmanager registered as the app's
    lifespan) directly. That handler is where ``prisma_client`` is connected
    and the rest of the startup wiring runs.
    """
    from litellm.proxy.proxy_server import (
        app,
        cleanup_router_config_variables,
        initialize,
        proxy_startup_event,
    )

    cleanup_router_config_variables()
    config_path = _write_minimal_proxy_config()
    await initialize(config=config_path)
    async with proxy_startup_event(app):
        yield app


@pytest_asyncio.fixture(scope="session")
async def proxy_client(proxy_app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=proxy_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def prisma(proxy_app):
    """The connected PrismaClient the lifespan opened."""
    from litellm.proxy import proxy_server

    assert (
        proxy_server.prisma_client is not None
    ), "FastAPI lifespan did not connect prisma — harness is wrong."
    return proxy_server.prisma_client


@pytest_asyncio.fixture(scope="session")
async def world(prisma):
    """The immutable read-world seed.

    Re-seeds at session start so each pytest invocation gets a clean world.
    Tests must not mutate these rows; write tests use the ``scratch`` fixture
    below for scoped entities that get torn down per-test.
    """
    from .actors import seed_world

    return await seed_world(prisma)


SCRATCH_PREFIX = "scratch-"


@dataclass(frozen=True)
class Scratch:
    """Per-test namespace for write scenarios.

    Tests must tag any entity they create with ``scratch.prefix`` in a column
    the teardown filter inspects (``key_alias``, ``key_name``, ``team_alias``,
    ``team_id``, ``user_id``, or ``budget_id``). Anything not tagged will be
    left behind and pollute the next session.
    """

    prefix: str

    def tag(self, suffix: str = "") -> str:
        return f"{self.prefix}-{suffix}" if suffix else self.prefix


@pytest_asyncio.fixture
async def scratch(prisma):
    """Function-scoped scratch namespace + targeted delete_many teardown.

    The teardown deletes any rows on the volatile tables whose namespace column
    starts with ``scratch.prefix``. Per CLAUDE.md, this is Prisma-only — no raw
    SQL — and uses ``delete_many`` to batch the writes.
    """
    handle = Scratch(prefix=f"{SCRATCH_PREFIX}{uuid.uuid4().hex[:12]}")
    try:
        yield handle
    finally:
        # Order matters: children before parents to avoid FK conflicts.
        await prisma.db.litellm_verificationtoken.delete_many(
            where={
                "OR": [
                    {"key_alias": {"startswith": handle.prefix}},
                    {"key_name": {"startswith": handle.prefix}},
                ]
            }
        )
        await prisma.db.litellm_teammembership.delete_many(
            where={"team_id": {"startswith": handle.prefix}}
        )
        await prisma.db.litellm_organizationmembership.delete_many(
            where={"user_id": {"startswith": handle.prefix}}
        )
        await prisma.db.litellm_teamtable.delete_many(
            where={"team_id": {"startswith": handle.prefix}}
        )
        await prisma.db.litellm_usertable.delete_many(
            where={"user_id": {"startswith": handle.prefix}}
        )
        await prisma.db.litellm_budgettable.delete_many(
            where={"budget_id": {"startswith": handle.prefix}}
        )
