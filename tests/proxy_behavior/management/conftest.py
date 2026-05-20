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

    # proxy_startup_event re-reads master_key from LITELLM_MASTER_KEY (line 776
    # in proxy_server.py). If unset, the global master_key is overwritten to
    # None *after* initialize()'s config-derived value, and the entire auth
    # stack falls into a degraded path that produces user_id=None and a
    # non-PROXY_ADMIN role for every key, including the master key itself.
    # Locally this is masked when LITELLM_MASTER_KEY happens to be set in the
    # shell; CI is clean, which is how this surfaced.
    os.environ.setdefault("LITELLM_MASTER_KEY", MASTER_KEY)
    os.environ.setdefault("CONFIG_FILE_PATH", config_path)

    await initialize(config=config_path)

    # /key/regenerate (and a few other Tier-1 endpoints) are gated behind
    # ``premium_user`` — without a LITELLM_LICENSE the proxy returns 500
    # "Enterprise feature" for those calls. The behavior matrix isn't about
    # licensing; it's about authz. Force the proxy into premium mode so the
    # matrix pins the real authz behavior, not the licensing gate.
    from litellm.proxy import proxy_server as _proxy_server

    _proxy_server.premium_user = True

    async with proxy_startup_event(app):
        # The lifespan re-runs ``premium_user = _license_check.is_premium()``
        # which flips it back. Force it again after the lifespan settles.
        _proxy_server.premium_user = True

        # The lifespan kicks off ``prisma_client.check_view_exists()`` as a
        # fire-and-forget background task. That task creates the
        # ``LiteLLM_VerificationTokenView`` SQL view used by ``user_api_key_auth``
        # to resolve a token to its user / role / team. On a fresh Postgres
        # (CI), the first test races the task — the view doesn't exist yet,
        # ``user_api_key_dict.user_id`` resolves to ``None``, and every authz
        # check that depends on it fails confusingly. Locally the view already
        # exists from prior runs, masking the race. Await it explicitly here
        # so the suite is deterministic regardless of DB state.
        from litellm.proxy import proxy_server as _proxy_server

        if _proxy_server.prisma_client is not None:
            await _proxy_server.prisma_client.check_view_exists()
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
