"""Session-scoped async ASGI client for HTTP-boundary behavior tests."""

import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx
import pytest_asyncio
import yaml


MASTER_KEY = "sk-1234"
SCRATCH_PREFIX = "scratch-"


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
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import (
        app,
        cleanup_router_config_variables,
        initialize,
        proxy_startup_event,
    )

    cleanup_router_config_variables()
    config_path = _write_minimal_proxy_config()

    # proxy_startup_event re-reads master_key from LITELLM_MASTER_KEY and
    # unconditionally overwrites the global, even when initialize() already
    # set it from the config YAML. Force (not setdefault) both vars: an
    # ambient LITELLM_MASTER_KEY with a different value would make the proxy
    # authenticate on that key while the tests still send MASTER_KEY.
    os.environ["LITELLM_MASTER_KEY"] = MASTER_KEY
    os.environ["CONFIG_FILE_PATH"] = config_path

    await initialize(config=config_path)

    # /key/regenerate is gated behind premium_user; flipping it lets the matrix
    # pin authz behavior instead of the licensing gate.
    proxy_server.premium_user = True

    async with proxy_startup_event(app):
        proxy_server.premium_user = True  # lifespan re-runs _license_check
        # The lifespan fires check_view_exists() as a background task; on a
        # fresh DB the first auth call races it and resolves user_id=None.
        if proxy_server.prisma_client is not None:
            await proxy_server.prisma_client.check_view_exists()
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
    from litellm.proxy import proxy_server

    assert proxy_server.prisma_client is not None
    return proxy_server.prisma_client


@pytest_asyncio.fixture(scope="session")
async def world(prisma):
    from .actors import seed_world

    return await seed_world(prisma)


@dataclass(frozen=True)
class Scratch:
    prefix: str

    def tag(self, suffix: str = "") -> str:
        return f"{self.prefix}-{suffix}" if suffix else self.prefix


async def create_scratch_key(
    proxy_client,
    seeder_cleartext: str,
    scratch_prefix: str,
    *,
    user_id: str,
    team_id: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> str:
    """Seed a scratch-tagged key via /key/generate; returns its cleartext.

    Shared by the write-scenario matrices (key update/regenerate/delete).
    """
    body: Dict[str, Any] = {"key_alias": scratch_prefix, "user_id": user_id}
    if team_id is not None:
        body["team_id"] = team_id
    if organization_id is not None:
        body["organization_id"] = organization_id
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder_cleartext}"},
        json=body,
    )
    assert resp.status_code == 200, f"setup failed: {resp.text}"
    return resp.json()["key"]


@pytest_asyncio.fixture
async def scratch(prisma):
    handle = Scratch(prefix=f"{SCRATCH_PREFIX}{uuid.uuid4().hex[:12]}")
    try:
        yield handle
    finally:
        # Children before parents to avoid FK violations.
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
