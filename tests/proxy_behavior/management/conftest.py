"""Session-scoped async ASGI client for behavior-pinning tests.

The proxy app is initialised once per pytest session against the real Postgres
pointed at by ``DATABASE_URL``. No mocks: auth runs, prisma runs, integrations
run. Tests assert at the HTTP boundary.
"""

import os
import tempfile
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
