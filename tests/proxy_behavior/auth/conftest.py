"""Session-scoped PrismaClient for auth behavior tests that run raw SQL against a real Postgres."""

import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from litellm.proxy.utils import PrismaClient


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def prisma():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")  # test-quality-ok: this suite exists to run SQL on a real Postgres
    client = PrismaClient(database_url=database_url, proxy_logging_obj=MagicMock())
    await client.connect()
    yield client
    await client.disconnect()
