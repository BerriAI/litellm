import os
import pytest
from fastapi.testclient import TestClient
from litellm.proxy.proxy_server import app, ProxyLogging
from litellm.caching import DualCache


@pytest.fixture(autouse=True)
def override_env_settings(monkeypatch):
    # Set environment variables only for tests using-monkeypatch (function scope by default).
    # Use DATABASE_URL from environment (set by CircleCI to local postgres)
    if "DATABASE_URL" not in os.environ:
        pytest.fail("DATABASE_URL not set - this test requires a local postgres database to be running")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-1234")
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")


@pytest.fixture(scope="module")
def test_client():
    """
    This fixture starts up the test client which triggers FastAPI's startup events.
    Prisma will connect to the DB using the provided DATABASE_URL.
    """
    with TestClient(app) as client:
        yield client


@pytest.mark.asyncio
async def test_master_key_not_inserted(test_client):
    """
    This test ensures that when the app starts (or when you hit the /health endpoint
    to trigger startup logic), no unexpected write occurs in the DB.
    """
    # Hit an endpoint (like /health) that triggers any startup tasks.
    response = test_client.get("/health/liveliness")
    assert response.status_code == 200

    from litellm.proxy.utils import PrismaClient

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"],
        proxy_logging_obj=ProxyLogging(
            user_api_key_cache=DualCache(), premium_user=True
        ),
    )

    # Connect directly to the test database to inspect the data.
    await prisma_client.connect()
    result = await prisma_client.db.litellm_verificationtoken.find_many()
    print(result)

    # The expectation is that no token (or unintended record) is added on startup.
    assert len(result) == 0, (
        "SECURITY ALERT SECURITY ALERT SECURITY ALERT: Expected no record in the litellm_verificationtoken table. On startup - the master key should NOT be Inserted into the DB."
        "We have found keys in the DB. This is unexpected and should not happen."
    )
