import os
import pytest
from fastapi.testclient import TestClient
from litellm.proxy.proxy_server import app, ProxyLogging
from litellm.caching import DualCache

TEST_DB_ENV_VAR_NAME = "MASTER_KEY_CHECK_DB_URL"


@pytest.fixture(scope="module", autouse=True)
def override_env_settings():
    # Point to your dedicated test PostgreSQL instance.
    # (Make sure this DB is isolated from production and is safe to run tests against.)
    os.environ["DATABASE_URL"] = os.environ[TEST_DB_ENV_VAR_NAME]
    os.environ["LITELLM_MASTER_KEY"] = "sk-1234"
    os.environ["LITELLM_LOG"] = "DEBUG"
    yield
    # Clean up environment variables (if needed) after tests.
    del os.environ["DATABASE_URL"]
    del os.environ["LITELLM_MASTER_KEY"]


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
        database_url=os.environ[TEST_DB_ENV_VAR_NAME],
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
        "🚨 🚨🚨🚨🚨🚨🚨 - Security Alert: Expected no record in the litellm_verificationtoken table. On startup - the master key should NOT be Inserted into the DB."
        "We have found keys in the DB. This is unexpected and should not happen."
    )
