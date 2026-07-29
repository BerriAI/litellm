import os
import pytest
from fastapi.testclient import TestClient
from litellm.proxy.proxy_server import app, ProxyLogging, hash_token
from litellm.caching import DualCache

MASTER_KEY = "sk-1234"


@pytest.fixture(autouse=True)
def override_env_settings(monkeypatch):
    if "DATABASE_URL" not in os.environ:
        pytest.fail(
            "DATABASE_URL not set - this test requires a postgres database to be running"
        )
    monkeypatch.setenv("LITELLM_MASTER_KEY", MASTER_KEY)
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")


@pytest.fixture(scope="module")
def test_client():
    """Starting the test client triggers FastAPI startup, where Prisma connects to the DB."""
    with TestClient(app) as client:
        yield client


@pytest.mark.asyncio
async def test_master_key_not_inserted(test_client):
    """The master key must never be persisted to the verification-token table on startup."""
    response = test_client.get("/health/liveliness")
    assert response.status_code == 200

    from litellm.proxy.utils import PrismaClient

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"],
        proxy_logging_obj=ProxyLogging(
            user_api_key_cache=DualCache(), premium_user=True
        ),
    )

    await prisma_client.connect()
    stored_tokens = {
        row.token
        for row in await prisma_client.db.litellm_verificationtoken.find_many()
    }

    for leaked in (hash_token(MASTER_KEY), MASTER_KEY):
        assert leaked not in stored_tokens, (
            "SECURITY ALERT: the master key was found in the litellm_verificationtoken "
            "table. The master key must never be inserted into the DB."
        )

    # Canary against any other unexpected startup write (default key, rotation
    # artifact, ...). The job gives each run a fresh DB, so a clean startup must
    # leave the table empty; if startup ever legitimately seeds a token, narrow
    # this while keeping the master-key assertion above.
    assert (
        not stored_tokens
    ), f"startup unexpectedly wrote token(s) to litellm_verificationtoken: {stored_tokens}"
