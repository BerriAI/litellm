import datetime
import json
import os
import sys
from datetime import timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.proxy_server import app, prisma_client


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_user_id(client, monkeypatch):
    # Mock data for the test
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    # Create a mock prisma client
    class MockDB:
        async def find_many(self, *args, **kwargs):
            # Filter based on user_id in the where conditions
            print("kwargs to find_many", json.dumps(kwargs, indent=4))
            if (
                "where" in kwargs
                and "user" in kwargs["where"]
                and kwargs["where"]["user"] == "test_user_1"
            ):
                return [mock_spend_logs[0]]
            return mock_spend_logs

        async def count(self, *args, **kwargs):
            # Return count based on user_id filter
            if (
                "where" in kwargs
                and "user" in kwargs["where"]
                and kwargs["where"]["user"] == "test_user_1"
            ):
                return 1
            return len(mock_spend_logs)

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db

    # Apply the monkeypatch to replace the prisma_client
    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Set up test dates
    start_date = (
        datetime.datetime.now(timezone.utc) - datetime.timedelta(days=7)
    ).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Make the request with user_id filter
    response = client.get(
        "/spend/logs/ui",
        params={
            "user_id": "test_user_1",
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    # Assert response
    assert response.status_code == 200
    data = response.json()

    # Verify the response structure
    assert "data" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data

    # Verify the filtered data
    assert data["total"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["user"] == "test_user_1"
