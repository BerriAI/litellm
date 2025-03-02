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


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_team_id(client, monkeypatch):
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
            "team_id": "team2",
            "spend": 0.10,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-4",
        },
    ]

    # Create a mock prisma client
    class MockDB:
        async def find_many(self, *args, **kwargs):
            # Filter based on team_id in the where conditions
            if (
                "where" in kwargs
                and "team_id" in kwargs["where"]
                and kwargs["where"]["team_id"] == "team1"
            ):
                return [mock_spend_logs[0]]
            return mock_spend_logs

        async def count(self, *args, **kwargs):
            # Return count based on team_id filter
            if (
                "where" in kwargs
                and "team_id" in kwargs["where"]
                and kwargs["where"]["team_id"] == "team1"
            ):
                return 1
            return len(mock_spend_logs)

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db

    # Apply the monkeypatch
    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Set up test dates
    start_date = (
        datetime.datetime.now(timezone.utc) - datetime.timedelta(days=7)
    ).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Make the request with team_id filter
    response = client.get(
        "/spend/logs/ui",
        params={
            "team_id": "team1",
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    # Assert response
    assert response.status_code == 200
    data = response.json()

    # Verify the filtered data
    assert data["total"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["team_id"] == "team1"


@pytest.mark.asyncio
async def test_ui_view_spend_logs_pagination(client, monkeypatch):
    # Create a larger set of mock data for pagination testing
    mock_spend_logs = [
        {
            "id": f"log{i}",
            "request_id": f"req{i}",
            "api_key": "sk-test-key",
            "user": f"test_user_{i % 3}",
            "team_id": f"team{i % 2 + 1}",
            "spend": 0.05 * i,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo" if i % 2 == 0 else "gpt-4",
        }
        for i in range(1, 26)  # 25 records
    ]

    # Create a mock prisma client with pagination support
    class MockDB:
        async def find_many(self, *args, **kwargs):
            # Handle pagination
            skip = kwargs.get("skip", 0)
            take = kwargs.get("take", 10)
            return mock_spend_logs[skip : skip + take]

        async def count(self, *args, **kwargs):
            return len(mock_spend_logs)

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db

    # Apply the monkeypatch
    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Set up test dates
    start_date = (
        datetime.datetime.now(timezone.utc) - datetime.timedelta(days=7)
    ).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Test first page
    response = client.get(
        "/spend/logs/ui",
        params={
            "page": 1,
            "page_size": 10,
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 25
    assert len(data["data"]) == 10
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert data["total_pages"] == 3

    # Test second page
    response = client.get(
        "/spend/logs/ui",
        params={
            "page": 2,
            "page_size": 10,
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 25
    assert len(data["data"]) == 10
    assert data["page"] == 2


@pytest.mark.asyncio
async def test_ui_view_spend_logs_date_range_filter(client, monkeypatch):
    # Create mock data with different dates
    today = datetime.datetime.now(timezone.utc)

    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": (today - datetime.timedelta(days=10)).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_2",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": (today - datetime.timedelta(days=2)).isoformat(),
            "model": "gpt-4",
        },
    ]

    # Create a mock prisma client with date filtering
    class MockDB:
        async def find_many(self, *args, **kwargs):
            # Check for date range filtering
            if "where" in kwargs and "startTime" in kwargs["where"]:
                date_filters = kwargs["where"]["startTime"]
                filtered_logs = []

                for log in mock_spend_logs:
                    log_date = datetime.datetime.fromisoformat(
                        log["startTime"].replace("Z", "+00:00")
                    )

                    # Apply gte filter if it exists
                    if "gte" in date_filters:
                        # Handle ISO format date strings
                        if "T" in date_filters["gte"]:
                            filter_date = datetime.datetime.fromisoformat(
                                date_filters["gte"].replace("Z", "+00:00")
                            )
                        else:
                            filter_date = datetime.datetime.strptime(
                                date_filters["gte"], "%Y-%m-%d %H:%M:%S"
                            )

                        if log_date < filter_date:
                            continue

                    # Apply lte filter if it exists
                    if "lte" in date_filters:
                        # Handle ISO format date strings
                        if "T" in date_filters["lte"]:
                            filter_date = datetime.datetime.fromisoformat(
                                date_filters["lte"].replace("Z", "+00:00")
                            )
                        else:
                            filter_date = datetime.datetime.strptime(
                                date_filters["lte"], "%Y-%m-%d %H:%M:%S"
                            )

                        if log_date > filter_date:
                            continue

                    filtered_logs.append(log)

                return filtered_logs

            return mock_spend_logs

        async def count(self, *args, **kwargs):
            # For simplicity, we'll just call find_many and count the results
            logs = await self.find_many(*args, **kwargs)
            return len(logs)

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db

    # Apply the monkeypatch
    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Test with a date range that should only include the second log
    start_date = (today - datetime.timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    end_date = today.strftime("%Y-%m-%d %H:%M:%S")

    response = client.get(
        "/spend/logs/ui",
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "log2"


@pytest.mark.asyncio
async def test_ui_view_spend_logs_unauthorized(client):
    # Test without authorization header
    response = client.get("/spend/logs/ui")
    assert response.status_code == 401 or response.status_code == 403

    # Test with invalid authorization
    response = client.get(
        "/spend/logs/ui",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401 or response.status_code == 403
