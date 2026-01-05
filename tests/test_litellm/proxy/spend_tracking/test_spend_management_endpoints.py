import asyncio
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

from unittest.mock import MagicMock, patch

import litellm
import litellm.proxy.proxy_server as ps
from litellm.proxy._types import (
    LitellmUserRoles,
    Member,
    SpendLogsPayload,
    UserAPIKeyAuth,
)
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.proxy_server import app, prisma_client
from litellm.proxy.spend_tracking import spend_management_endpoints
from litellm.router import Router
from litellm.types.utils import BudgetConfig


@pytest.mark.asyncio
async def test_is_admin_view_safe_true(monkeypatch):
    # Force underlying check to return True
    monkeypatch.setattr(
        spend_management_endpoints, "_user_has_admin_view", lambda user_api_key_dict: True
    )
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user")
    assert spend_management_endpoints._is_admin_view_safe(auth) is True


@pytest.mark.asyncio
async def test_is_admin_view_safe_false(monkeypatch):
    # Force underlying check to return False
    monkeypatch.setattr(
        spend_management_endpoints, "_user_has_admin_view", lambda user_api_key_dict: False
    )
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    assert spend_management_endpoints._is_admin_view_safe(auth) is False


@pytest.mark.asyncio
async def test_is_admin_view_safe_exception(monkeypatch):
    # Ensure exceptions are swallowed and return False
    def raise_err(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(spend_management_endpoints, "_user_has_admin_view", raise_err)
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    assert spend_management_endpoints._is_admin_view_safe(auth) is False


@pytest.mark.asyncio
async def test_can_team_member_view_log_none_team_id():
    # team_id=None should immediately return False
    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return None

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, None
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_can_team_member_view_log_team_not_found(monkeypatch):
    # Non-existent team should return False
    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return None

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    # Even if admin check would return True, no team means False
    monkeypatch.setattr(
        spend_management_endpoints, "_is_user_team_admin", lambda user_api_key_dict, team_obj: True
    )
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, "team_x"
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_can_team_member_view_log_not_admin(monkeypatch):
    # Existing team but caller is not a team admin -> False
    class MockTeam:
        pass

    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return MockTeam()

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    monkeypatch.setattr(
        spend_management_endpoints, "_is_user_team_admin", lambda user_api_key_dict, team_obj: False
    )
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, "team_x"
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_can_team_member_view_log_admin(monkeypatch):
    # Existing team and caller is team admin -> True
    class MockTeam:
        pass

    class MockPrisma:
        class DB:
            class TeamTable:
                async def find_unique(self, where: dict):
                    return MockTeam()

            def __init__(self):
                self.litellm_teamtable = self.TeamTable()

        def __init__(self):
            self.db = self.DB()

    prisma = MockPrisma()
    monkeypatch.setattr(
        spend_management_endpoints, "_is_user_team_admin", lambda user_api_key_dict, team_obj: True
    )
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user_1")
    allowed = await spend_management_endpoints._can_team_member_view_log(
        prisma, auth, "team_x"
    )
    assert allowed is True


def test_can_user_view_spend_log_true_for_internal_user():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="u1")
    assert spend_management_endpoints._can_user_view_spend_log(auth) is True


def test_can_user_view_spend_log_true_for_internal_view_only():
    auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, user_id="u1"
    )
    assert spend_management_endpoints._can_user_view_spend_log(auth) is True


def test_can_user_view_spend_log_false_without_user_id():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id=None)
    assert spend_management_endpoints._can_user_view_spend_log(auth) is False


def test_can_user_view_spend_log_false_for_other_roles():
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")
    assert spend_management_endpoints._can_user_view_spend_log(auth) is False

ignored_keys = [
    "request_id",
    "session_id",
    "startTime",
    "endTime",
    "completionStartTime",
    "endTime",
    "organization_id",
    "metadata.model_map_information",
    "metadata.usage_object",
    "metadata.cold_storage_object_key",
    "metadata.additional_usage_values.prompt_tokens_details.cache_creation_tokens",
    "metadata.litellm_overhead_time_ms",
]

MODEL_LIST = [
    {
        "model_name": "azure-gpt-4o",
        "litellm_params": {
            "model": "azure/gpt-4o-mini",
            "mock_response": "Hello, world!",
            "tags": ["default"],
            "base_model": "gpt-4o-mini",
        },
    },
]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def add_anthropic_api_key_to_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-1234567890")


@pytest.fixture
def disable_budget_sync(monkeypatch):
    """Disable periodic sync during tests"""

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


@pytest.fixture(autouse=True)
def reset_router_callbacks():
    """Ensure router budget callbacks from previous tests do not leak state."""
    litellm.logging_callback_manager._reset_all_callbacks()
    yield
    litellm.logging_callback_manager._reset_all_callbacks()


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
    
    # Mock _is_admin_view_safe to return True to bypass permission checks
    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints._is_admin_view_safe",
        lambda user_api_key_dict: True
    )

    # Override auth dependency to return PROXY_ADMIN
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
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
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_internal_user_scoped_without_user_id(client, monkeypatch):
    """
    Internal users should only be able to view their own spend even if user_id is not provided.
    """
    # Mock spend logs for 2 users
    mock_spend_logs = [
        {"id": "log1", "request_id": "req1", "api_key": "sk-test-key", "user": "internal_user_1", "team_id": "team1", "spend": 0.05, "startTime": datetime.datetime.now(timezone.utc).isoformat(), "model": "gpt-3.5-turbo"},
        {"id": "log2", "request_id": "req2", "api_key": "sk-test-key", "user": "internal_user_2", "team_id": "team1", "spend": 0.10, "startTime": datetime.datetime.now(timezone.utc).isoformat(), "model": "gpt-4"},
    ]

    # Prisma client mock that filters by "user" where condition
    class MockDB:
        async def find_many(self, *args, **kwargs):
            where = kwargs.get("where", {})
            if "user" in where and where["user"] == "internal_user_1":
                return [mock_spend_logs[0]]
            return mock_spend_logs

        async def count(self, *args, **kwargs):
            where = kwargs.get("where", {})
            if "user" in where and where["user"] == "internal_user_1":
                return 1
            return len(mock_spend_logs)

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db

    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Override auth dependency to return INTERNAL_USER with specific user_id
    # Override using the function reference attached to the running app module
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="internal_user_1"
    )

    try:
        start_date = (datetime.datetime.now(timezone.utc) - datetime.timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        end_date = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # No user_id provided; should auto-scope to authenticated internal user's own id
        response = client.get(
            "/spend/logs/ui",
            params={"start_date": start_date, "end_date": end_date},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["user"] == "internal_user_1"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_ui_view_spend_logs_team_admin_can_view_team_spend(client, monkeypatch):
    """
    Team admins should be able to view team-wide spend when team_id is provided.
    """
    # Mock spend logs for two teams
    mock_spend_logs = [
        {"id": "log1", "request_id": "req1", "api_key": "sk-test-key", "user": "member1", "team_id": "team_admin_team", "spend": 0.05, "startTime": datetime.datetime.now(timezone.utc).isoformat(), "model": "gpt-3.5-turbo"},
        {"id": "log2", "request_id": "req2", "api_key": "sk-test-key", "user": "member2", "team_id": "team_other", "spend": 0.10, "startTime": datetime.datetime.now(timezone.utc).isoformat(), "model": "gpt-4"},
    ]

    class MockDB:
        async def find_many(self, *args, **kwargs):
            where = kwargs.get("where", {})
            if "team_id" in where and where["team_id"] == "team_admin_team":
                return [mock_spend_logs[0]]
            return mock_spend_logs

        async def count(self, *args, **kwargs):
            where = kwargs.get("where", {})
            if "team_id" in where and where["team_id"] == "team_admin_team":
                return 1
            return len(mock_spend_logs)

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db
            # Team lookup for RBAC check
            class TeamTable:
                def __init__(self):
                    # user "admin_user" is team admin
                    self.members_with_roles = [Member(user_id="admin_user", role="admin")]

            async def find_unique(where: dict):
                if where == {"team_id": "team_admin_team"}:
                    return TeamTable()
                return None

            self.db.litellm_teamtable = self
            self.litellm_teamtable = self
            self.find_unique = find_unique

    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Override auth dependency to return INTERNAL_USER (who is a team admin via team.members_with_roles)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="admin_user"
    )

    try:
        start_date = (datetime.datetime.now(timezone.utc) - datetime.timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        end_date = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        response = client.get(
            "/spend/logs/ui",
            params={"team_id": "team_admin_team", "start_date": start_date, "end_date": end_date},
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["team_id"] == "team_admin_team"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

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
async def test_ui_view_session_spend_logs_pagination(client, monkeypatch):
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "session_id": "session-123",
            "startTime": "2024-01-01T00:00:00Z",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "session_id": "session-123",
            "startTime": "2024-01-02T00:00:00Z",
        },
    ]

    class MockDB:
        async def count(self, *args, **kwargs):
            assert kwargs.get("where") == {"session_id": "session-123"}
            return len(mock_spend_logs)

        async def find_many(self, *args, **kwargs):
            assert kwargs.get("where") == {"session_id": "session-123"}
            assert kwargs.get("order") == {"startTime": "asc"}
            assert kwargs.get("skip") == 1  # page=2, page_size=1
            assert kwargs.get("take") == 1
            return [mock_spend_logs[1]]

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()
            self.db.litellm_spendlogs = self.db

    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    response = client.get(
        "/spend/logs/session/ui",
        params={"session_id": "session-123", "page": 2, "page_size": 1},
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["page"] == 2
    assert data["page_size"] == 1
    assert data["total_pages"] == 2
    assert len(data["data"]) == 1
    assert data["data"][0]["request_id"] == "req2"


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


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_status(client, monkeypatch):
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
            "status": "success",
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
            "status": "failure",
        },
    ]

    # Create a mock prisma client
    class MockDB:
        async def find_many(self, *args, **kwargs):
            # Filter based on status in the where conditions
            if "where" in kwargs:
                where_conditions = kwargs["where"]
                if "OR" in where_conditions:
                    # Handle success case (which includes None status)
                    return [mock_spend_logs[0]]
                elif (
                    "status" in where_conditions
                    and where_conditions["status"]["equals"] == "failure"
                ):
                    return [mock_spend_logs[1]]
            return mock_spend_logs

        async def count(self, *args, **kwargs):
            # Return count based on status filter
            if "where" in kwargs:
                where_conditions = kwargs["where"]
                if "OR" in where_conditions:
                    return 1
                elif (
                    "status" in where_conditions
                    and where_conditions["status"]["equals"] == "failure"
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

    # Test success status
    response = client.get(
        "/spend/logs/ui",
        params={
            "status_filter": "success",
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["status"] == "success"

    # Test failure status
    response = client.get(
        "/spend/logs/ui",
        params={
            "status_filter": "failure",
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["status"] == "failure"


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_model(client, monkeypatch):
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
            "status": "success",
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
            "status": "success",
        },
    ]

    # Create a mock prisma client
    class MockDB:
        async def find_many(self, *args, **kwargs):
            # Filter based on model in the where conditions
            if (
                "where" in kwargs
                and "model" in kwargs["where"]
                and kwargs["where"]["model"] == "gpt-3.5-turbo"
            ):
                return [mock_spend_logs[0]]
            return mock_spend_logs

        async def count(self, *args, **kwargs):
            # Return count based on model filter
            if (
                "where" in kwargs
                and "model" in kwargs["where"]
                and kwargs["where"]["model"] == "gpt-3.5-turbo"
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

    # Make the request with model filter
    response = client.get(
        "/spend/logs/ui",
        params={
            "model": "gpt-3.5-turbo",
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
    assert data["data"][0]["model"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_ui_view_spend_logs_with_key_hash(client, monkeypatch):
    # Mock data for the test
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key-1",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": datetime.datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key-2",
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
            # Filter based on key_hash in the where conditions
            if (
                "where" in kwargs
                and "api_key" in kwargs["where"]
                and kwargs["where"]["api_key"] == "sk-test-key-1"
            ):
                return [mock_spend_logs[0]]
            return mock_spend_logs

        async def count(self, *args, **kwargs):
            # Return count based on key_hash filter
            if (
                "where" in kwargs
                and "api_key" in kwargs["where"]
                and kwargs["where"]["api_key"] == "sk-test-key-1"
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

    # Make the request with key_hash filter
    response = client.get(
        "/spend/logs/ui",
        params={
            "api_key": "sk-test-key-1",
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
    assert data["data"][0]["api_key"] == "sk-test-key-1"


class TestSpendLogsPayload:
    @pytest.mark.asyncio
    async def test_spend_logs_payload_e2e(self):
        litellm.callbacks = [_ProxyDBLogger(message_logging=False)]
        # litellm._turn_on_debug()

        with patch.object(
            litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter,
            "_insert_spend_log_to_db",
        ) as mock_client, patch.object(litellm.proxy.proxy_server, "prisma_client"):
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello, world!"}],
                mock_response="Hello, world!",
                metadata={"user_api_key_end_user_id": "test_user_1"},
            )

            assert response.choices[0].message.content == "Hello, world!"

            await asyncio.sleep(1)

            mock_client.assert_called_once()

            kwargs = mock_client.call_args.kwargs
            payload: SpendLogsPayload = kwargs["payload"]
            expected_payload = SpendLogsPayload(
                **{
                    "request_id": "chatcmpl-34df56d5-4807-45c1-bb99-61e52586b802",
                    "session_id": "1234567890",
                    "call_type": "acompletion",
                    "api_key": "",
                    "cache_hit": "None",
                    "startTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 975883, tzinfo=datetime.timezone.utc
                    ),
                    "endTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "completionStartTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "model": "gpt-4o",
                    "user": "",
                    "team_id": "",
                    "metadata": '{"applied_guardrails": [], "batch_models": null, "mcp_tool_call_metadata": null, "vector_store_request_metadata": null, "guardrail_information": null, "usage_object": {"completion_tokens": 20, "prompt_tokens": 10, "total_tokens": 30, "completion_tokens_details": null, "prompt_tokens_details": null}, "model_map_information": {"model_map_key": "gpt-4o", "model_map_value": {"key": "gpt-4o", "max_tokens": 16384, "max_input_tokens": 128000, "max_output_tokens": 16384, "input_cost_per_token": 2.5e-06, "cache_creation_input_token_cost": null, "cache_read_input_token_cost": 1.25e-06, "input_cost_per_character": null, "input_cost_per_token_above_128k_tokens": null, "input_cost_per_token_above_200k_tokens": null, "input_cost_per_query": null, "input_cost_per_second": null, "input_cost_per_audio_token": null, "input_cost_per_token_batches": 1.25e-06, "output_cost_per_token_batches": 5e-06, "output_cost_per_token": 1e-05, "output_cost_per_audio_token": null, "output_cost_per_character": null, "output_cost_per_token_above_128k_tokens": null, "output_cost_per_character_above_128k_tokens": null, "output_cost_per_token_above_200k_tokens": null, "output_cost_per_second": null, "output_cost_per_reasoning_token": null, "output_cost_per_image": null, "output_vector_size": null, "litellm_provider": "openai", "mode": "chat", "supports_system_messages": true, "supports_response_schema": true, "supports_vision": true, "supports_function_calling": true, "supports_tool_choice": true, "supports_assistant_prefill": false, "supports_prompt_caching": true, "supports_audio_input": false, "supports_audio_output": false, "supports_pdf_input": false, "supports_embedding_image_input": false, "supports_native_streaming": null, "supports_web_search": true, "supports_reasoning": false, "search_context_cost_per_query": {"search_context_size_low": 0.03, "search_context_size_medium": 0.035, "search_context_size_high": 0.05}, "tpm": null, "rpm": null, "supported_openai_params": ["frequency_penalty", "logit_bias", "logprobs", "top_logprobs", "max_tokens", "max_completion_tokens", "modalities", "prediction", "n", "presence_penalty", "seed", "stop", "stream", "stream_options", "temperature", "top_p", "tools", "tool_choice", "function_call", "functions", "max_retries", "extra_headers", "parallel_tool_calls", "audio", "response_format", "user"]}}, "additional_usage_values": {"completion_tokens_details": null, "prompt_tokens_details": null}}',
                    "cache_key": "Cache OFF",
                    "spend": 0.00022500000000000002,
                    "total_tokens": 30,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "request_tags": "[]",
                    "end_user": "test_user_1",
                    "api_base": "",
                    "model_group": "",
                    "model_id": "",
                    "requester_ip_address": None,
                    "custom_llm_provider": "openai",
                    "messages": "{}",
                    "response": "{}",
                    "proxy_server_request": "{}",
                    "status": "success",
                    "mcp_namespaced_tool_name": None,
                    "agent_id": None,
                }
            )

            differences = _compare_nested_dicts(
                payload, expected_payload, ignore_keys=ignored_keys
            )
            if differences:
                assert False, f"Dictionary mismatch: {differences}"

    def mock_anthropic_response(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "content": [{"text": "Hi! My name is Claude.", "type": "text"}],
            "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
            "model": "claude-3-7-sonnet-20250219",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {"input_tokens": 2095, "output_tokens": 503},
        }
        return mock_response

    @pytest.mark.asyncio
    async def test_spend_logs_payload_success_log_with_api_base(self, monkeypatch):
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        litellm.callbacks = [_ProxyDBLogger(message_logging=False)]
        # litellm._turn_on_debug()

        client = AsyncHTTPHandler()

        with patch.object(
            litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter,
            "_insert_spend_log_to_db",
        ) as mock_client, patch.object(
            litellm.proxy.proxy_server, "prisma_client"
        ), patch.object(
            client, "post", side_effect=self.mock_anthropic_response
        ):
            response = await litellm.acompletion(
                model="claude-3-7-sonnet-20250219",
                messages=[{"role": "user", "content": "Hello, world!"}],
                metadata={"user_api_key_end_user_id": "test_user_1"},
                client=client,
            )

            assert response.choices[0].message.content == "Hi! My name is Claude."

            await asyncio.sleep(1)

            mock_client.assert_called_once()

            kwargs = mock_client.call_args.kwargs
            payload: SpendLogsPayload = kwargs["payload"]
            expected_payload = SpendLogsPayload(
                **{
                    "request_id": "chatcmpl-34df56d5-4807-45c1-bb99-61e52586b802",
                    "call_type": "acompletion",
                    "api_key": "",
                    "cache_hit": "None",
                    "startTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 975883, tzinfo=datetime.timezone.utc
                    ),
                    "endTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "completionStartTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "model": "claude-3-7-sonnet-20250219",
                    "user": "",
                    "team_id": "",
                    "metadata": '{"applied_guardrails": [], "batch_models": null, "mcp_tool_call_metadata": null, "vector_store_request_metadata": null, "guardrail_information": null, "usage_object": {"completion_tokens": 503, "prompt_tokens": 2095, "total_tokens": 2598, "completion_tokens_details": null, "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "model_map_information": {"model_map_key": "claude-3-7-sonnet-20250219", "model_map_value": {"key": "claude-3-7-sonnet-20250219", "max_tokens": 128000, "max_input_tokens": 200000, "max_output_tokens": 128000, "input_cost_per_token": 3e-06, "cache_creation_input_token_cost": 3.75e-06, "cache_read_input_token_cost": 3e-07, "input_cost_per_character": null, "input_cost_per_token_above_128k_tokens": null, "input_cost_per_token_above_200k_tokens": null, "input_cost_per_query": null, "input_cost_per_second": null, "input_cost_per_audio_token": null, "input_cost_per_token_batches": null, "output_cost_per_token_batches": null, "output_cost_per_token": 1.5e-05, "output_cost_per_audio_token": null, "output_cost_per_character": null, "output_cost_per_token_above_128k_tokens": null, "output_cost_per_character_above_128k_tokens": null, "output_cost_per_token_above_200k_tokens": null, "output_cost_per_second": null, "output_cost_per_image": null, "output_vector_size": null, "litellm_provider": "anthropic", "mode": "chat", "supports_system_messages": null, "supports_response_schema": true, "supports_vision": true, "supports_function_calling": true, "supports_tool_choice": true, "supports_assistant_prefill": true, "supports_prompt_caching": true, "supports_audio_input": false, "supports_audio_output": false, "supports_pdf_input": true, "supports_embedding_image_input": false, "supports_native_streaming": null, "supports_web_search": false, "supports_reasoning": true, "search_context_cost_per_query": null, "tpm": null, "rpm": null, "supported_openai_params": ["stream", "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens", "tools", "tool_choice", "extra_headers", "parallel_tool_calls", "response_format", "user", "reasoning_effort", "thinking"]}}, "additional_usage_values": {"completion_tokens_details": null, "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0, "text_tokens": null, "image_tokens": null}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}',
                    "cache_key": "Cache OFF",
                    "spend": 0.01383,
                    "total_tokens": 2598,
                    "prompt_tokens": 2095,
                    "completion_tokens": 503,
                    "request_tags": "[]",
                    "end_user": "test_user_1",
                    "api_base": "https://api.anthropic.com/v1/messages",
                    "model_group": "",
                    "model_id": "",
                    "requester_ip_address": None,
                    "custom_llm_provider": "anthropic",
                    "messages": "{}",
                    "response": "{}",
                    "proxy_server_request": "{}",
                    "status": "success",
                    "mcp_namespaced_tool_name": None,
                    "agent_id": None,
                }
            )

            differences = _compare_nested_dicts(
                payload, expected_payload, ignore_keys=ignored_keys
            )
            if differences:
                assert False, f"Dictionary mismatch: {differences}"

    @pytest.mark.asyncio
    async def test_spend_logs_payload_success_log_with_router(self):
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        litellm.callbacks = [_ProxyDBLogger(message_logging=False)]
        # litellm._turn_on_debug()

        client = AsyncHTTPHandler()

        router = Router(
            model_list=[
                {
                    "model_name": "my-anthropic-model-group",
                    "litellm_params": {
                        "model": "claude-3-7-sonnet-20250219",
                    },
                    "model_info": {
                        "id": "my-unique-model-id",
                    },
                }
            ]
        )

        with patch.object(
            litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter,
            "_insert_spend_log_to_db",
        ) as mock_client, patch.object(
            litellm.proxy.proxy_server, "prisma_client"
        ), patch.object(
            client, "post", side_effect=self.mock_anthropic_response
        ):
            response = await router.acompletion(
                model="my-anthropic-model-group",
                messages=[{"role": "user", "content": "Hello, world!"}],
                metadata={"user_api_key_end_user_id": "test_user_1"},
                client=client,
            )

            assert response.choices[0].message.content == "Hi! My name is Claude."

            await asyncio.sleep(1)

            mock_client.assert_called_once()

            kwargs = mock_client.call_args.kwargs
            payload: SpendLogsPayload = kwargs["payload"]
            expected_payload = SpendLogsPayload(
                **{
                    "request_id": "chatcmpl-34df56d5-4807-45c1-bb99-61e52586b802",
                    "call_type": "acompletion",
                    "api_key": "",
                    "cache_hit": "None",
                    "startTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 975883, tzinfo=datetime.timezone.utc
                    ),
                    "endTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "completionStartTime": datetime.datetime(
                        2025, 3, 24, 22, 2, 42, 989132, tzinfo=datetime.timezone.utc
                    ),
                    "model": "claude-3-7-sonnet-20250219",
                    "user": "",
                    "team_id": "",
                    "metadata": '{"applied_guardrails": [], "batch_models": null, "mcp_tool_call_metadata": null, "vector_store_request_metadata": null, "guardrail_information": null, "usage_object": {"completion_tokens": 503, "prompt_tokens": 2095, "total_tokens": 2598, "completion_tokens_details": null, "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, "model_map_information": {"model_map_key": "claude-3-7-sonnet-20250219", "model_map_value": {"key": "claude-3-7-sonnet-20250219", "max_tokens": 128000, "max_input_tokens": 200000, "max_output_tokens": 128000, "input_cost_per_token": 3e-06, "cache_creation_input_token_cost": 3.75e-06, "cache_read_input_token_cost": 3e-07, "input_cost_per_character": null, "input_cost_per_token_above_128k_tokens": null, "input_cost_per_token_above_200k_tokens": null, "input_cost_per_query": null, "input_cost_per_second": null, "input_cost_per_audio_token": null, "input_cost_per_token_batches": null, "output_cost_per_token_batches": null, "output_cost_per_token": 1.5e-05, "output_cost_per_audio_token": null, "output_cost_per_character": null, "output_cost_per_token_above_128k_tokens": null, "output_cost_per_character_above_128k_tokens": null, "output_cost_per_token_above_200k_tokens": null, "output_cost_per_second": null, "output_cost_per_image": null, "output_vector_size": null, "litellm_provider": "anthropic", "mode": "chat", "supports_system_messages": null, "supports_response_schema": true, "supports_vision": true, "supports_function_calling": true, "supports_tool_choice": true, "supports_assistant_prefill": true, "supports_prompt_caching": true, "supports_audio_input": false, "supports_audio_output": false, "supports_pdf_input": true, "supports_embedding_image_input": false, "supports_native_streaming": null, "supports_web_search": false, "supports_reasoning": true, "search_context_cost_per_query": null, "tpm": null, "rpm": null, "supported_openai_params": ["stream", "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens", "tools", "tool_choice", "extra_headers", "parallel_tool_calls", "response_format", "user", "reasoning_effort", "thinking"]}}, "additional_usage_values": {"completion_tokens_details": null, "prompt_tokens_details": {"audio_tokens": null, "cached_tokens": 0, "text_tokens": null, "image_tokens": null}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}',
                    "cache_key": "Cache OFF",
                    "spend": 0.01383,
                    "total_tokens": 2598,
                    "prompt_tokens": 2095,
                    "completion_tokens": 503,
                    "request_tags": "[]",
                    "end_user": "test_user_1",
                    "api_base": "https://api.anthropic.com/v1/messages",
                    "model_group": "my-anthropic-model-group",
                    "model_id": "my-unique-model-id",
                    "requester_ip_address": None,
                    "custom_llm_provider": "anthropic",
                    "messages": "{}",
                    "response": "{}",
                    "proxy_server_request": "{}",
                    "status": "success",
                    "mcp_namespaced_tool_name": None,
                    "agent_id": None,
                }
            )

            print(f"payload: {payload}")
            print(f"expected_payload: {expected_payload}")

            differences = _compare_nested_dicts(
                payload, expected_payload, ignore_keys=ignored_keys
            )
            if differences:
                assert False, f"Dictionary mismatch: {differences}"


def _compare_nested_dicts(
    actual: dict, expected: dict, path: str = "", ignore_keys: list[str] = []
) -> list[str]:
    """Compare nested dictionaries and return a list of differences in a human-friendly format."""
    differences = []

    # Check if current path should be ignored
    if path in ignore_keys:
        return differences

    # Check for keys in actual but not in expected
    for key in actual.keys():
        current_path = f"{path}.{key}" if path else key
        if current_path not in ignore_keys and key not in expected:
            differences.append(f"Extra key in actual: {current_path}")

    for key, expected_value in expected.items():
        current_path = f"{path}.{key}" if path else key
        if current_path in ignore_keys:
            continue
        if key not in actual:
            differences.append(f"Missing key: {current_path}")
            continue

        actual_value = actual[key]

        # Try to parse JSON strings
        if isinstance(expected_value, str):
            try:
                expected_value = json.loads(expected_value)
            except json.JSONDecodeError:
                pass
        if isinstance(actual_value, str):
            try:
                actual_value = json.loads(actual_value)
            except json.JSONDecodeError:
                pass

        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            differences.extend(
                _compare_nested_dicts(
                    actual_value, expected_value, current_path, ignore_keys
                )
            )
        elif isinstance(expected_value, dict) or isinstance(actual_value, dict):
            differences.append(
                f"Type mismatch at {current_path}: expected dict, got {type(actual_value).__name__}"
            )
        else:
            # For non-dict values, only report if they're different
            if actual_value != expected_value:
                # Format the values to be more readable
                actual_str = str(actual_value)
                expected_str = str(expected_value)
                if len(actual_str) > 50 or len(expected_str) > 50:
                    actual_str = f"{actual_str[:50]}..."
                    expected_str = f"{expected_str[:50]}..."
                differences.append(
                    f"Value mismatch at {current_path}:\n  expected: {expected_str}\n  got:      {actual_str}"
                )
    return differences


@pytest.mark.asyncio
async def test_global_spend_keys_endpoint_limit_validation(client, monkeypatch):
    """
    Test to ensure that the global_spend_keys endpoint is protected against SQL injection attacks.
    Verifies that the limit parameter is properly parameterized and not directly interpolated.
    """
    # Create a simple mock for prisma client with empty response
    mock_prisma_client = MagicMock()
    mock_db = MagicMock()
    mock_query_raw = MagicMock()
    mock_query_raw.return_value = asyncio.Future()
    mock_query_raw.return_value.set_result([])
    mock_db.query_raw = mock_query_raw
    mock_prisma_client.db = mock_db
    # Apply the mock to the prisma_client module
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Call the endpoint without specifying a limit
    no_limit_response = client.get("/global/spend/keys")
    assert no_limit_response.status_code == 200
    mock_query_raw.assert_called_once_with('SELECT * FROM "Last30dKeysBySpend";')
    # Reset the mock for the next test
    mock_query_raw.reset_mock()
    # Test with valid input
    normal_limit = "10"
    good_input_response = client.get(f"/global/spend/keys?limit={normal_limit}")
    assert good_input_response.status_code == 200
    # Verify the mock was called with the correct parameters
    mock_query_raw.assert_called_once_with(
        'SELECT * FROM "Last30dKeysBySpend" LIMIT $1 ;', 10
    )
    # Reset the mock for the next test
    mock_query_raw.reset_mock()
    # Test with SQL injection payload
    sql_injection_limit = "10; DROP TABLE spend_logs; --"
    response = client.get(f"/global/spend/keys?limit={sql_injection_limit}")
    # Verify the response is a validation error (422)
    assert response.status_code == 422
    # Verify the mock was not called with the SQL injection payload
    # This confirms that the validation happens before the database query
    mock_query_raw.assert_not_called()
    # Reset the mock for the next test
    mock_query_raw.reset_mock()
    # Test with non-numeric input
    non_numeric_limit = "abc"
    response = client.get(f"/global/spend/keys?limit={non_numeric_limit}")
    assert response.status_code == 422
    mock_query_raw.assert_not_called()
    mock_query_raw.reset_mock()
    # Test with negative number
    negative_limit = "-5"
    response = client.get(f"/global/spend/keys?limit={negative_limit}")
    assert response.status_code == 422
    mock_query_raw.assert_not_called()
    mock_query_raw.reset_mock()
    # Test with zero
    zero_limit = "0"
    response = client.get(f"/global/spend/keys?limit={zero_limit}")
    assert response.status_code == 422
    mock_query_raw.assert_not_called()
    mock_query_raw.reset_mock()


@pytest.mark.asyncio
async def test_view_spend_logs_summarize_parameter(client, monkeypatch):
    """Test the new summarize parameter in the /spend/logs endpoint"""
    import datetime
    from datetime import timedelta, timezone

    # Mock spend logs data
    mock_spend_logs = [
        {
            "id": "log1",
            "request_id": "req1",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.05,
            "startTime": (
                datetime.datetime.now(timezone.utc) - timedelta(days=1)
            ).isoformat(),
            "model": "gpt-3.5-turbo",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        {
            "id": "log2",
            "request_id": "req2",
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "team_id": "team1",
            "spend": 0.10,
            "startTime": (
                datetime.datetime.now(timezone.utc) - timedelta(days=1)
            ).isoformat(),
            "model": "gpt-4",
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        },
    ]

    # Mock for unsummarized data (summarize=false)
    class MockDB:
        def __init__(self):
            self.litellm_spendlogs = self

        async def find_many(self, *args, **kwargs):
            # Return individual log entries when summarize=false
            return mock_spend_logs

        async def group_by(self, *args, **kwargs):
            # Return grouped data when summarize=true
            # Simplified mock response for grouped data
            yesterday = datetime.datetime.now(timezone.utc) - timedelta(days=1)
            return [
                {
                    "api_key": "sk-test-key",
                    "user": "test_user_1",
                    "model": "gpt-3.5-turbo",
                    "startTime": yesterday.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "_sum": {"spend": 0.05},
                },
                {
                    "api_key": "sk-test-key",
                    "user": "test_user_1",
                    "model": "gpt-4",
                    "startTime": yesterday.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "_sum": {"spend": 0.10},
                },
            ]

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()

    # Apply the monkeypatch
    mock_prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Set up test dates
    start_date = (datetime.datetime.now(timezone.utc) - timedelta(days=2)).strftime(
        "%Y-%m-%d"
    )
    end_date = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Test 1: summarize=false should return individual log entries
    response = client.get(
        "/spend/logs",
        params={
            "start_date": start_date,
            "end_date": end_date,
            "summarize": "false",
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()

    # Should return the raw log entries
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == "log1"
    assert data[1]["id"] == "log2"
    assert data[0]["request_id"] == "req1"
    assert data[1]["request_id"] == "req2"

    # Test 2: summarize=true should return grouped data
    response = client.get(
        "/spend/logs",
        params={
            "start_date": start_date,
            "end_date": end_date,
            "summarize": "true",
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()

    # Should return grouped/summarized data
    assert isinstance(data, list)
    # The structure should be different - grouped by date with aggregated spend
    assert "startTime" in data[0]
    assert "spend" in data[0]
    assert "users" in data[0]
    assert "models" in data[0]

    # Test 3: default behavior (no summarize parameter) should maintain backward compatibility
    response = client.get(
        "/spend/logs",
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()

    # Should return grouped/summarized data (same as summarize=true)
    assert isinstance(data, list)
    assert "startTime" in data[0]
    assert "spend" in data[0]
    assert "users" in data[0]
    assert "models" in data[0]


@pytest.mark.asyncio
async def test_view_spend_tags(client, monkeypatch):
    """Test the /spend/tags endpoint"""

    # Mock the prisma client and get_spend_by_tags function
    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock response data
    mock_response = [
        {"individual_request_tag": "tag1", "log_count": 10, "total_spend": 0.15},
        {"individual_request_tag": "tag2", "log_count": 5, "total_spend": 0.08},
    ]

    # Mock the get_spend_by_tags function
    async def mock_get_spend_by_tags(prisma_client, start_date=None, end_date=None):
        return mock_response

    monkeypatch.setattr(
        "litellm.proxy.spend_tracking.spend_management_endpoints.get_spend_by_tags",
        mock_get_spend_by_tags,
    )

    # Test without date filters
    response = client.get(
        "/spend/tags",
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["individual_request_tag"] == "tag1"
    assert data[0]["log_count"] == 10
    assert data[0]["total_spend"] == 0.15

    # Test with date filters
    start_date = "2024-01-01"
    end_date = "2024-01-31"

    response = client.get(
        "/spend/tags",
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


@pytest.mark.asyncio
async def test_view_spend_tags_no_database(client, monkeypatch):
    """Test /spend/tags endpoint when database is not connected"""

    # Mock prisma_client as None
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    response = client.get(
        "/spend/tags",
        headers={"Authorization": "Bearer sk-test"},
    )

    assert response.status_code == 500
    data = response.json()
    # Check the actual error message structure
    assert "error" in data
    assert "Database not connected" in data["error"]["message"]


@pytest.mark.asyncio
async def test_provider_budget_under(disable_budget_sync):
    """Test that router allows completion when under budget"""
    provider_budget_config = {
        "azure": BudgetConfig(max_budget=0.01, budget_duration="10d")
    }

    router = Router(
        enable_pre_call_checks=True,
        provider_budget_config=provider_budget_config,
        model_list=MODEL_LIST,
    )

    response = await router.acompletion(
        model="azure-gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    assert response is not None


@pytest.mark.asyncio
async def test_provider_budget_over(disable_budget_sync):
    """Test that router allows completion when over budget"""
    provider_budget_config = {
        "azure": BudgetConfig(max_budget=-0.01, budget_duration="10d")
    }

    router = Router(
        num_retries=0,
        enable_pre_call_checks=True,
        provider_budget_config=provider_budget_config,
        model_list=MODEL_LIST,
    )

    with pytest.raises(Exception) as e:
        response = await router.acompletion(
            model="azure-gpt-4o",
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
    assert "Exceeded budget for provider" in str(e.value)


@pytest.mark.asyncio
async def test_provider_budget_provider_budgets(disable_budget_sync):
    """Test that provider_budgets() returns correct values"""
    provider = "azure"
    max_budget = -0.01
    budget_duration = "10d"
    provider_budget_config = {
        provider: BudgetConfig(max_budget=max_budget, budget_duration=budget_duration)
    }

    router = Router(
        num_retries=0,
        enable_pre_call_checks=True,
        provider_budget_config=provider_budget_config,
        model_list=MODEL_LIST,
    )

    with patch("litellm.proxy.proxy_server.llm_router", router):
        response = await spend_management_endpoints.provider_budgets()
        provider_budget_response = response.providers[provider]
        assert provider_budget_response.budget_limit == max_budget
        assert provider_budget_response.time_period == budget_duration


@pytest.mark.asyncio
async def test_view_spend_logs_with_date_range_summarized(client, monkeypatch):
    """
    Tests the /spend/logs endpoint with both start_date and end_date,
    ensuring it returns summarized data and not an empty list.
    This test specifically validates the fix for dates being passed as ISO strings.
    """
    from datetime import datetime, timedelta, timezone

    # This simulates the summarized data that Prisma's `group_by` would return.
    mock_summarized_response = [
        {
            "api_key": "sk-test-key",
            "user": "test_user_1",
            "model": "gpt-4",
            "startTime": (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "_sum": {"spend": 0.15},
        }
    ]

    # This mock class will replace the real Prisma client.
    class MockDB:
        def __init__(self):
            self.litellm_spendlogs = self

        async def group_by(self, *args, **kwargs):
            # We assert that the `gte` and `lte` values are strings in ISO format.
            # If they were datetime objects, this test would fail.
            where_clause = kwargs.get("where", {})
            start_time_filter = where_clause.get("startTime", {})

            assert "gte" in start_time_filter
            assert "lte" in start_time_filter
            assert isinstance(start_time_filter["gte"], str)
            assert isinstance(start_time_filter["lte"], str)
            assert "T" in start_time_filter["gte"]  # Check for ISO format 'T' separator

            # If the assertions pass, return the mock response.
            return mock_summarized_response

    class MockPrismaClient:
        def __init__(self):
            self.db = MockDB()

    # Apply the monkeypatch to replace the real prisma_client with our mock.
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MockPrismaClient())

    # Define a date range for the test.
    start_date = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Call the endpoint with both start and end dates.
    # We don't need `summarize=true` as it's the default.
    response = client.get(
        "/spend/logs",
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
        headers={"Authorization": "Bearer sk-test"},
    )

    # ASSERTIONS
    assert response.status_code == 200
    data = response.json()

    # Check that the response is not empty and has the summarized structure.
    assert isinstance(data, list)
    assert len(data) > 0
    assert "startTime" in data[0]
    assert "spend" in data[0]
    assert "users" in data[0]
    assert "models" in data[0]
