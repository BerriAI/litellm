from datetime import datetime, timedelta
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from litellm_enterprise.proxy.audit_logging_endpoints import router as audit_router
from litellm_enterprise.types.proxy.audit_logging_endpoints import AuditLogResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

# Create an app with just the audit router for testing
app = FastAPI()
app.include_router(audit_router)
app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role="proxy_admin")
client = TestClient(app)

# Mock data for testing
MOCK_AUDIT_LOG = {
    "id": "test-audit-log-1",
    "updated_at": datetime.utcnow(),
    "changed_by": "test-user",
    "changed_by_api_key": "test-api-key-hash",
    "action": "create",
    "table_name": "test_table",
    "object_id": "test-object-1",
    "before_value": None,
    "updated_values": {"name": "test", "value": 123},
}


@pytest.fixture
def mock_prisma_client():
    with patch("litellm.proxy.proxy_server.prisma_client") as mock:
        mock.db.litellm_auditlog.find_many = AsyncMock()
        mock.db.litellm_auditlog.find_unique = AsyncMock()
        mock.db.litellm_auditlog.count = AsyncMock()
        yield mock


@pytest.mark.asyncio
async def test_get_audit_logs(mock_prisma_client):
    """Test successful retrieval of audit logs with pagination"""
    # Mock the database responses
    mock_prisma_client.db.litellm_auditlog.find_many.return_value = [
        AuditLogResponse(**MOCK_AUDIT_LOG)
    ]
    mock_prisma_client.db.litellm_auditlog.count.return_value = 1

    # Mock the auth dependency
    with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth") as mock_auth:
        mock_auth.return_value = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id=None,
            organization_id=None,
            user_role="proxy_admin",
        )

        # Make the request
        response = client.get("/audit?page=1&page_size=10")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert "audit_logs" in data
        assert len(data["audit_logs"]) == 1
        assert data["total"] == 1
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert data["total_pages"] == 1

        # Verify the audit log data
        audit_log = data["audit_logs"][0]
        assert audit_log["id"] == MOCK_AUDIT_LOG["id"]
        assert audit_log["action"] == MOCK_AUDIT_LOG["action"]
        assert audit_log["table_name"] == MOCK_AUDIT_LOG["table_name"]


@pytest.mark.asyncio
async def test_get_audit_log_by_id(mock_prisma_client):
    """Test successful retrieval of a specific audit log by ID"""
    # Mock the database response
    mock_prisma_client.db.litellm_auditlog.find_unique.return_value = AuditLogResponse(
        **MOCK_AUDIT_LOG
    )

    # Mock the auth dependency
    with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth") as mock_auth:
        mock_auth.return_value = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id=None,
            organization_id=None,
            user_role="proxy_admin",
        )

        # Make the request
        response = client.get(f"/audit/{MOCK_AUDIT_LOG['id']}")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == MOCK_AUDIT_LOG["id"]
        assert data["action"] == MOCK_AUDIT_LOG["action"]
        assert data["table_name"] == MOCK_AUDIT_LOG["table_name"]
        assert data["object_id"] == MOCK_AUDIT_LOG["object_id"]


@pytest.mark.asyncio
async def test_get_audit_log_by_id_not_found(mock_prisma_client):
    """Test error handling when audit log is not found"""
    # Mock the database response to return None
    mock_prisma_client.db.litellm_auditlog.find_unique.return_value = None

    # Mock the auth dependency
    with patch("litellm.proxy.auth.user_api_key_auth.user_api_key_auth") as mock_auth:
        mock_auth.return_value = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id=None,
            organization_id=None,
            user_role="proxy_admin",
        )

        # Make the request
        response = client.get("/audit/non-existent-id")

        # Assert response
        assert response.status_code == 404
        data = response.json()
        assert "message" in data["detail"]
        assert "not found" in data["detail"]["message"].lower()


def _list_audit_logs_where(mock_prisma_client: MagicMock, query: str) -> dict[str, object]:
    mock_prisma_client.db.litellm_auditlog.find_many.return_value = []
    mock_prisma_client.db.litellm_auditlog.count.return_value = 0

    response: Final = client.get(f"/audit?{query}")

    assert response.status_code == 200, response.text
    find_many_where: Final = mock_prisma_client.db.litellm_auditlog.find_many.call_args.kwargs["where"]
    assert mock_prisma_client.db.litellm_auditlog.count.call_args.kwargs["where"] == find_many_where
    return find_many_where


def test_search_matches_any_id_column_alongside_the_other_filters(mock_prisma_client):
    where: Final = _list_audit_logs_where(mock_prisma_client, "search=abc-123&action=create&object_team_id=team-1")

    assert where == {
        "action": "create",
        "AND": (
            {
                "OR": [
                    {"before_value": {"path": ["team_id"], "string_contains": "team-1"}},
                    {"updated_values": {"path": ["team_id"], "string_contains": "team-1"}},
                ]
            },
            {
                "OR": (
                    {"id": "abc-123"},
                    {"changed_by": "abc-123"},
                    {"object_id": "abc-123"},
                    {"changed_by_api_key": "abc-123"},
                )
            },
        ),
    }


def test_an_empty_search_leaves_the_where_clause_unchanged(mock_prisma_client):
    where: Final = _list_audit_logs_where(mock_prisma_client, "action=create&search=")

    assert where == {"action": "create"}
