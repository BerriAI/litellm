import asyncio
import json

from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app


class MockPrismaClient:
    def __init__(self):
        self.spend_log_transactions = []
        self._spend_log_transactions_lock = asyncio.Lock()


def test_collector_spend_logs_enqueues_batcher_rows(monkeypatch):
    prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        api_key="hashed-admin-key",
        team_id="team-1",
    )

    try:
        response = TestClient(app).post(
            "/collector/spend-logs",
            json={
                "logs": [
                    {
                        "request_id": "relay-test-request",
                        "model": "notion-ai",
                        "metadata": {"app": "notion", "host": "www.notion.so"},
                        "proxy_server_request": {
                            "method": "POST",
                            "body_preview": "hi",
                        },
                        "response": {
                            "status_code": 200,
                            "body_preview": "hello",
                        },
                        "request_duration_ms": 42,
                    }
                ]
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200, response.text
        assert response.json() == {"enqueued": 1}
        assert len(prisma_client.spend_log_transactions) == 1
        row = prisma_client.spend_log_transactions[0]
        assert row["request_id"].startswith("collector-")
        assert row["request_id"] != "relay-test-request"
        assert row["call_type"] == "litellm-relay"
        assert row["model"] == "notion-ai"
        assert row["spend"] == 0.0
        assert row["team_id"] == "team-1"
        assert row["metadata"]["source"] == "litellm-relay"
        assert row["metadata"]["collector_request_id"] == "relay-test-request"
        assert row["metadata"]["app"] == "notion"
        assert row["proxy_server_request"]["body_preview"] == "hi"
        assert row["response"]["body_preview"] == "hello"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_attributes_valid_virtual_key(monkeypatch):
    prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="user_1",
        api_key="hashed-virtual-key",
        team_id="team_1",
        team_alias="Relay Team",
        key_alias="relay-key",
        organization_id="org_1",
    )

    try:
        response = TestClient(app).post(
            "/collector/spend-logs",
            json={
                "logs": [
                    {
                        "request_id": "relay-test-request",
                        "api_key": "client-supplied-key-is-ignored",
                        "spend": 10,
                        "team_id": "client-team-is-ignored",
                        "organization_id": "client-org-is-ignored",
                        "user": "client-user-is-ignored",
                        "custom_llm_provider": "client-provider-is-ignored",
                        "metadata": {
                            "source": "client-source-is-ignored",
                            "user_api_key": "client-key-is-ignored",
                        },
                    }
                ]
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200, response.text
        row = prisma_client.spend_log_transactions[0]
        assert row["api_key"] == "hashed-virtual-key"
        assert row["spend"] == 0.0
        assert row["team_id"] == "team_1"
        assert row["organization_id"] == "org_1"
        assert row["user"] == "user_1"
        assert row["custom_llm_provider"] == ""
        assert row["metadata"]["source"] == "litellm-relay"
        assert row["metadata"]["user_api_key"] == "hashed-virtual-key"
        assert row["metadata"]["user_api_key_alias"] == "relay-key"
        assert row["metadata"]["user_api_key_team_alias"] == "Relay Team"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_strips_nul_bytes(monkeypatch):
    prisma_client = MockPrismaClient()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="user_1",
        api_key="hashed-virtual-key",
    )

    try:
        response = TestClient(app).post(
            "/collector/spend-logs",
            json={
                "logs": [
                    {
                        "request_id": "relay-nul-test",
                        "proxy_server_request": {"body_preview": "hello\u0000world"},
                        "response": {"body_preview": "ok\u0000"},
                    }
                ]
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200, response.text
        row_text = json.dumps(prisma_client.spend_log_transactions[0])
        assert "\u0000" not in row_text
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
