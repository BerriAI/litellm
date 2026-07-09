import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
import litellm.proxy.collector_endpoints.spend_logs as collector_spend_logs
from litellm.constants import MAX_COLLECTOR_SPEND_LOG_BYTES
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app


class MockPrismaClient:
    def __init__(self):
        self.spend_log_transactions = []
        self._spend_log_transactions_lock = asyncio.Lock()


def _set_collector_runtime(monkeypatch, prisma_client):
    monkeypatch.setattr(ps, "prisma_client", prisma_client)
    monkeypatch.setattr(app.state, "prisma_client", prisma_client, raising=False)


def test_collector_spend_logs_enqueues_batcher_rows(monkeypatch):
    prisma_client = MockPrismaClient()
    _set_collector_runtime(monkeypatch, prisma_client)
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
        assert row["metadata"]["user_api_key_team_alias"] is None
        assert row["metadata"]["collector_request_id"] == "relay-test-request"
        assert row["metadata"]["app"] == "notion"
        assert row["proxy_server_request"]["body_preview"] == "hi"
        assert row["response"]["body_preview"] == "hello"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_attributes_valid_virtual_key(monkeypatch):
    prisma_client = MockPrismaClient()
    _set_collector_runtime(monkeypatch, prisma_client)
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


def test_collector_spend_logs_accepts_json_object_request_tags(monkeypatch):
    prisma_client = MockPrismaClient()
    _set_collector_runtime(monkeypatch, prisma_client)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        api_key="hashed-admin-key",
    )

    try:
        response = TestClient(app).post(
            "/collector/spend-logs",
            json={
                "logs": [
                    {
                        "request_id": "relay-json-tags-test",
                        "request_tags": '{"source":"notion"}',
                    }
                ]
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 200, response.text
        row = prisma_client.spend_log_transactions[0]
        assert json.loads(row["request_tags"]) == [
            {"source": "notion"},
            "litellm-relay",
        ]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_strips_nul_bytes(monkeypatch):
    prisma_client = MockPrismaClient()
    _set_collector_runtime(monkeypatch, prisma_client)
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
        row_text = json.dumps(prisma_client.spend_log_transactions[0], default=str)
        assert "\u0000" not in row_text
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_rejects_oversized_log(monkeypatch):
    prisma_client = MockPrismaClient()
    _set_collector_runtime(monkeypatch, prisma_client)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        api_key="hashed-admin-key",
    )

    try:
        response = TestClient(app).post(
            "/collector/spend-logs",
            json={
                "logs": [
                    {
                        "request_id": "large-relay-request",
                        "proxy_server_request": {
                            "body_preview": "x" * (MAX_COLLECTOR_SPEND_LOG_BYTES + 1)
                        },
                    }
                ]
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 413, response.text
        assert prisma_client.spend_log_transactions == []
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_rejects_normalized_row_over_size_limit(monkeypatch):
    prisma_client = MockPrismaClient()
    _set_collector_runtime(monkeypatch, prisma_client)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        api_key="hashed-admin-key",
    )

    try:
        response = TestClient(app).post(
            "/collector/spend-logs",
            json={
                "logs": [
                    {
                        "request_id": "normalized-large-relay-request",
                        "proxy_server_request": {
                            "body_preview": "x" * (MAX_COLLECTOR_SPEND_LOG_BYTES - 50)
                        },
                    }
                ]
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 413, response.text
        assert prisma_client.spend_log_transactions == []
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_rejects_oversized_batch(monkeypatch):
    prisma_client = MockPrismaClient()
    _set_collector_runtime(monkeypatch, prisma_client)
    monkeypatch.setattr(
        collector_spend_logs,
        "MAX_COLLECTOR_SPEND_LOG_BATCH_BYTES",
        900,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        api_key="hashed-admin-key",
    )

    try:
        response = TestClient(app).post(
            "/collector/spend-logs",
            json={
                "logs": [
                    {
                        "request_id": f"batch-relay-request-{idx}",
                        "proxy_server_request": {"body_preview": "x" * 200},
                    }
                    for idx in range(5)
                ]
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 413, response.text
        assert prisma_client.spend_log_transactions == []
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_rejects_when_queue_is_full(monkeypatch):
    prisma_client = MockPrismaClient()
    prisma_client.spend_log_transactions = [{} for _ in range(3)]
    _set_collector_runtime(monkeypatch, prisma_client)
    monkeypatch.setattr(
        collector_spend_logs,
        "LITELLM_ASYNCIO_QUEUE_MAXSIZE",
        3,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        api_key="hashed-admin-key",
    )

    try:
        response = TestClient(app).post(
            "/collector/spend-logs",
            json={
                "logs": [
                    {
                        "request_id": "queued-relay-request",
                    }
                ]
            },
            headers={"Authorization": "Bearer sk-test"},
        )

        assert response.status_code == 429, response.text
        assert response.json()["detail"] == {
            "error": "Collector spend-log queue is full",
            "queued": 3,
            "limit": 3,
        }
        assert len(prisma_client.spend_log_transactions) == 3
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_collector_spend_logs_enqueue_is_capacity_checked_under_lock(monkeypatch):
    prisma_client = MockPrismaClient()
    monkeypatch.setattr(
        collector_spend_logs,
        "LITELLM_ASYNCIO_QUEUE_MAXSIZE",
        3,
    )

    async def enqueue_two_batches():
        await collector_spend_logs._enqueue_collector_spend_logs(
            prisma_client=prisma_client,
            spend_logs=[{"request_id": "one"}, {"request_id": "two"}],
        )
        with pytest.raises(collector_spend_logs.HTTPException) as exc_info:
            await collector_spend_logs._enqueue_collector_spend_logs(
                prisma_client=prisma_client,
                spend_logs=[{"request_id": "three"}, {"request_id": "four"}],
            )
        return exc_info.value

    error = asyncio.run(enqueue_two_batches())

    assert error.status_code == 429
    assert error.detail == {
        "error": "Collector spend-log queue is full",
        "queued": 2,
        "limit": 3,
    }
    assert [row["request_id"] for row in prisma_client.spend_log_transactions] == [
        "one",
        "two",
    ]
