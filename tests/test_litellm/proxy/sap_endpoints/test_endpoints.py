import asyncio
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.sap_endpoints.endpoints import (
    SapDeploymentInfo,
    _default_discovery,
    discover_running_deployments,
    router,
)

_MODEL = "anthropic--claude-4.8-opus"
_BASE_URL = "https://api.example.com/v2"
_RESOURCE_GROUP = "default"
_DEPLOYMENT_URL = "https://api.example.com/v2/inference/deployments/d38af17dc133a768"

_PAYLOAD = {
    "count": 2,
    "resources": [
        {
            "id": "d38af17dc133a768",
            "createdAt": "2026-07-14T02:44:50Z",
            "status": "RUNNING",
            "scenarioId": "foundation-models",
            "deploymentUrl": _DEPLOYMENT_URL,
            "details": {"resources": {"backendDetails": {"model": {"name": _MODEL, "version": "1"}}}},
        },
        {
            "id": "no-backend",
            "createdAt": "2026-07-14T02:44:50Z",
            "status": "RUNNING",
            "scenarioId": "foundation-models",
            "deploymentUrl": "https://api.example.com/v2/inference/deployments/no-backend",
            "details": None,
        },
    ],
}


class _FakeAsyncResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return _FakeAsyncResponse(self._payload)


def _fake_token_factory(service_key, *, resource_group=None):
    return (lambda: "Bearer test-token", _BASE_URL, resource_group or _RESOURCE_GROUP)


def test_discover_returns_named_running_deployments_and_forwards_auth_headers():
    client = _FakeAsyncClient(_PAYLOAD)
    result = asyncio.run(
        discover_running_deployments(
            service_key="svc-key",
            resource_group=None,
            token_creator_factory=_fake_token_factory,
            http_client=client,
        )
    )
    assert [d.model_name for d in result] == [_MODEL]
    assert result[0].deployment_url == _DEPLOYMENT_URL
    assert result[0].id == "d38af17dc133a768"
    call = client.calls[0]
    assert call["url"] == f"{_BASE_URL}/lm/deployments"
    assert call["params"] == {"scenarioId": "foundation-models", "status": "RUNNING"}
    assert call["headers"]["Authorization"] == "Bearer test-token"
    assert call["headers"]["AI-Resource-Group"] == _RESOURCE_GROUP


def test_discover_forwards_non_default_resource_group_to_ai_resource_group_header():
    client = _FakeAsyncClient(_PAYLOAD)
    asyncio.run(
        discover_running_deployments(
            service_key="svc-key",
            resource_group="team-a",
            token_creator_factory=_fake_token_factory,
            http_client=client,
        )
    )
    assert client.calls[0]["headers"]["AI-Resource-Group"] == "team-a"


def _client(discover):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: None
    app.dependency_overrides[_default_discovery] = lambda: discover
    return TestClient(app, raise_server_exceptions=False)


def test_endpoint_returns_deployment_list():
    async def discover(*, service_key, resource_group, token_creator_factory, http_client):
        return (
            SapDeploymentInfo(
                model_name=_MODEL,
                deployment_url=_DEPLOYMENT_URL,
                id="d38af17dc133a768",
                status="RUNNING",
                created_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            ),
        )

    resp = _client(discover).post("/sap/deployments", json={"service_key": "svc-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deployments"][0]["model_name"] == _MODEL
    assert body["deployments"][0]["deployment_url"] == _DEPLOYMENT_URL


def test_endpoint_forwards_request_resource_group_to_discovery():
    seen = {}

    async def discover(*, service_key, resource_group, token_creator_factory, http_client):
        seen["resource_group"] = resource_group
        return ()

    resp = _client(discover).post("/sap/deployments", json={"service_key": "svc-key", "resource_group": "team-a"})
    assert resp.status_code == 200
    assert seen["resource_group"] == "team-a"


def test_endpoint_rejects_os_environ_references_before_discovery():
    called = {"hit": False}

    async def discover(*, service_key, resource_group, token_creator_factory, http_client):
        called["hit"] = True
        return ()

    resp = _client(discover).post("/sap/deployments", json={"service_key": "os.environ/AICORE_SERVICE_KEY"})
    assert resp.status_code == 400
    assert called["hit"] is False


def test_endpoint_maps_invalid_service_key_to_400():
    async def discover(*, service_key, resource_group, token_creator_factory, http_client):
        raise ValueError("SAP AI Core credentials not found.")

    resp = _client(discover).post("/sap/deployments", json={"service_key": "svc-key"})
    assert resp.status_code == 400


def test_endpoint_maps_token_failure_to_502():
    async def discover(*, service_key, resource_group, token_creator_factory, http_client):
        raise RuntimeError("Token request failed: 401 unauthorized")

    resp = _client(discover).post("/sap/deployments", json={"service_key": "svc-key"})
    assert resp.status_code == 502


def test_endpoint_maps_discovery_http_error_to_502():
    async def discover(*, service_key, resource_group, token_creator_factory, http_client):
        raise httpx.ConnectError("cannot reach host")

    resp = _client(discover).post("/sap/deployments", json={"service_key": "svc-key"})
    assert resp.status_code == 502
