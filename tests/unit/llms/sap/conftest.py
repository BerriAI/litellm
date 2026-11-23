import json
import uuid
from dataclasses import dataclass

import httpx
import pytest
import respx

import litellm


@dataclass(frozen=True, slots=True)
class MockAICore:
    deployment_url: str
    token: respx.Route
    deployments: respx.Route

    def reject_token(self, route: respx.Route, authorization: str) -> None:
        accepted = route.return_value
        route.side_effect = lambda request: (
            httpx.Response(401, text="Unauthorized") if request.headers["Authorization"] == authorization else accepted
        )


@pytest.fixture
def aicore(respx_mock, monkeypatch) -> MockAICore:
    for var in (
        "AICORE_CLIENT_ID",
        "AICORE_CLIENT_SECRET",
        "AICORE_AUTH_URL",
        "AICORE_RESOURCE_GROUP",
        "AICORE_BASE_URL",
        "AICORE_CERT_URL",
        "AICORE_SERVICE_KEY",
        "VCAP_SERVICES",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AICORE_HOME", "notexist")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    host = f"https://{uuid.uuid4().hex}.mock-sap.com"
    service_key = {
        "clientid": "mock-client",
        "clientsecret": "mock-secret",
        "url": f"{host}/auth",
        "serviceurls": {"AI_API_URL": f"{host}/api"},
    }
    monkeypatch.setattr(litellm, "sap_service_key", json.dumps(service_key))

    base_url = f"{host}/api/v2"
    deployment_url = f"{base_url}/inference/deployments/orchestration"
    deployments = respx_mock.get(f"{base_url}/lm/deployments").respond(
        json={
            "resources": [
                {
                    "configurationId": "orchestration-config",
                    "scenarioId": "orchestration",
                    "deploymentUrl": deployment_url,
                    "createdAt": "2026-01-01T00:00:00Z",
                }
            ]
        }
    )
    respx_mock.get(f"{base_url}/lm/configurations/orchestration-config").respond(json={"executableId": "orchestration"})
    return MockAICore(
        deployment_url=deployment_url,
        token=respx_mock.post(f"{host}/auth/oauth/token"),
        deployments=deployments,
    )
