import json
import uuid
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
import respx

import litellm
import litellm.llms.sap.deployment as _depl


@dataclass(frozen=True, slots=True)
class MockAICore:
    deployment_url: str
    token: respx.Route
    deployments: respx.Route

    def reject_token(self, route: respx.Route, authorization: str) -> None:
        accepted: Final = route.return_value
        route.side_effect = lambda request: (
            httpx.Response(401, text="Unauthorized") if request.headers["Authorization"] == authorization else accepted
        )


@pytest.fixture
def aicore(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> MockAICore:
    _depl._deployment_url_cache.clear()

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
    monkeypatch.delenv("AICORE_ORCHESTRATION_DEPLOYMENT_URL", raising=False)
    monkeypatch.setenv("AICORE_HOME", "notexist")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    host: Final = f"https://{uuid.uuid4().hex}.mock-sap.com"
    service_key: Final = {
        "clientid": "mock-client",
        "clientsecret": "mock-secret",
        "url": f"{host}/auth",
        "serviceurls": {"AI_API_URL": f"{host}/api"},
    }
    monkeypatch.setattr(litellm, "sap_service_key", json.dumps(service_key))

    base_url: Final = f"{host}/api/v2"
    deployment_url: Final = f"{base_url}/inference/deployments/orchestration"
    deployments: Final = respx_mock.get(f"{base_url}/lm/deployments").respond(
        json={
            "resources": [
                {
                    "configurationId": "orchestration-config",
                    "configurationName": "orchestration",
                    "scenarioId": "orchestration",
                    "deploymentUrl": deployment_url,
                    "createdAt": "2026-01-01T00:00:00Z",
                }
            ]
        }
    )
    return MockAICore(
        deployment_url=deployment_url,
        token=respx_mock.post(f"{host}/auth/oauth/token"),
        deployments=deployments,
    )
