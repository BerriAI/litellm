from collections.abc import Iterator
from typing import Final
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.credential_endpoints.nim_model_discovery import nim_http_client, router
from litellm.types.utils import CredentialItem


@pytest.fixture
def app() -> FastAPI:
    application: Final = FastAPI()
    application.include_router(router)
    application.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with (
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch.object(litellm, "credential_list", []),
        patch("litellm.proxy.credential_endpoints.nim_model_discovery._CACHE", InMemoryCache()),
    ):
        with TestClient(app) as test_client:
            yield test_client


def test_saved_credential_controls_the_upstream_and_only_ids_return(client: TestClient, app: FastAPI) -> None:
    credential: Final = CredentialItem(
        credential_name="nim-saved",
        credential_info={"custom_llm_provider": "nvidia_nim"},
        credential_values={"api_key": "saved-secret", "api_base": "https://nim.example/v1"},
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://nim.example/v1/models"
        assert request.headers["authorization"] == "Bearer saved-secret"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "new-model"}, {"id": "new-model"}]})

    async def connection() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(upstream))

    app.dependency_overrides[nim_http_client] = connection
    with patch.object(litellm, "credential_list", [credential]):
        response: Final = client.post("/credentials/nvidia_nim/models", json={"credential_name": "nim-saved"})
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "success", "models": ["nvidia_nim/new-model"]}


@pytest.mark.parametrize("status", (401, 403, 429, 500))
def test_upstream_errors_allow_catalog_fallback_without_leaking_secrets(
    client: TestClient, app: FastAPI, status: int
) -> None:
    async def connection() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json={"error": "private-details"}))
        )

    app.dependency_overrides[nim_http_client] = connection
    response: Final = client.post(
        "/credentials/nvidia_nim/models", json={"credential_values": {"api_key": "private-key"}}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "unavailable", "models": []}


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
def test_only_admins_can_use_saved_credentials_for_discovery(
    client: TestClient, app: FastAPI, role: LitellmUserRoles
) -> None:
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    response: Final = client.post("/credentials/nvidia_nim/models", json={"credential_name": "nim-saved"})
    assert response.status_code == 403, response.text


@pytest.mark.parametrize("override", ({"api_base": "https://elsewhere.example/v1"}, {"api_key": "replacement"}))
def test_cannot_redirect_a_saved_credential(client: TestClient, override: dict[str, str]) -> None:
    response: Final = client.post(
        "/credentials/nvidia_nim/models", json={"credential_name": "nim-saved", "credential_values": override}
    )
    assert response.status_code == 400, response.text


def test_missing_credential_is_not_replaced_with_environment_credentials(client: TestClient, app: FastAPI) -> None:
    response: Final = client.post("/credentials/nvidia_nim/models", json={"credential_name": "missing"})
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("listing", ({"data": []}, {"data": [{"id": 42}]}, {"data": None}))
def test_empty_or_malformed_listing_allows_catalog_fallback(
    client: TestClient, app: FastAPI, listing: dict[str, object]
) -> None:
    async def connection() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=listing)))

    app.dependency_overrides[nim_http_client] = connection
    response: Final = client.post("/credentials/nvidia_nim/models", json={"credential_values": {"api_key": "test-key"}})
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "unavailable", "models": []}


def test_cached_models_are_scoped_to_endpoint_and_key(client: TestClient, app: FastAPI) -> None:
    async def success() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "cached"}]}))
        )

    async def failure() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503)))

    app.dependency_overrides[nim_http_client] = success
    original: Final = {"api_base": "https://nim.example/v1", "api_key": "first-key"}
    first: Final = client.post("/credentials/nvidia_nim/models", json={"credential_values": original})
    assert first.json() == {"status": "success", "models": ["nvidia_nim/cached"]}
    app.dependency_overrides[nim_http_client] = failure
    cached: Final = client.post("/credentials/nvidia_nim/models", json={"credential_values": original})
    assert cached.json() == first.json()
    other_key: Final = client.post(
        "/credentials/nvidia_nim/models", json={"credential_values": {**original, "api_key": "other-key"}}
    )
    assert other_key.json() == {"status": "unavailable", "models": []}
    other_endpoint: Final = client.post(
        "/credentials/nvidia_nim/models",
        json={"credential_values": {**original, "api_base": "https://other.example/v1"}},
    )
    assert other_endpoint.json() == {"status": "unavailable", "models": []}
