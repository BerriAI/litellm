"""Gateway coverage for Parallel AI's Extract pass-through."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

PARALLEL_EXTRACT_URL: Final = "https://api.parallel.ai/v1/extract"


@pytest.fixture
def client() -> TestClient:
    return TestClient(proxy_server.app, raise_server_exceptions=False)


@pytest.fixture
def auth_as() -> Iterator[None]:
    async def _authorized_request() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key="hashed-sk-test",
            user_id="parallel-test-user",
        )

    previous: Final = proxy_server.app.dependency_overrides.get(user_api_key_auth)
    proxy_server.app.dependency_overrides[user_api_key_auth] = _authorized_request
    try:
        yield
    finally:
        if previous is None:
            proxy_server.app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            proxy_server.app.dependency_overrides[user_api_key_auth] = previous


def _parallel_extract_body() -> dict[str, object]:
    return {
        "extract_id": "extract_parallel_gateway",
        "results": [
            {
                "url": "https://example.com/parallel",
                "title": "Parallel result",
                "publish_date": "2026-08-14",
                "excerpts": ["Focused excerpt"],
                "full_content": "# Full content",
            }
        ],
        "errors": [
            {
                "url": "https://example.com/unavailable",
                "error_type": "fetch_error",
                "http_status_code": 503,
                "content": "Upstream unavailable",
            }
        ],
        "warnings": None,
        "usage": [{"name": "sku_extract_excerpts", "count": 2}],
        "session_id": "session_parallel_gateway",
    }


def _parallel_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "parallel-gateway",
                "litellm_params": {
                    "model": "parallel_ai/parallel",
                    "api_key": "parallel-responses-key",
                    "use_in_pass_through": True,
                },
            }
        ],
        num_retries=0,
    )


def test_parallel_extract_gateway_route(client, auth_as, monkeypatch, respx_mock):
    """The native Extract route preserves Parallel's V1 request and partial-success response."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(proxy_server, "llm_router", _parallel_router())
    upstream_route = respx_mock.post(PARALLEL_EXTRACT_URL).respond(json=_parallel_extract_body())
    request_body = {
        "urls": [
            "https://example.com/parallel",
            "https://example.com/unavailable",
        ],
        "objective": "Find the integration details",
        "search_queries": ["Parallel integration"],
        "max_chars_total": 50000,
        "session_id": "session_parallel_gateway",
        "client_model": "gpt-5.4",
        "advanced_settings": {
            "fetch_policy": {
                "max_age_seconds": 3600,
                "timeout_seconds": 30,
                "disable_cache_fallback": False,
            },
            "excerpt_settings": {"max_chars_per_result": 5000},
            "full_content": {"max_chars_per_result": 50000},
        },
    }

    response = client.post("/parallel_ai/v1/extract", json=request_body)

    assert response.status_code == 200, response.text
    assert response.json() == _parallel_extract_body()
    assert upstream_route.called

    upstream_request = upstream_route.calls.last.request
    assert upstream_request.headers["x-api-key"] == "parallel-responses-key"
    assert "authorization" not in upstream_request.headers
    assert json.loads(upstream_request.content) == request_body


def test_parallel_extract_route_is_classified_as_an_llm_api_route() -> None:
    assert RouteChecks.is_llm_api_route(route="/parallel_ai/v1/extract") is True


def test_parallel_extract_gateway_uses_environment_configuration(client, auth_as, monkeypatch, respx_mock):
    custom_url = "https://parallel-proxy.example.com/v1/extract"
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
    monkeypatch.setenv("PARALLEL_API_KEY", "parallel-env-key")
    monkeypatch.setenv("PARALLEL_AI_API_BASE", "https://parallel-proxy.example.com/v1")
    upstream_route = respx_mock.post(custom_url).respond(json=_parallel_extract_body())

    response = client.post(
        "/parallel_ai/v1/extract",
        json={"urls": ["https://example.com/parallel"]},
    )

    assert response.status_code == 200, response.text
    assert upstream_route.called
    assert upstream_route.calls.last.request.headers["x-api-key"] == "parallel-env-key"


def test_parallel_extract_gateway_requires_a_provider_key(client, auth_as, monkeypatch) -> None:
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    response = client.post(
        "/parallel_ai/v1/extract",
        json={"urls": ["https://example.com/parallel"]},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "PARALLEL_AI_API_KEY or PARALLEL_API_KEY is required for the Parallel AI Extract pass-through."
    )


def test_parallel_extract_gateway_preserves_validation_errors(client, auth_as, monkeypatch, respx_mock):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(proxy_server, "llm_router", _parallel_router())
    error_body = {
        "error": {
            "type": "validation_error",
            "message": "urls must contain at most 20 items",
        }
    }
    respx_mock.post(PARALLEL_EXTRACT_URL).respond(status_code=422, json=error_body)

    response = client.post(
        "/parallel_ai/v1/extract",
        json={"urls": [f"https://example.com/{index}" for index in range(21)]},
    )

    assert response.status_code == 422
    assert response.json() == error_body
