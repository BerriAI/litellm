import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import SpecialHeaders
from litellm.proxy.middleware.prometheus_auth_middleware import PrometheusAuthMiddleware


# Fake auth functions to simulate valid and invalid auth behavior.
async def fake_valid_auth(request, api_key, **kwargs):
    # Simulate valid authentication: do nothing (i.e. pass)
    return


async def fake_invalid_auth(request, api_key, **kwargs):
    print("running fake invalid auth", request, api_key)
    # Simulate invalid auth by raising an exception.
    raise Exception("Invalid API key")


from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with the PrometheusAuthMiddleware and dummy endpoints."""
    app = FastAPI()
    # Add the PrometheusAuthMiddleware to the app.
    app.add_middleware(PrometheusAuthMiddleware)

    @app.get("/metrics")
    async def metrics():
        return {"msg": "metrics OK"}

    # Also allow /metrics/ (trailing slash)
    @app.get("/metrics/")
    async def metrics_slash():
        return {"msg": "metrics OK"}

    @app.get("/chat/completions")
    async def chat():
        return {"msg": "chat completions OK"}

    @app.get("/embeddings")
    async def embeddings():
        return {"msg": "embeddings OK"}

    return app


def test_valid_auth_metrics(app_with_middleware, monkeypatch):
    """
    Test that a request to /metrics (and /metrics/) with valid auth headers passes.
    """
    # Enable auth on metrics endpoints.
    litellm.require_auth_for_metrics_endpoint = True
    # Patch the auth function to simulate a valid authentication.
    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        fake_valid_auth,
    )

    client = TestClient(app_with_middleware)
    headers = {SpecialHeaders.openai_authorization.value: "valid"}

    # Test for /metrics (no trailing slash)
    response = client.get("/metrics", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "metrics OK"}

    # Test for /metrics/ (with trailing slash)
    response = client.get("/metrics/", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "metrics OK"}


def test_invalid_auth_metrics(app_with_middleware, monkeypatch):
    """
    Test that a request to /metrics with invalid auth headers fails with a 401.
    """
    litellm.require_auth_for_metrics_endpoint = True
    # Patch the auth function to simulate a failed authentication.
    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        fake_invalid_auth,
    )

    client = TestClient(app_with_middleware)
    headers = {SpecialHeaders.openai_authorization.value: "invalid"}

    response = client.get("/metrics", headers=headers)
    assert response.status_code == 401, response.text
    assert "Unauthorized access to metrics endpoint" in response.text


def test_metrics_auth_uses_real_auth_when_route_is_public(
    app_with_middleware, monkeypatch
):
    """
    Regression: /metrics is statically public, but require_auth_for_metrics_endpoint
    must still force the real auth path.
    """
    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-master")
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})

    client = TestClient(app_with_middleware)

    response = client.get("/metrics")

    assert response.status_code == 401, response.text
    assert "Unauthorized access to metrics endpoint" in response.text


def test_no_auth_metrics_when_disabled(app_with_middleware, monkeypatch):
    """
    Test that when require_auth_for_metrics_endpoint is False, requests to /metrics
    bypass the auth check.
    """
    litellm.require_auth_for_metrics_endpoint = False

    # To ensure auth is not run, patch the auth function with one that will raise if called.
    def should_not_be_called(*args, **kwargs):
        raise Exception("Auth should not be called")

    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        should_not_be_called,
    )

    client = TestClient(app_with_middleware)
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "metrics OK"}


def test_non_metrics_requests_pass_through(app_with_middleware):
    """
    Test that non-metrics endpoints pass through the middleware unaffected.
    """
    litellm.require_auth_for_metrics_endpoint = True

    client = TestClient(app_with_middleware)

    response = client.get("/chat/completions")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "chat completions OK"}

    response = client.get("/embeddings")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "embeddings OK"}


def test_non_metrics_requests_dont_trigger_auth(app_with_middleware, monkeypatch):
    """
    Test that non-metrics requests never trigger auth, even when auth is enabled
    and the auth function would reject the request.
    """
    litellm.require_auth_for_metrics_endpoint = True

    def should_not_be_called(*args, **kwargs):
        raise Exception("Auth should not be called for non-metrics requests")

    monkeypatch.setattr(
        "litellm.proxy.middleware.prometheus_auth_middleware.user_api_key_auth",
        should_not_be_called,
    )

    client = TestClient(app_with_middleware)

    response = client.get("/chat/completions")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "chat completions OK"}

    response = client.get("/embeddings")
    assert response.status_code == 200, response.text
    assert response.json() == {"msg": "embeddings OK"}
