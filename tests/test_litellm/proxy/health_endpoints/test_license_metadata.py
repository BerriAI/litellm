"""
Unit test for verifying license metadata is returned by /health/readiness.

The test sets up a FastAPI application with the health_endpoints router,
monkey-patches authentication and license checking, and asserts that the
expected keys are present in the JSON response.
"""

import pytest


@pytest.mark.asyncio
async def test_health_readiness_contains_license_metadata(monkeypatch):
    """Ensure /health/readiness includes the `license` field with metadata."""

    # Import inside the test to avoid heavy imports before monkey-patching
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Router containing the /health/readiness endpoint
    from litellm.proxy.health_endpoints import _health_endpoints as he

    # ------------------------------------------------------------------
    # Patch authentication dependency so we don't need an API key
    # ------------------------------------------------------------------
    monkeypatch.setattr(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
        lambda *args, **kwargs: {"user_id": "test", "api_key": "test"},
        raising=True,
    )

    # ------------------------------------------------------------------
    # Patch the global _license_check used by the endpoint
    # ------------------------------------------------------------------
    from litellm.proxy import proxy_server

    class _DummyLicenseCheck:
        def __init__(self):
            self._is_premium = True
            self.airgapped_license_data = {"expiration_date": "2099-12-31"}

        def is_premium(self):
            return self._is_premium

    proxy_server._license_check = _DummyLicenseCheck()  # type: ignore

    # ------------------------------------------------------------------
    # Build test app and client
    # ------------------------------------------------------------------
    app = FastAPI()
    app.include_router(he.router)

    client = TestClient(app)

    response = client.get("/health/readiness", headers={"Authorization": "Bearer dummy"})
    assert response.status_code == 200, response.text

    data = response.json()

    # Validate presence and structure of license metadata
    assert "license" in data, "Missing 'license' key in readiness response"
    license_obj = data["license"]
    assert license_obj["has_license"] is True
    assert license_obj["expiration_date"] == "2099-12-31"

