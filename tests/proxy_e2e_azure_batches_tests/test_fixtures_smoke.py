"""
Smoke test to verify fixtures start and stop correctly.
Run this first to ensure the infrastructure works before running full E2E tests.
"""

import httpx
import pytest


pytestmark = pytest.mark.usefixtures("mock_azure_server", "litellm_proxy_server")


def test_mock_server_health(mock_azure_server):
    """Verify mock Azure server is running and healthy."""
    response = httpx.get(f"{mock_azure_server}/health", timeout=5.0)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print(f"✓ Mock Azure server is healthy at {mock_azure_server}")


def test_litellm_proxy_health(litellm_proxy_server):
    """Verify LiteLLM proxy is running and healthy."""
    response = httpx.get(f"{litellm_proxy_server}/health", timeout=5.0)
    assert response.status_code == 200
    print(f"✓ LiteLLM proxy is healthy at {litellm_proxy_server}")


def test_litellm_proxy_model_list(litellm_proxy_server):
    """Verify LiteLLM proxy can list models."""
    response = httpx.get(
        f"{litellm_proxy_server}/v1/models",
        headers={"Authorization": "Bearer sk-1234"},
        timeout=5.0,
    )
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    models = [m["id"] for m in data["data"]]
    print(f"✓ LiteLLM proxy has {len(models)} models configured")
    assert "azure-fake-gpt-5-batch-2025-08-07" in models
    print(f"✓ Azure batch model is configured")
