"""
Tests for Databricks model hub integration.

Tests the ability to fetch Databricks serving endpoints from the Databricks API
and display them in the model hub instead of using the LiteLLM cost map.
"""

import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_databricks_endpoints_response():
    """Mock response from Databricks serving endpoints API."""
    return {
        "endpoints": [
            {
                "name": "databricks-dbrx-instruct",
                "creator": "test@databricks.com",
                "creation_timestamp": 1234567890,
                "last_updated_timestamp": 1234567890,
                "state": {
                    "ready": "READY",
                    "config_update": "NOT_UPDATING",
                },
                "config": {
                    "served_models": [
                        {
                            "name": "databricks-dbrx-instruct",
                            "model_name": "databricks-dbrx-instruct",
                            "model_version": "1",
                            "workload_size": "Small",
                            "scale_to_zero_enabled": True,
                        }
                    ],
                    "served_entities": [],
                    "traffic_config": {
                        "routes": [
                            {
                                "served_model_name": "databricks-dbrx-instruct",
                                "traffic_percentage": 100,
                            }
                        ]
                    },
                },
                "tags": [],
                "id": "test-endpoint-id",
            },
            {
                "name": "llama-3-70b-instruct",
                "creator": "test@databricks.com",
                "creation_timestamp": 1234567891,
                "last_updated_timestamp": 1234567891,
                "state": {
                    "ready": "READY",
                    "config_update": "NOT_UPDATING",
                },
                "config": {
                    "served_models": [
                        {
                            "name": "llama-3-70b-instruct",
                            "model_name": "llama-3-70b-instruct",
                            "model_version": "1",
                            "workload_size": "Medium",
                            "scale_to_zero_enabled": False,
                        }
                    ],
                    "served_entities": [],
                    "traffic_config": {
                        "routes": [
                            {
                                "served_model_name": "llama-3-70b-instruct",
                                "traffic_percentage": 100,
                            }
                        ]
                    },
                },
                "tags": [],
                "id": "test-endpoint-id-2",
            },
        ]
    }


@pytest.mark.asyncio
async def test_get_databricks_serving_endpoints_with_api_key(
    monkeypatch, mock_databricks_endpoints_response
):
    """Test fetching Databricks endpoints with API key authentication."""
    from litellm.proxy.public_endpoints.public_endpoints import (
        get_databricks_serving_endpoints,
    )

    # Set up environment
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapi123456"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    # Mock httpx client
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_databricks_endpoints_response

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await get_databricks_serving_endpoints()

        assert "endpoints" in result
        assert "workspace_url" in result
        assert len(result["endpoints"]) == 2
        assert result["endpoints"][0]["name"] == "databricks-dbrx-instruct"
        assert result["endpoints"][1]["name"] == "llama-3-70b-instruct"


@pytest.mark.asyncio
async def test_get_databricks_serving_endpoints_with_oauth(
    monkeypatch, mock_databricks_endpoints_response
):
    """Test fetching Databricks endpoints with OAuth M2M authentication."""
    from litellm.proxy.public_endpoints.public_endpoints import (
        get_databricks_serving_endpoints,
    )

    # Set up environment
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    client_id = "test-client-id"
    client_secret = "test-client-secret"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", client_id)
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", client_secret)

    # Mock OAuth token response
    mock_token_response = Mock()
    mock_token_response.status_code = 200
    mock_token_response.json.return_value = {"access_token": "oauth-token-123"}

    # Mock endpoints response
    mock_endpoints_response = Mock()
    mock_endpoints_response.status_code = 200
    mock_endpoints_response.json.return_value = mock_databricks_endpoints_response

    with patch("requests.post", return_value=mock_token_response):
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_endpoints_response
            )

            result = await get_databricks_serving_endpoints()

            assert "endpoints" in result
            assert len(result["endpoints"]) == 2


@pytest.mark.asyncio
async def test_get_databricks_serving_endpoints_missing_credentials():
    """Test that appropriate error is raised when credentials are missing."""
    from fastapi import HTTPException

    from litellm.proxy.public_endpoints.public_endpoints import (
        get_databricks_serving_endpoints,
    )

    # Clear environment variables
    for var in [
        "DATABRICKS_API_KEY",
        "DATABRICKS_API_BASE",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
    ]:
        if var in os.environ:
            del os.environ[var]

    with pytest.raises(HTTPException) as exc_info:
        await get_databricks_serving_endpoints()

    assert exc_info.value.status_code == 400
    assert "Missing Databricks credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_public_model_hub_with_databricks_provider(
    monkeypatch, mock_databricks_endpoints_response
):
    """Test that model hub returns Databricks endpoints when provider=databricks."""
    from litellm.proxy.public_endpoints.public_endpoints import public_model_hub

    # Set up environment
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapi123456"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    # Mock httpx client for Databricks API call
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_databricks_endpoints_response

    # Mock llm_router
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
        mock_router.__bool__.return_value = True  # Make sure it's not None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await public_model_hub(provider="databricks")

            assert len(result) == 2
            assert result[0].model_group == "databricks/databricks-dbrx-instruct"
            assert result[0].providers == ["databricks"]
            assert result[0].mode == "chat"
            assert result[1].model_group == "databricks/llama-3-70b-instruct"


@pytest.mark.asyncio
async def test_databricks_endpoints_formatted_correctly(
    monkeypatch, mock_databricks_endpoints_response
):
    """Test that Databricks endpoint response is formatted correctly for model hub."""
    from litellm.proxy.public_endpoints.public_endpoints import (
        get_databricks_serving_endpoints,
    )

    # Set up environment
    base_url = "https://my.workspace.cloud.databricks.com"
    api_key = "dapi123456"
    monkeypatch.setenv("DATABRICKS_API_BASE", f"{base_url}/serving-endpoints")
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    # Mock httpx client
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_databricks_endpoints_response

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await get_databricks_serving_endpoints()

        # Verify formatting
        endpoint = result["endpoints"][0]
        assert endpoint["name"] == "databricks-dbrx-instruct"
        assert endpoint["creator"] == "test@databricks.com"
        assert endpoint["state"] == "READY"
        assert "config" in endpoint
        assert "served_models" in endpoint["config"]
        assert (
            endpoint["endpoint_url"]
            == f"{base_url}/serving-endpoints/databricks-dbrx-instruct"
        )


@pytest.mark.asyncio
async def test_databricks_api_error_handling(monkeypatch):
    """Test that API errors from Databricks are handled properly."""
    from fastapi import HTTPException

    from litellm.proxy.public_endpoints.public_endpoints import (
        get_databricks_serving_endpoints,
    )

    # Set up environment
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapi123456"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    # Mock httpx client with error response
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_databricks_serving_endpoints()

        assert exc_info.value.status_code == 401
        assert "Failed to fetch Databricks endpoints" in exc_info.value.detail
