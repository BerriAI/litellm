import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


import httpx
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
    get_litellm_virtual_key,
    vertex_proxy_route,
    _get_vertex_env_vars,
    set_default_vertex_config,
    VertexPassThroughCredentials,
    default_vertex_config,
)
from litellm.proxy.vertex_ai_endpoints.vertex_passthrough_router import (
    VertexPassThroughRouter,
)


@pytest.mark.asyncio
async def test_get_litellm_virtual_key():
    """
    Test that the get_litellm_virtual_key function correctly handles the API key authentication
    """
    # Test with x-litellm-api-key
    mock_request = Mock()
    mock_request.headers = {"x-litellm-api-key": "test-key-123"}
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer test-key-123"

    # Test with Authorization header
    mock_request.headers = {"Authorization": "Bearer auth-key-456"}
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer auth-key-456"

    # Test with both headers (x-litellm-api-key should take precedence)
    mock_request.headers = {
        "x-litellm-api-key": "test-key-123",
        "Authorization": "Bearer auth-key-456",
    }
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer test-key-123"


@pytest.mark.asyncio
async def test_vertex_proxy_route_api_key_auth():
    """
    Critical

    This is how Vertex AI JS SDK will Auth to Litellm Proxy
    """
    # Mock dependencies
    mock_request = Mock()
    mock_request.headers = {"x-litellm-api-key": "test-key-123"}
    mock_request.method = "POST"
    mock_response = Mock()

    with patch(
        "litellm.proxy.vertex_ai_endpoints.vertex_endpoints.user_api_key_auth"
    ) as mock_auth:
        mock_auth.return_value = {"api_key": "test-key-123"}

        with patch(
            "litellm.proxy.vertex_ai_endpoints.vertex_endpoints.create_pass_through_route"
        ) as mock_pass_through:
            mock_pass_through.return_value = AsyncMock(
                return_value={"status": "success"}
            )

            # Call the function
            result = await vertex_proxy_route(
                endpoint="v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
                request=mock_request,
                fastapi_response=mock_response,
            )

            # Verify user_api_key_auth was called with the correct Bearer token
            mock_auth.assert_called_once()
            call_args = mock_auth.call_args[1]
            assert call_args["api_key"] == "Bearer test-key-123"


@pytest.mark.asyncio
async def test_get_vertex_env_vars():
    """Test that _get_vertex_env_vars correctly reads environment variables"""
    # Set environment variables for the test
    os.environ["DEFAULT_VERTEXAI_PROJECT"] = "test-project-123"
    os.environ["DEFAULT_VERTEXAI_LOCATION"] = "us-central1"
    os.environ["DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/creds"

    try:
        result = _get_vertex_env_vars()
        print(result)

        # Verify the result
        assert isinstance(result, VertexPassThroughCredentials)
        assert result.vertex_project == "test-project-123"
        assert result.vertex_location == "us-central1"
        assert result.vertex_credentials == "/path/to/creds"

    finally:
        # Clean up environment variables
        del os.environ["DEFAULT_VERTEXAI_PROJECT"]
        del os.environ["DEFAULT_VERTEXAI_LOCATION"]
        del os.environ["DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"]


@pytest.mark.asyncio
async def test_set_default_vertex_config():
    """Test set_default_vertex_config with various inputs"""
    # Test with None config - set environment variables first
    os.environ["DEFAULT_VERTEXAI_PROJECT"] = "env-project"
    os.environ["DEFAULT_VERTEXAI_LOCATION"] = "env-location"
    os.environ["DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"] = "env-creds"
    os.environ["GOOGLE_CREDS"] = "secret-creds"

    try:
        # Test with None config
        set_default_vertex_config()
        from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
            default_vertex_config,
        )

        assert default_vertex_config.vertex_project == "env-project"
        assert default_vertex_config.vertex_location == "env-location"
        assert default_vertex_config.vertex_credentials == "env-creds"

        # Test with valid config.yaml settings on vertex_config
        test_config = {
            "vertex_project": "my-project-123",
            "vertex_location": "us-central1",
            "vertex_credentials": "path/to/creds",
        }
        set_default_vertex_config(test_config)
        from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
            default_vertex_config,
        )

        assert default_vertex_config.vertex_project == "my-project-123"
        assert default_vertex_config.vertex_location == "us-central1"
        assert default_vertex_config.vertex_credentials == "path/to/creds"

        # Test with environment variable reference
        test_config = {
            "vertex_project": "my-project-123",
            "vertex_location": "us-central1",
            "vertex_credentials": "os.environ/GOOGLE_CREDS",
        }
        set_default_vertex_config(test_config)
        from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
            default_vertex_config,
        )

        assert default_vertex_config.vertex_credentials == "secret-creds"

    finally:
        # Clean up environment variables
        del os.environ["DEFAULT_VERTEXAI_PROJECT"]
        del os.environ["DEFAULT_VERTEXAI_LOCATION"]
        del os.environ["DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"]
        del os.environ["GOOGLE_CREDS"]


@pytest.mark.asyncio
async def test_vertex_passthrough_router_init():
    """Test VertexPassThroughRouter initialization"""
    router = VertexPassThroughRouter()
    assert isinstance(router.deployment_key_to_vertex_credentials, dict)
    assert len(router.deployment_key_to_vertex_credentials) == 0


@pytest.mark.asyncio
async def test_get_vertex_credentials_none():
    """Test get_vertex_credentials with various inputs"""
    from litellm.proxy.vertex_ai_endpoints import vertex_endpoints

    setattr(vertex_endpoints, "default_vertex_config", VertexPassThroughCredentials())
    router = VertexPassThroughRouter()

    # Test with None project_id and location - should return default config
    creds = router.get_vertex_credentials(None, None)
    assert isinstance(creds, VertexPassThroughCredentials)

    # Test with valid project_id and location but no stored credentials
    creds = router.get_vertex_credentials("test-project", "us-central1")
    assert isinstance(creds, VertexPassThroughCredentials)
    assert creds.vertex_project is None
    assert creds.vertex_location is None
    assert creds.vertex_credentials is None


@pytest.mark.asyncio
async def test_get_vertex_credentials_stored():
    """Test get_vertex_credentials with stored credentials"""
    router = VertexPassThroughRouter()
    router.add_vertex_credentials(
        project_id="test-project",
        location="us-central1",
        vertex_credentials="test-creds",
    )

    creds = router.get_vertex_credentials(
        project_id="test-project", location="us-central1"
    )
    assert creds.vertex_project == "test-project"
    assert creds.vertex_location == "us-central1"
    assert creds.vertex_credentials == "test-creds"


@pytest.mark.asyncio
async def test_add_vertex_credentials():
    """Test add_vertex_credentials functionality"""
    router = VertexPassThroughRouter()

    # Test adding valid credentials
    router.add_vertex_credentials(
        project_id="test-project",
        location="us-central1",
        vertex_credentials="test-creds",
    )

    assert "test-project-us-central1" in router.deployment_key_to_vertex_credentials
    creds = router.deployment_key_to_vertex_credentials["test-project-us-central1"]
    assert creds.vertex_project == "test-project"
    assert creds.vertex_location == "us-central1"
    assert creds.vertex_credentials == "test-creds"

    # Test adding with None values
    router.add_vertex_credentials(
        project_id=None, location=None, vertex_credentials="test-creds"
    )
    # Should not add None values
    assert len(router.deployment_key_to_vertex_credentials) == 1


@pytest.mark.asyncio
async def test_get_deployment_key():
    """Test _get_deployment_key with various inputs"""
    router = VertexPassThroughRouter()

    # Test with valid inputs
    key = router._get_deployment_key("test-project", "us-central1")
    assert key == "test-project-us-central1"

    # Test with None values
    key = router._get_deployment_key(None, "us-central1")
    assert key is None

    key = router._get_deployment_key("test-project", None)
    assert key is None

    key = router._get_deployment_key(None, None)
    assert key is None


@pytest.mark.asyncio
async def test_get_vertex_project_id_from_url():
    """Test _get_vertex_project_id_from_url with various URLs"""
    # Test with valid URL
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent"
    project_id = VertexPassThroughRouter._get_vertex_project_id_from_url(url)
    assert project_id == "test-project"

    # Test with invalid URL
    url = "https://invalid-url.com"
    project_id = VertexPassThroughRouter._get_vertex_project_id_from_url(url)
    assert project_id is None


@pytest.mark.asyncio
async def test_get_vertex_location_from_url():
    """Test _get_vertex_location_from_url with various URLs"""
    # Test with valid URL
    url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent"
    location = VertexPassThroughRouter._get_vertex_location_from_url(url)
    assert location == "us-central1"

    # Test with invalid URL
    url = "https://invalid-url.com"
    location = VertexPassThroughRouter._get_vertex_location_from_url(url)
    assert location is None
