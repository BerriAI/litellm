import json
import os
import sys
import traceback
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    BaseOpenAIPassThroughHandler,
    RouteChecks,
    create_pass_through_route,
    vertex_discovery_proxy_route,
    vertex_proxy_route,
)
from litellm.types.passthrough_endpoints.vertex_ai import VertexPassThroughCredentials


class TestBaseOpenAIPassThroughHandler:
    def test_join_url_paths(self):
        print("\nTesting _join_url_paths method...")

        # Test joining base URL with no path and a path
        base_url = httpx.URL("https://api.example.com")
        path = "/v1/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with no path: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test joining base URL with path and another path
        base_url = httpx.URL("https://api.example.com/v1")
        path = "/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with path: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test with path not starting with slash
        base_url = httpx.URL("https://api.example.com/v1")
        path = "chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Path without leading slash: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test with base URL having trailing slash
        base_url = httpx.URL("https://api.example.com/v1/")
        path = "/chat/completions"
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with trailing slash: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

    def test_append_openai_beta_header(self):
        print("\nTesting _append_openai_beta_header method...")

        # Create mock requests with different paths
        assistants_request = MagicMock(spec=Request)
        assistants_request.url = MagicMock()
        assistants_request.url.path = "/v1/threads/thread_123456/messages"

        non_assistants_request = MagicMock(spec=Request)
        non_assistants_request.url = MagicMock()
        non_assistants_request.url.path = "/v1/chat/completions"

        headers = {"authorization": "Bearer test_key"}

        # Test with assistants API request
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, assistants_request
        )
        print(f"Assistants API request: Added header: {result}")
        assert result["OpenAI-Beta"] == "assistants=v2"

        # Test with non-assistants API request
        headers = {"authorization": "Bearer test_key"}
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, non_assistants_request
        )
        print(f"Non-assistants API request: Headers: {result}")
        assert "OpenAI-Beta" not in result

        # Test with assistant in the path
        assistant_request = MagicMock(spec=Request)
        assistant_request.url = MagicMock()
        assistant_request.url.path = "/v1/assistants/asst_123456"

        headers = {"authorization": "Bearer test_key"}
        result = BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers, assistant_request
        )
        print(f"Assistant API request: Added header: {result}")
        assert result["OpenAI-Beta"] == "assistants=v2"

    def test_assemble_headers(self):
        print("\nTesting _assemble_headers method...")

        # Mock request
        mock_request = MagicMock(spec=Request)
        api_key = "test_api_key"

        # Patch the _append_openai_beta_header method to avoid testing it again
        with patch.object(
            BaseOpenAIPassThroughHandler,
            "_append_openai_beta_header",
            return_value={
                "authorization": "Bearer test_api_key",
                "api-key": "test_api_key",
                "test-header": "value",
            },
        ):
            result = BaseOpenAIPassThroughHandler._assemble_headers(
                api_key, mock_request
            )
            print(f"Assembled headers: {result}")
            assert result["authorization"] == "Bearer test_api_key"
            assert result["api-key"] == "test_api_key"
            assert result["test-header"] == "value"

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
    )
    async def test_base_openai_pass_through_handler(self, mock_create_pass_through):
        print("\nTesting _base_openai_pass_through_handler method...")

        # Mock dependencies
        mock_request = MagicMock(spec=Request)
        mock_request.query_params = {"model": "gpt-4"}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        # Mock the endpoint function returned by create_pass_through_route
        mock_endpoint_func = MagicMock()
        mock_endpoint_func.return_value = {"result": "success"}
        mock_create_pass_through.return_value = mock_endpoint_func

        print("Testing standard endpoint pass-through...")
        # Test with standard endpoint
        result = await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
            endpoint="/chat/completions",
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
            base_target_url="https://api.openai.com",
            api_key="test_api_key",
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        )

        # Verify the result
        print(f"Result from handler: {result}")
        assert result == {"result": "success"}

        # Verify create_pass_through_route was called with correct parameters
        call_args = mock_create_pass_through.call_args[1]
        print(
            f"create_pass_through_route called with endpoint: {call_args['endpoint']}"
        )
        print(f"create_pass_through_route called with target: {call_args['target']}")
        assert call_args["endpoint"] == "/chat/completions"
        assert call_args["target"] == "https://api.openai.com/v1/chat/completions"

        # Verify endpoint_func was called with correct parameters
        print("Verifying endpoint_func call parameters...")
        call_kwargs = mock_endpoint_func.call_args[1]
        print(f"stream parameter: {call_kwargs['stream']}")
        print(f"query_params: {call_kwargs['query_params']}")
        assert call_kwargs["stream"] is False
        assert call_kwargs["query_params"] == {"model": "gpt-4"}


class TestVertexAIPassThroughHandler:
    """
    Case 1: User set passthrough credentials - confirm credentials used.

    Case 2: User set default credentials, no exact passthrough credentials - confirm default credentials used.

    Case 3: No default credentials, no mapped credentials - request passed through directly.
    """

    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_credentials(self, monkeypatch):
        """
        Test that when passthrough credentials are set, they are correctly used in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        vertex_project = "test-project"
        vertex_location = "us-central1"
        vertex_credentials = "test-creds"

        pass_through_router = PassthroughEndpointRouter()

        pass_through_router.add_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        endpoint = f"/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/gemini-1.5-flash:generateContent"

        # Mock request
        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.headers = {
            "Authorization": "Bearer test-creds",
            "Content-Type": "application/json",
        }
        mock_request.url = Mock()
        mock_request.url.path = endpoint

        # Mock response
        mock_response = Response()

        # Mock vertex credentials
        test_project = vertex_project
        test_location = vertex_location
        test_token = vertex_credentials

        with mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async"
        ) as mock_ensure_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._get_token_and_url"
        ) as mock_get_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
        ) as mock_get_virtual_key, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_user_auth:
            # Setup mocks
            mock_ensure_token.return_value = ("test-auth-header", test_project)
            mock_get_token.return_value = (test_token, "")
            mock_get_virtual_key.return_value = "Bearer test-key"
            mock_user_auth.return_value = {"api_key": "test-key"}
            
            # Mock create_pass_through_route to return a function that returns a mock response
            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            # Call the route
            try:
                result = await vertex_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict={"api_key": "test-key"},
                )
            except Exception as e:
                print(f"Error: {e}")

            # Verify create_pass_through_route was called with correct arguments
            mock_create_route.assert_called_once_with(
                endpoint=endpoint,
                target=f"https://{test_location}-aiplatform.googleapis.com/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent",
                custom_headers={"Authorization": f"Bearer {test_token}"},
            )

    @pytest.mark.parametrize(
        "initial_endpoint",
        [
            "publishers/google/models/gemini-1.5-flash:generateContent",
            "v1/projects/bad-project/locations/bad-location/publishers/google/models/gemini-1.5-flash:generateContent",
        ],
    )
    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_default_credentials(
        self, monkeypatch, initial_endpoint
    ):
        """
        Test that when no passthrough credentials are set, default credentials are used in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        # Setup default credentials
        default_project = "default-project"
        default_location = "us-central1"
        default_credentials = "default-creds"

        pass_through_router = PassthroughEndpointRouter()
        pass_through_router.default_vertex_config = VertexPassThroughCredentials(
            vertex_project=default_project,
            vertex_location=default_location,
            vertex_credentials=default_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        # Use different project/location in request than the default
        endpoint = initial_endpoint

        mock_request = Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": f"/vertex_ai/{endpoint}",
                "headers": {},
            }
        )
        mock_response = Response()

        with mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async"
        ) as mock_ensure_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._get_token_and_url"
        ) as mock_get_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_ensure_token.return_value = ("test-auth-header", default_project)
            mock_get_token.return_value = (default_credentials, "")

            try:
                await vertex_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error: {e}")

            # Verify default credentials were used
            mock_create_route.assert_called_once_with(
                endpoint=endpoint,
                target=f"https://{default_location}-aiplatform.googleapis.com/v1/projects/{default_project}/locations/{default_location}/publishers/google/models/gemini-1.5-flash:generateContent",
                custom_headers={"Authorization": f"Bearer {default_credentials}"},
            )

    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_no_default_credentials(self, monkeypatch):
        """
        Test that when no default credentials are set, the request fails
        """
        """
        Test that when passthrough credentials are set, they are correctly used in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        vertex_project = "my-project"
        vertex_location = "us-central1"
        vertex_credentials = "test-creds"

        test_project = "test-project"
        test_location = "test-location"
        test_token = "test-creds"

        pass_through_router = PassthroughEndpointRouter()

        pass_through_router.add_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        endpoint = f"/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent"

        # Mock request
        mock_request = Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": endpoint,
                "headers": [
                    (b"authorization", b"Bearer test-creds"),
                ],
            }
        )

        # Mock response
        mock_response = Response()

        with mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async"
        ) as mock_ensure_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._get_token_and_url"
        ) as mock_get_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            mock_ensure_token.return_value = ("test-auth-header", test_project)
            mock_get_token.return_value = (test_token, "")

            # Call the route
            try:
                await vertex_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error: {e}")

            # Verify create_pass_through_route was called with correct arguments
            mock_create_route.assert_called_once_with(
                endpoint=endpoint,
                target=f"https://{test_location}-aiplatform.googleapis.com/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent",
                custom_headers={"authorization": f"Bearer {test_token}"},
            )

    @pytest.mark.asyncio
    async def test_async_vertex_proxy_route_api_key_auth(self):
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
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_auth:
            mock_auth.return_value = {"api_key": "test-key-123"}

            with patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
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


class TestVertexAIDiscoveryPassThroughHandler:
    """
    Test cases for Vertex AI Discovery passthrough endpoint
    """

    @pytest.mark.asyncio
    async def test_vertex_discovery_passthrough_with_credentials(self, monkeypatch):
        """
        Test that when passthrough credentials are set, they are correctly used in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        vertex_project = "test-project"
        vertex_location = "us-central1"
        vertex_credentials = "test-creds"

        pass_through_router = PassthroughEndpointRouter()

        pass_through_router.add_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

        monkeypatch.setattr(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router",
            pass_through_router,
        )

        endpoint = f"/v1/projects/{vertex_project}/locations/{vertex_location}/dataStores/default/servingConfigs/default:search"

        # Mock request
        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.headers = {
            "Authorization": "Bearer test-creds",
            "Content-Type": "application/json",
        }
        mock_request.url = Mock()
        mock_request.url.path = endpoint

        # Mock response
        mock_response = Response()

        # Mock vertex credentials
        test_project = vertex_project
        test_location = vertex_location
        test_token = vertex_credentials

        with mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async"
        ) as mock_ensure_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._get_token_and_url"
        ) as mock_get_token, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
        ) as mock_get_virtual_key, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_user_auth:
            # Setup mocks
            mock_ensure_token.return_value = ("test-auth-header", test_project)
            mock_get_token.return_value = (test_token, "")
            mock_get_virtual_key.return_value = "Bearer test-key"
            mock_user_auth.return_value = {"api_key": "test-key"}
            
            # Mock create_pass_through_route to return a function that returns a mock response
            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            # Call the route
            try:
                result = await vertex_discovery_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                )
            except Exception as e:
                print(f"Error: {e}")

            # Verify create_pass_through_route was called with correct arguments
            mock_create_route.assert_called_once_with(
                endpoint=endpoint,
                target=f"https://discoveryengine.googleapis.com/v1/projects/{test_project}/locations/{test_location}/dataStores/default/servingConfigs/default:search",
                custom_headers={"Authorization": f"Bearer {test_token}"},
            )

    @pytest.mark.asyncio
    async def test_vertex_discovery_proxy_route_api_key_auth(self):
        """
        Test that the route correctly handles API key authentication
        """
        # Mock dependencies
        mock_request = Mock()
        mock_request.headers = {"x-litellm-api-key": "test-key-123"}
        mock_request.method = "POST"
        mock_response = Mock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_auth:
            mock_auth.return_value = {"api_key": "test-key-123"}

            with patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_pass_through:
                mock_pass_through.return_value = AsyncMock(
                    return_value={"status": "success"}
                )

                # Call the function
                result = await vertex_discovery_proxy_route(
                    endpoint="v1/projects/test-project/locations/us-central1/dataStores/default/servingConfigs/default:search",
                    request=mock_request,
                    fastapi_response=mock_response,
                )

                # Verify user_api_key_auth was called with the correct Bearer token
                mock_auth.assert_called_once()
                call_args = mock_auth.call_args[1]
                assert call_args["api_key"] == "Bearer test-key-123"


@pytest.mark.asyncio
async def test_is_streaming_request_fn():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        is_streaming_request_fn,
    )

    mock_request = Mock()
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.form = AsyncMock(return_value={"stream": "true"})
    assert await is_streaming_request_fn(mock_request) is True
