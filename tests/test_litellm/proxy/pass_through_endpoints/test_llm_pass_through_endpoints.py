import base64
import contextlib
import json
import os
import traceback
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, Response
from fastapi.testclient import TestClient
from starlette.datastructures import FormData


import litellm
from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    BaseOpenAIPassThroughHandler,
    RouteChecks,
    _join_url_paths,
    azure_proxy_route,
    bedrock_llm_proxy_route,
    bedrock_proxy_route,
    create_pass_through_route,
    cursor_proxy_route,
    get_azure_ai_search_index_from_endpoint,
    get_vertex_base_url,
    is_azure_ai_search_service_level_index_create,
    llm_passthrough_factory_proxy_route,
    milvus_proxy_route,
    mistral_proxy_route,
    openai_proxy_route,
    vertex_discovery_proxy_route,
    vertex_proxy_route,
    vllm_proxy_route,
)
from litellm.proxy._types import LitellmUserRoles, SpecialHeaders, UserAPIKeyAuth
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.types.passthrough_endpoints.vertex_ai import VertexPassThroughCredentials


class TestVertexPassthroughGetVertexBaseUrl:
    """Module-local get_vertex_base_url (trailing slash); rules match common_utils."""

    @pytest.mark.parametrize(
        "vertex_location, expected",
        [
            ("global", "https://aiplatform.googleapis.com/"),
            ("us-central1", "https://us-central1-aiplatform.googleapis.com/"),
            ("us", "https://aiplatform.us.rep.googleapis.com/"),
            ("eu", "https://aiplatform.eu.rep.googleapis.com/"),
        ],
    )
    def test_returns_base_with_trailing_slash(self, vertex_location, expected):
        assert get_vertex_base_url(vertex_location) == expected

    @pytest.mark.parametrize(
        "vertex_location, expected_host",
        [
            ("global", "aiplatform.googleapis.com"),
            ("us-central1", "us-central1-aiplatform.googleapis.com"),
            ("us", "aiplatform.us.rep.googleapis.com"),
            ("eu", "aiplatform.eu.rep.googleapis.com"),
        ],
    )
    def test_websocket_host_strips_scheme(self, vertex_location, expected_host):
        host = get_vertex_base_url(vertex_location).removeprefix("https://").rstrip("/")
        assert host == expected_host


class TestBaseOpenAIPassThroughHandler:
    def test_join_url_paths(self):
        print("\nTesting _join_url_paths method...")

        # Test joining base URL with no path and a path
        base_url = httpx.URL("https://api.example.com")
        path = "/v1/chat/completions"
        result = _join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with no path: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test joining base URL with path and another path
        base_url = httpx.URL("https://api.example.com/v1")
        path = "/chat/completions"
        result = _join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Base URL with path: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test with path not starting with slash
        base_url = httpx.URL("https://api.example.com/v1")
        path = "chat/completions"
        result = _join_url_paths(
            base_url, path, litellm.LlmProviders.OPENAI.value
        )
        print(f"Path without leading slash: '{base_url}' + '{path}' → '{result}'")
        assert str(result) == "https://api.example.com/v1/chat/completions"

        # Test with base URL having trailing slash
        base_url = httpx.URL("https://api.example.com/v1/")
        path = "/chat/completions"
        result = _join_url_paths(
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
        mock_endpoint_func = AsyncMock(return_value={"result": "success"})
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
        mock_endpoint_func.assert_awaited_once()
        assert mock_endpoint_func.await_args is not None
        # The endpoint_func is called with request, fastapi_response, user_api_key_dict
        # No longer checking for stream and query_params as they're handled differently


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
        mock_request.state = (
            None  # Prevent Mock from returning a truthy _cached_headers
        )
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

        with (
            mock.patch(
                "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
            ) as mock_load_auth,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
            ) as mock_get_virtual_key,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
            ) as mock_user_auth,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
            ) as mock_get_handler,
        ):
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = test_token

            # Setup mocks
            mock_load_auth.return_value = (mock_credentials, test_project)
            mock_get_virtual_key.return_value = "Bearer test-key"
            mock_user_auth.return_value = {"api_key": "test-key"}

            # Mock the vertex handler
            mock_handler = Mock()
            mock_handler.get_default_base_target_url.return_value = (
                f"https://{test_location}-aiplatform.googleapis.com/"
            )
            mock_handler.update_base_target_url_with_credential_location = Mock(
                return_value=f"https://{test_location}-aiplatform.googleapis.com/"
            )
            mock_get_handler.return_value = mock_handler

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
                is_streaming_request=False,
            )

    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_global_location(self, monkeypatch):
        """
        Test that when global location is used, it is correctly handled in the request
        """
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        vertex_project = "test-project"
        vertex_location = "global"
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
        mock_request.state = (
            None  # Prevent Mock from returning a truthy _cached_headers
        )
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

        with (
            mock.patch(
                "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
            ) as mock_load_auth,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
            ) as mock_get_virtual_key,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
            ) as mock_user_auth,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
            ) as mock_get_handler,
        ):
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = test_token

            # Setup mocks
            mock_load_auth.return_value = (mock_credentials, test_project)
            mock_get_virtual_key.return_value = "Bearer test-key"
            mock_user_auth.return_value = {"api_key": "test-key"}

            # Mock the vertex handler for global location
            mock_handler = Mock()
            mock_handler.get_default_base_target_url.return_value = (
                "https://aiplatform.googleapis.com/"
            )
            mock_handler.update_base_target_url_with_credential_location = Mock(
                return_value="https://aiplatform.googleapis.com/"
            )
            mock_get_handler.return_value = mock_handler

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
                target=f"https://aiplatform.googleapis.com/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent",
                custom_headers={"Authorization": f"Bearer {test_token}"},
                is_streaming_request=False,
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

        with (
            mock.patch(
                "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
            ) as mock_load_auth,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
            ) as mock_get_handler,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
        ):
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = default_credentials

            mock_load_auth.return_value = (mock_credentials, default_project)
            mock_auth.return_value = MagicMock()

            # Mock the vertex handler
            mock_handler = Mock()
            mock_handler.get_default_base_target_url.return_value = (
                f"https://{default_location}-aiplatform.googleapis.com/"
            )
            mock_handler.update_base_target_url_with_credential_location = Mock(
                return_value=f"https://{default_location}-aiplatform.googleapis.com/"
            )
            mock_get_handler.return_value = mock_handler

            # Mock create_pass_through_route to return a function that returns a mock response
            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

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
                is_streaming_request=False,
            )

    @pytest.mark.asyncio
    async def test_vertex_passthrough_with_no_default_credentials(self, monkeypatch):
        """
        With no Vertex credential matching the request, the only Authorization present
        is the caller's own virtual key. It must not be forwarded to Google; the
        request fails with a clean 401 instead (LIT-5997).
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
                    (b"authorization", b"Bearer sk-test-creds"),
                ],
            }
        )

        # Mock response
        mock_response = Response()

        with (
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._ensure_access_token_async"
            ) as mock_ensure_token,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.vertex_llm_base._get_token_and_url"
            ) as mock_get_token,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
                new_callable=AsyncMock,
            ) as mock_auth,
        ):
            mock_ensure_token.return_value = ("test-auth-header", test_project)
            mock_get_token.return_value = (test_token, "")
            mock_auth.return_value = UserAPIKeyAuth(api_key="sk-test-creds")

            with pytest.raises(HTTPException) as exc_info:
                await vertex_proxy_route(
                    endpoint=endpoint,
                    request=mock_request,
                    fastapi_response=mock_response,
                )

            assert exc_info.value.status_code == 401
            mock_create_route.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_vertex_proxy_route_api_key_auth(self):
        """
        Critical

        This is how Vertex AI JS SDK will Auth to Litellm Proxy: the virtual key
        arrives in x-litellm-api-key and must reach user_api_key_auth. With no Vertex
        credential configured, that virtual key must not be forwarded to Google, so
        the request fails with a clean 401 (LIT-5997).
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

                with pytest.raises(HTTPException) as exc_info:
                    await vertex_proxy_route(
                        endpoint="v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
                        request=mock_request,
                        fastapi_response=mock_response,
                    )

                assert exc_info.value.status_code == 401
                mock_pass_through.assert_not_called()
                mock_auth.assert_called_once()
                call_args = mock_auth.call_args[1]
                assert call_args["api_key"] == "Bearer test-key-123"

    def test_vertex_passthrough_handler_multimodal_embedding_response(self):
        """
        Test that vertex_passthrough_handler correctly identifies and processes multimodal embedding responses
        """
        import datetime
        from unittest.mock import Mock

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        # Create mock multimodal embedding response data
        multimodal_response_data = {
            "predictions": [
                {
                    "textEmbedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "imageEmbedding": [0.6, 0.7, 0.8, 0.9, 1.0],
                },
                {
                    "videoEmbeddings": [
                        {
                            "embedding": [0.11, 0.22, 0.33, 0.44, 0.55],
                            "startOffsetSec": 0,
                            "endOffsetSec": 5,
                        }
                    ]
                },
            ]
        }

        # Create mock httpx.Response
        mock_httpx_response = Mock()
        mock_httpx_response.json.return_value = multimodal_response_data
        mock_httpx_response.status_code = 200

        # Create mock logging object
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        mock_logging_obj.model_call_details = {}

        # Test URL with multimodal embedding model
        url_route = "/v1/projects/test-project/locations/us-central1/publishers/google/models/multimodalembedding@001:predict"

        start_time = datetime.datetime.now()
        end_time = datetime.datetime.now()

        with patch(
            "litellm.llms.vertex_ai.multimodal_embeddings.transformation.VertexAIMultimodalEmbeddingConfig"
        ) as mock_multimodal_config:
            # Mock the multimodal config instance and its methods
            mock_config_instance = Mock()
            mock_multimodal_config.return_value = mock_config_instance

            # Create a mock embedding response that would be returned by the transformation
            from litellm.types.utils import Embedding, EmbeddingResponse, Usage

            mock_embedding_response = EmbeddingResponse(
                object="list",
                data=[
                    Embedding(
                        embedding=[0.1, 0.2, 0.3, 0.4, 0.5], index=0, object="embedding"
                    ),
                    Embedding(
                        embedding=[0.6, 0.7, 0.8, 0.9, 1.0], index=1, object="embedding"
                    ),
                ],
                model="multimodalembedding@001",
                usage=Usage(prompt_tokens=0, total_tokens=0, completion_tokens=0),
            )
            mock_config_instance.transform_embedding_response.return_value = (
                mock_embedding_response
            )

            # Call the handler
            result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="test-result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
            )

            # Verify multimodal embedding detection and processing
            assert result is not None
            assert "result" in result
            assert "kwargs" in result

            # Verify that the multimodal config was instantiated and used
            mock_multimodal_config.assert_called_once()
            mock_config_instance.transform_embedding_response.assert_called_once()

            # Verify the response is an EmbeddingResponse
            assert isinstance(result["result"], EmbeddingResponse)
            assert result["result"].model == "multimodalembedding@001"
            assert len(result["result"].data) == 2

    def test_vertex_passthrough_handler_multimodal_detection_method(self):
        """
        Test the _is_multimodal_embedding_response detection method specifically
        """
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        # Test case 1: Response with textEmbedding should be detected as multimodal
        response_with_text_embedding = {
            "predictions": [{"textEmbedding": [0.1, 0.2, 0.3]}]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                response_with_text_embedding
            )
            is True
        )

        # Test case 2: Response with imageEmbedding should be detected as multimodal
        response_with_image_embedding = {
            "predictions": [{"imageEmbedding": [0.4, 0.5, 0.6]}]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                response_with_image_embedding
            )
            is True
        )

        # Test case 3: Response with videoEmbeddings should be detected as multimodal
        response_with_video_embeddings = {
            "predictions": [
                {
                    "videoEmbeddings": [
                        {
                            "embedding": [0.7, 0.8, 0.9],
                            "startOffsetSec": 0,
                            "endOffsetSec": 5,
                        }
                    ]
                }
            ]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                response_with_video_embeddings
            )
            is True
        )

        # Test case 4: Regular text embedding response should NOT be detected as multimodal
        regular_embedding_response = {
            "predictions": [{"embeddings": {"values": [0.1, 0.2, 0.3]}}]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                regular_embedding_response
            )
            is False
        )

        # Test case 5: Non-embedding response should NOT be detected as multimodal
        non_embedding_response = {
            "candidates": [{"content": {"parts": [{"text": "Hello world"}]}}]
        }
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                non_embedding_response
            )
            is False
        )

        # Test case 6: Empty response should NOT be detected as multimodal
        empty_response = {}
        assert (
            VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                empty_response
            )
            is False
        )

    def test_vertex_passthrough_handler_predict_cost_tracking(self):
        """
        Test that vertex_passthrough_handler correctly tracks costs for /predict endpoint
        """
        import datetime
        from unittest.mock import Mock, patch

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        # Create mock embedding response data
        embedding_response_data = {
            "predictions": [
                {
                    "embeddings": {
                        "values": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "statistics": {"token_count": 10},
                    }
                }
            ]
        }

        # Create mock httpx.Response
        mock_httpx_response = Mock()
        mock_httpx_response.json.return_value = embedding_response_data
        mock_httpx_response.status_code = 200

        # Create mock logging object
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        mock_logging_obj.model_call_details = {}

        # Test URL with /predict endpoint
        url_route = "/v1/projects/test-project/locations/us-central1/publishers/google/models/textembedding-gecko@001:predict"

        start_time = datetime.datetime.now()
        end_time = datetime.datetime.now()

        with patch("litellm.completion_cost") as mock_completion_cost:
            # Mock the completion cost calculation
            mock_completion_cost.return_value = 0.0001

            # Call the handler
            result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="test-result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
            )

            # Verify cost tracking was implemented
            assert result is not None
            assert "result" in result
            assert "kwargs" in result

            # Verify cost calculation was called
            mock_completion_cost.assert_called_once()

            # Verify cost is set in kwargs
            assert "response_cost" in result["kwargs"]
            assert result["kwargs"]["response_cost"] == 0.0001

            # Verify cost is set in logging object
            assert "response_cost" in mock_logging_obj.model_call_details
            assert mock_logging_obj.model_call_details["response_cost"] == 0.0001

            # Verify model is set in kwargs
            assert "model" in result["kwargs"]
            assert result["kwargs"]["model"] == "textembedding-gecko@001"

    def test_vertex_passthrough_handler_embed_content_response(self):
        """
        Test that vertex_passthrough_handler correctly handles :embedContent responses
        and invokes cost/logging callbacks (regression for silent drop bug).
        """
        import datetime
        from unittest.mock import Mock, patch

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        embed_content_response_data = {
            "embedding": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        }

        mock_httpx_response = Mock()
        mock_httpx_response.json.return_value = embed_content_response_data
        mock_httpx_response.status_code = 200

        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-embed"
        mock_logging_obj.model_call_details = {}

        url_route = "/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-embedding-001:embedContent"

        start_time = datetime.datetime.now()
        end_time = datetime.datetime.now()

        with patch("litellm.completion_cost") as mock_completion_cost:
            mock_completion_cost.return_value = 0.0002

            result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="test-result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
            )

        assert result is not None
        assert (
            result["result"] is not None
        ), "result must not be None — logging callbacks need a non-null response"
        assert "kwargs" in result
        assert result["kwargs"].get("response_cost") == 0.0002
        assert result["kwargs"].get("model") == "gemini-embedding-001"
        assert result["kwargs"].get("custom_llm_provider") == "vertex_ai"
        assert mock_logging_obj.model_call_details.get("response_cost") == 0.0002
        mock_completion_cost.assert_called_once()

    def test_vertex_passthrough_handler_batch_embed_contents_response(self):
        """
        Test that vertex_passthrough_handler correctly handles :batchEmbedContents responses.
        """
        import datetime
        from unittest.mock import Mock, patch

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        batch_embed_response_data = {
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]},
                {"values": [0.4, 0.5, 0.6]},
            ]
        }

        mock_httpx_response = Mock()
        mock_httpx_response.json.return_value = batch_embed_response_data
        mock_httpx_response.status_code = 200

        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-batch"
        mock_logging_obj.model_call_details = {}

        url_route = "/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-embedding-001:batchEmbedContents"

        start_time = datetime.datetime.now()
        end_time = datetime.datetime.now()

        with patch("litellm.completion_cost") as mock_completion_cost:
            mock_completion_cost.return_value = 0.0003

            result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="test-result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
            )

        assert result is not None
        assert (
            result["result"] is not None
        ), "result must not be None for batchEmbedContents"
        assert result["kwargs"].get("response_cost") == 0.0003
        assert result["kwargs"].get("model") == "gemini-embedding-001"
        assert result["kwargs"].get("custom_llm_provider") == "vertex_ai"
        mock_completion_cost.assert_called_once()

    def test_vertex_passthrough_handler_embed_content_google_ai_studio_url(self):
        """
        Test that _handle_embed_content_response sets custom_llm_provider=gemini
        when the URL is a generativelanguage.googleapis.com (Google AI Studio) endpoint.
        """
        import datetime
        from unittest.mock import Mock, patch

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        embed_content_response_data = {
            "embedding": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        }

        mock_httpx_response = Mock()
        mock_httpx_response.json.return_value = embed_content_response_data
        mock_httpx_response.status_code = 200

        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-gemini-studio"
        mock_logging_obj.model_call_details = {}

        # Google AI Studio URL (not Vertex AI)
        url_route = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent"

        start_time = datetime.datetime.now()
        end_time = datetime.datetime.now()

        with patch("litellm.completion_cost") as mock_completion_cost:
            mock_completion_cost.return_value = 0.0001

            result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="test-result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
            )

        assert result is not None
        assert result["result"] is not None
        assert (
            result["kwargs"].get("custom_llm_provider") == "gemini"
        ), "Google AI Studio embedContent URLs must set custom_llm_provider=gemini, not vertex_ai"
        assert result["kwargs"].get("model") == "gemini-embedding-2-preview"
        mock_completion_cost.assert_called_once()

    @pytest.mark.parametrize("streaming", [False, True])
    def test_vertex_passthrough_handler_prices_regional_endpoint_with_uplift(self, monkeypatch, streaming):
        """
        Both cost computations for a passthrough call must price on the URL's serving location:
        the handler-computed cost, and the async success recompute, which re-resolves the
        location from the logging object and previously fell through empty optional_params to
        the us-central1 default, billing the regional uplift on global traffic too (#34393).
        """
        import datetime

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {
                **litellm.get_model_cost_map(url=""),
                "vertex_ai/gemini-fake-regional": {
                    "litellm_provider": "vertex_ai",
                    "mode": "chat",
                    "input_cost_per_token": 1e-06,
                    "output_cost_per_token": 2e-06,
                    "regional_endpoint_uplift_multiplier": 1.1,
                },
            },
        )

        response_body: Final = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hello"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            },
        }

        def costs_for(location: str) -> tuple[float, float]:
            url_route: Final = (
                f"https://{location}-aiplatform.googleapis.com/v1/projects/p/locations/{location}"
                "/publishers/google/models/gemini-fake-regional:"
                f"{'streamGenerateContent' if streaming else 'generateContent'}"
            )
            start_time: Final = datetime.datetime.now()
            end_time: Final = datetime.datetime.now()
            logging_obj: Final = Logging(
                model="gemini-fake-regional",
                messages=[{"role": "user", "content": "hi"}],
                stream=streaming,
                call_type="pass_through_endpoint",
                start_time=start_time,
                litellm_call_id="call-id",
                function_id="fn-id",
            )
            logging_obj.update_environment_variables(
                model="gemini-fake-regional",
                user="unknown",
                optional_params={},
                litellm_params={},
                call_type="pass_through_endpoint",
            )
            if streaming:
                result = VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
                    litellm_logging_obj=logging_obj,
                    passthrough_success_handler_obj=Mock(),
                    url_route=url_route,
                    request_body={},
                    endpoint_type="vertex_ai",
                    start_time=start_time,
                    all_chunks=[json.dumps(response_body)],
                    model=None,
                    end_time=end_time,
                )
            else:
                mock_httpx_response: Final = Mock()
                mock_httpx_response.json.return_value = response_body
                mock_httpx_response.headers = {}
                mock_httpx_response.status_code = 200
                result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                    httpx_response=mock_httpx_response,
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result="test-result",
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=False,
                )
            recomputed: Final = logging_obj._response_cost_calculator(result=result["result"])
            return result["kwargs"]["response_cost"], recomputed

        global_handler_cost, global_recomputed_cost = costs_for("global")
        regional_handler_cost, regional_recomputed_cost = costs_for("us-east5")

        plain_cost: Final = 10 * 1e-06 + 20 * 2e-06
        assert global_handler_cost == pytest.approx(plain_cost, rel=1e-9)
        assert regional_handler_cost == pytest.approx(plain_cost * 1.10, rel=1e-9), (
            "regional Vertex passthrough traffic must bill at 1.1x the global rate"
        )
        assert global_recomputed_cost == pytest.approx(plain_cost, rel=1e-9), (
            "the logging recompute must not price global passthrough traffic as regional"
        )
        assert regional_recomputed_cost == pytest.approx(plain_cost * 1.10, rel=1e-9)


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

        endpoint = f"v1/projects/{vertex_project}/locations/{vertex_location}/dataStores/default/servingConfigs/default:search"

        # Mock request
        mock_request = Mock()
        mock_request.state = (
            None  # Prevent Mock from returning a truthy _cached_headers
        )
        mock_request.method = "POST"
        mock_request.headers = {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        }
        mock_request.url = Mock()
        mock_request.url.path = endpoint

        # Mock response
        mock_response = Response()

        # Mock vertex credentials
        test_project = vertex_project
        test_location = vertex_location
        test_token = "test-auth-token"

        with (
            mock.patch(
                "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
            ) as mock_load_auth,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
            ) as mock_get_virtual_key,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
            ) as mock_user_auth,
            mock.patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
            ) as mock_get_handler,
        ):
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = test_token

            # Setup mocks
            mock_load_auth.return_value = (mock_credentials, test_project)
            mock_get_virtual_key.return_value = "Bearer test-key"
            mock_user_auth.return_value = {"api_key": "test-key"}

            # Mock the discovery handler
            mock_handler = Mock()
            mock_handler.get_default_base_target_url.return_value = (
                "https://discoveryengine.googleapis.com"
            )
            mock_handler.update_base_target_url_with_credential_location = Mock(
                return_value="https://discoveryengine.googleapis.com"
            )
            mock_get_handler.return_value = mock_handler

            # Mock create_pass_through_route to return a function that returns a mock response
            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            # Call the route
            result = await vertex_discovery_proxy_route(
                endpoint=endpoint,
                request=mock_request,
                fastapi_response=mock_response,
            )

            # Verify create_pass_through_route was called with correct arguments
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args
            assert call_args[1]["endpoint"] == endpoint
            assert test_project in call_args[1]["target"]
            assert test_location in call_args[1]["target"]
            assert "Authorization" in call_args[1]["custom_headers"]
            assert (
                call_args[1]["custom_headers"]["Authorization"]
                == f"Bearer {test_token}"
            )

    @pytest.mark.asyncio
    async def test_vertex_discovery_proxy_route_api_key_auth(self):
        """
        The virtual key arrives in x-litellm-api-key and must reach user_api_key_auth.
        With no Vertex credential configured, that virtual key must not be forwarded to
        Google, so the request fails with a clean 401 (LIT-5997).
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

                with pytest.raises(HTTPException) as exc_info:
                    await vertex_discovery_proxy_route(
                        endpoint="v1/projects/test-project/locations/us-central1/dataStores/default/servingConfigs/default:search",
                        request=mock_request,
                        fastapi_response=mock_response,
                    )

                assert exc_info.value.status_code == 401
                mock_pass_through.assert_not_called()
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
    mock_request.form = AsyncMock(return_value=FormData({"stream": "true"}))
    assert await is_streaming_request_fn(mock_request) is True


@pytest.mark.asyncio
async def test_mistral_passthrough_accepts_multipart_without_json_parsing():
    boundary = "----litellm-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        "ocr\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="document.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
        "%PDF-1.4 test\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    async def receive():
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/mistral/v1/files",
            "headers": [
                (
                    b"content-type",
                    f"multipart/form-data; boundary={boundary}".encode("utf-8"),
                )
            ],
            "query_string": b"",
        },
        receive=receive,
    )

    captured_kwargs = {}

    async def fake_endpoint(request, fastapi_response, user_api_key_dict):
        return {"ok": True}

    def fake_create_pass_through_route(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_endpoint

    user_api_key_dict = UserAPIKeyAuth(token="test-key")

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="mistral-test-key",
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
            side_effect=fake_create_pass_through_route,
        ),
    ):
        response = await mistral_proxy_route(
            endpoint="v1/files",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=user_api_key_dict,
        )

    assert response == {"ok": True}
    assert captured_kwargs["is_streaming_request"] is False
    assert captured_kwargs["custom_headers"] == {
        "Authorization": "Bearer mistral-test-key"
    }


class TestBedrockLLMProxyRoute:
    @pytest.mark.asyncio
    async def test_bedrock_llm_proxy_route_application_inference_profile(self):
        mock_request = Mock()
        mock_request.method = "POST"
        mock_response = Mock()
        mock_user_api_key_dict = Mock()
        mock_request_body = {"messages": [{"role": "user", "content": "test"}]}
        mock_processor = Mock()
        mock_processor.base_passthrough_process_llm_request = AsyncMock(
            return_value="success"
        )

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._read_request_body",
                return_value=mock_request_body,
            ),
            patch(
                "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing",
                return_value=mock_processor,
            ),
        ):

            # Test application-inference-profile endpoint
            endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/r742sbn2zckd/converse"

            result = await bedrock_llm_proxy_route(
                endpoint=endpoint,
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_processor.base_passthrough_process_llm_request.assert_called_once()
            call_kwargs = (
                mock_processor.base_passthrough_process_llm_request.call_args.kwargs
            )

            # For application-inference-profile, model should be "arn:aws:bedrock:us-east-1:026090525607:application-inference-profile/r742sbn2zckd"
            assert (
                call_kwargs["model"]
                == "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/r742sbn2zckd"
            )
            assert result == "success"

    @pytest.mark.asyncio
    async def test_bedrock_llm_proxy_route_regular_model(self):
        mock_request = Mock()
        mock_request.method = "POST"
        mock_response = Mock()
        mock_user_api_key_dict = Mock()
        mock_request_body = {"messages": [{"role": "user", "content": "test"}]}
        mock_processor = Mock()
        mock_processor.base_passthrough_process_llm_request = AsyncMock(
            return_value="success"
        )

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._read_request_body",
                return_value=mock_request_body,
            ),
            patch(
                "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing",
                return_value=mock_processor,
            ),
        ):

            # Test regular model endpoint
            endpoint = "model/anthropic.claude-3-sonnet-20240229-v1:0/converse"

            result = await bedrock_llm_proxy_route(
                endpoint=endpoint,
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )
            mock_processor.base_passthrough_process_llm_request.assert_called_once()
            call_kwargs = (
                mock_processor.base_passthrough_process_llm_request.call_args.kwargs
            )

            # For regular models, model should be just the model ID
            assert call_kwargs["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert result == "success"

    @pytest.mark.asyncio
    async def test_bedrock_error_handling_returns_actual_error(self):
        """
        Test that when Bedrock API returns an error, it is properly propagated to the user
        instead of being returned as a generic "Internal Server Error".
        """
        from fastapi import HTTPException

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            handle_bedrock_passthrough_router_model,
        )

        bedrock_error_message = '{"message":"ContentBlock object at messages.0.content.0 must set one of the following keys: text, image, toolUse, toolResult, document, video."}'

        # Create a mock httpx.Response for the error
        mock_error_response = Mock(spec=httpx.Response)
        mock_error_response.status_code = 400
        mock_error_response.aread = AsyncMock(
            return_value=bedrock_error_message.encode("utf-8")
        )

        # Create the HTTPStatusError
        mock_http_error = httpx.HTTPStatusError(
            message="Bad Request",
            request=Mock(spec=httpx.Request),
            response=mock_error_response,
        )

        # Create mocks for all required parameters
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/bedrock/model/test-model/converse"

        mock_request_body = {
            "messages": [{"role": "user", "content": [{"textaaa": "Hello"}]}]
        }

        mock_llm_router = Mock()

        # Mock ProxyBaseLLMRequestProcessing to raise the httpx error
        with patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_passthrough_process_llm_request",
            new_callable=AsyncMock,
            side_effect=mock_http_error,
        ):
            mock_user_api_key_dict = Mock()
            mock_user_api_key_dict.api_key = "test-key"
            mock_user_api_key_dict.allowed_model_region = None

            mock_proxy_logging_obj = Mock()
            mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

            endpoint = "model/test-model/converse"
            model = "test-model"

            with pytest.raises(HTTPException) as exc_info:
                await handle_bedrock_passthrough_router_model(
                    model=model,
                    endpoint=endpoint,
                    request=mock_request,
                    request_body=mock_request_body,
                    llm_router=mock_llm_router,
                    user_api_key_dict=mock_user_api_key_dict,
                    proxy_logging_obj=mock_proxy_logging_obj,
                    general_settings={},
                    proxy_config=None,
                    select_data_generator=None,
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    version=None,
                )

            assert exc_info.value.status_code == 400
            assert (
                "ContentBlock object at messages.0.content.0 must set one of the following keys"
                in str(exc_info.value.detail)
            )

    @pytest.mark.asyncio
    async def test_bedrock_passthrough_uses_model_specific_credentials(self):
        """
        Test that Bedrock passthrough endpoints use credentials from model configuration
        instead of environment variables when a router model is used.

        This test verifies the fix for the bug where passthrough endpoints were using
        environment variables instead of model-specific credentials from config.yaml.
        """
        from litellm import Router
        from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            handle_bedrock_passthrough_router_model,
        )

        # Model-specific credentials (different from env vars)
        model_access_key = "MODEL_SPECIFIC_ACCESS_KEY"
        model_secret_key = "MODEL_SPECIFIC_SECRET_KEY"
        model_region = "us-west-2"
        model_session_token = "MODEL_SESSION_TOKEN"

        # Environment variables (should NOT be used)
        env_access_key = "ENV_ACCESS_KEY"
        env_secret_key = "ENV_SECRET_KEY"
        env_region = "us-east-1"

        # Set environment variables to different values
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": env_access_key,
                "AWS_SECRET_ACCESS_KEY": env_secret_key,
                "AWS_REGION_NAME": env_region,
            },
        ):
            # Test 1: Verify get_litellm_params extracts AWS credentials from kwargs
            kwargs_with_creds = {
                "aws_access_key_id": model_access_key,
                "aws_secret_access_key": model_secret_key,
                "aws_region_name": model_region,
                "aws_session_token": model_session_token,
                "model": "bedrock/test-model",
            }
            litellm_params = get_litellm_params(**kwargs_with_creds)

            # Verify credentials are extracted
            assert litellm_params.get("aws_access_key_id") == model_access_key
            assert litellm_params.get("aws_secret_access_key") == model_secret_key
            assert litellm_params.get("aws_region_name") == model_region
            assert litellm_params.get("aws_session_token") == model_session_token

            # Test 2: Verify router passes model credentials to passthrough
            router = Router(
                model_list=[
                    {
                        "model_name": "claude-opus-4-1",
                        "litellm_params": {
                            "model": "bedrock/us.anthropic.claude-opus-4-20250514-v1:0",
                            "aws_access_key_id": model_access_key,
                            "aws_secret_access_key": model_secret_key,
                            "aws_region_name": model_region,
                            "aws_session_token": model_session_token,
                            "custom_llm_provider": "bedrock",
                        },
                    }
                ]
            )

            # Verify router has model-specific credentials
            deployments = router.get_model_list(model_name="claude-opus-4-1")
            assert len(deployments) > 0
            deployment = deployments[0]
            deployment_litellm_params = deployment.get("litellm_params", {})

            # Verify model-specific credentials are in the deployment
            assert (
                deployment_litellm_params.get("aws_access_key_id") == model_access_key
            )
            assert (
                deployment_litellm_params.get("aws_secret_access_key")
                == model_secret_key
            )
            assert deployment_litellm_params.get("aws_region_name") == model_region
            assert (
                deployment_litellm_params.get("aws_session_token")
                == model_session_token
            )

            # Verify environment variables are NOT in the deployment
            assert deployment_litellm_params.get("aws_access_key_id") != env_access_key
            assert (
                deployment_litellm_params.get("aws_secret_access_key") != env_secret_key
            )
            assert deployment_litellm_params.get("aws_region_name") != env_region

            # Test 3: Verify credentials are passed through the passthrough route
            # Mock the passthrough route to capture what credentials are used
            captured_kwargs = {}

            async def mock_llm_passthrough_route(**kwargs):
                captured_kwargs.update(kwargs)
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.aread = AsyncMock(
                    return_value=b'{"content": [{"text": "Hello"}]}'
                )
                return mock_response

            mock_request = MagicMock(spec=Request)
            mock_request.method = "POST"
            mock_request.headers = {"content-type": "application/json"}
            mock_request.query_params = {}
            mock_request.url = MagicMock()
            mock_request.url.path = "/bedrock/model/claude-opus-4-1/converse"

            mock_request_body = {
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}]
            }

            mock_user_api_key_dict = Mock()
            mock_user_api_key_dict.api_key = "test-key"
            mock_proxy_logging_obj = Mock()
            mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

            with (
                patch(
                    "litellm.passthrough.main.llm_passthrough_route",
                    new_callable=AsyncMock,
                    side_effect=mock_llm_passthrough_route,
                ),
                patch(
                    "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_passthrough_process_llm_request",
                    new_callable=AsyncMock,
                ) as mock_process,
            ):
                # Setup mock response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.aread = AsyncMock(
                    return_value=b'{"content": [{"text": "Hello"}]}'
                )
                mock_process.return_value = mock_response

                # Call the handler
                await handle_bedrock_passthrough_router_model(
                    model="claude-opus-4-1",
                    endpoint="model/claude-opus-4-1/converse",
                    request=mock_request,
                    request_body=mock_request_body,
                    llm_router=router,
                    user_api_key_dict=mock_user_api_key_dict,
                    proxy_logging_obj=mock_proxy_logging_obj,
                    general_settings={},
                    proxy_config=None,
                    select_data_generator=None,
                    user_model=None,
                    user_temperature=None,
                    user_request_timeout=None,
                    user_max_tokens=None,
                    user_api_base=None,
                    version=None,
                )

                # Verify that the router was called (which means credentials flow through)
                # The key verification is that get_litellm_params extracts the credentials
                # and they're available in the router's deployment
                assert mock_process.called

    @pytest.mark.asyncio
    async def test_key_guardrail_blocks_bedrock_converse_passthrough(self):
        """
        Regression: key/team guardrails must fire for /bedrock/model/.../converse requests.
        Before the fix, CallTypes.allm_passthrough_route was not in the guardrail
        translation registry, so UnifiedLLMGuardrails silently skipped all guardrails.
        """
        from fastapi import HTTPException

        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.llms.pass_through.guardrail_translation import (
            guardrail_translation_mappings,
        )
        from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs

        assert CallTypes.allm_passthrough_route in guardrail_translation_mappings, (
            "allm_passthrough_route missing from guardrail_translation_mappings; "
            "this is the regression that lets guardrails bypass bedrock passthrough"
        )

        class _BlockingGuardrail(CustomGuardrail):
            async def apply_guardrail(
                self,
                inputs: GenericGuardrailAPIInputs,
                request_data: dict,
                input_type: str,
                logging_obj=None,
            ) -> GenericGuardrailAPIInputs:
                raise HTTPException(status_code=400, detail="Blocked by guardrail")

        handler_cls = guardrail_translation_mappings[CallTypes.allm_passthrough_route]
        handler = handler_cls()

        guardrail = _BlockingGuardrail(guardrail_name="block-all")

        data = {
            "custom_llm_provider": "bedrock",
            "endpoint": "model/anthropic.claude-3-sonnet-20240229-v1:0/converse",
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "data": {
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
            },
        }

        with pytest.raises(HTTPException) as exc_info:
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert exc_info.value.status_code == 400
        assert "Blocked by guardrail" in str(exc_info.value.detail)


class TestBedrockAgentRuntimePassthroughToggle:
    AGENT_RUNTIME_ENDPOINT: Final = "knowledgebases/KB1234567/retrieve"
    MODEL_ENDPOINT: Final = "model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse"
    DISABLED: Final = MappingProxyType({"disable_bedrock_agent_runtime_passthrough": True})

    @staticmethod
    def _mock_request() -> Mock:
        request: Final = Mock()
        request.method = "POST"
        request.state = SimpleNamespace()
        request.json = AsyncMock(return_value={"retrievalQuery": {"text": "hi"}})  # mutable-ok: must be json.dumps-able
        return request

    @contextlib.contextmanager
    def _patched_dispatch(self, general_settings: Mapping[str, object]):
        from botocore.credentials import Credentials

        bedrock_llm: Final = Mock()
        bedrock_llm.get_credentials = Mock(return_value=Credentials("ak", "sk"))
        forwarder: Final = AsyncMock(return_value="forwarded")

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.utils.get_secret", return_value="us-east-1"),
            patch("litellm.llms.bedrock.chat.BedrockConverseLLM", return_value=bedrock_llm),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_request_copy",
                Mock(),
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=forwarder,
            ) as create_route,
        ):
            yield create_route, forwarder

    @pytest.mark.asyncio
    async def test_agent_runtime_dispatch_allowed_by_default(self):
        with self._patched_dispatch(MappingProxyType({})) as (create_route, forwarder):
            result: Final = await bedrock_proxy_route(
                endpoint=self.AGENT_RUNTIME_ENDPOINT,
                request=self._mock_request(),
                fastapi_response=Mock(),
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert result == "forwarded"
        forwarder.assert_awaited_once()
        assert "bedrock-agent-runtime.us-east-1.amazonaws.com" in create_route.call_args.kwargs["target"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", (True, "true", "True"))
    async def test_agent_runtime_dispatch_rejected_when_disabled(self, value: bool | str):
        settings: Final = MappingProxyType({"disable_bedrock_agent_runtime_passthrough": value})

        with self._patched_dispatch(settings) as (create_route, forwarder):
            with pytest.raises(HTTPException) as exc_info:
                await bedrock_proxy_route(
                    endpoint=self.AGENT_RUNTIME_ENDPOINT,
                    request=self._mock_request(),
                    fastapi_response=Mock(),
                    user_api_key_dict=UserAPIKeyAuth(),
                )

        assert exc_info.value.status_code == 403
        assert "bedrock-agent-runtime pass-through is disabled" in str(exc_info.value.detail)
        create_route.assert_not_called()
        forwarder.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_model_invoke_still_routed_when_agent_runtime_disabled(self):
        with (
            patch("litellm.proxy.proxy_server.general_settings", self.DISABLED),
            patch("litellm.utils.get_secret", return_value="us-east-1"),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_request_copy",
                Mock(),
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.bedrock_llm_proxy_route",
                new=AsyncMock(return_value="llm-route"),
            ) as llm_route,
        ):
            result: Final = await bedrock_proxy_route(
                endpoint=self.MODEL_ENDPOINT,
                request=self._mock_request(),
                fastapi_response=Mock(),
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert result == "llm-route"
        llm_route.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", (False, "false", None, "", "yes"))
    async def test_agent_runtime_dispatch_allowed_for_non_true_values(self, value: object):
        settings: Final = MappingProxyType({"disable_bedrock_agent_runtime_passthrough": value})

        with self._patched_dispatch(settings) as (create_route, forwarder):
            result: Final = await bedrock_proxy_route(
                endpoint=self.AGENT_RUNTIME_ENDPOINT,
                request=self._mock_request(),
                fastapi_response=Mock(),
                user_api_key_dict=UserAPIKeyAuth(),
            )

        assert result == "forwarded"
        create_route.assert_called_once()


class TestLLMPassthroughFactoryProxyRoute:
    @pytest.mark.asyncio
    async def test_llm_passthrough_factory_proxy_route_success(self):
        from litellm.types.utils import LlmProviders

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.json = AsyncMock(return_value={"stream": False})
        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.utils.ProviderConfigManager.get_provider_model_info"
            ) as mock_get_provider,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
            ) as mock_get_creds,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
        ):
            mock_provider_config = MagicMock()
            mock_provider_config.get_api_base.return_value = "https://example.com/v1"
            mock_provider_config.validate_environment.return_value = {
                "x-api-key": "dummy"
            }
            mock_get_provider.return_value = mock_provider_config
            mock_get_creds.return_value = "dummy"

            mock_endpoint_func = AsyncMock(return_value="success")
            mock_create_route.return_value = mock_endpoint_func

            result = await llm_passthrough_factory_proxy_route(
                custom_llm_provider=LlmProviders.VLLM,
                endpoint="/chat/completions",
                request=mock_request,
                fastapi_response=mock_fastapi_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            assert result == "success"
            mock_get_provider.assert_called_once_with(
                provider=litellm.LlmProviders(LlmProviders.VLLM), model=None
            )
            mock_get_creds.assert_called_once_with(
                custom_llm_provider=LlmProviders.VLLM, region_name=None
            )
            mock_create_route.assert_called_once_with(
                endpoint="/chat/completions",
                target="https://example.com/v1/chat/completions",
                custom_headers={"x-api-key": "dummy"},
                is_streaming_request=False,
            )
            mock_endpoint_func.assert_awaited_once()


class TestVLLMProxyRoute:
    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
        return_value={"model": "router-model", "stream": False},
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_passthrough_request_using_router_model",
        return_value=True,
    )
    @patch("litellm.proxy.proxy_server.llm_router")
    async def test_vllm_proxy_route_with_router_model(
        self, mock_llm_router, mock_is_router, mock_get_body
    ):
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        mock_llm_router.allm_passthrough_route = AsyncMock(
            return_value=httpx.Response(200, json={"response": "success"})
        )

        await vllm_proxy_route(
            endpoint="/chat/completions",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        mock_is_router.assert_called_once()
        mock_llm_router.allm_passthrough_route.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
        return_value={"model": "other-model"},
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_passthrough_request_using_router_model",
        return_value=False,
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.llm_passthrough_factory_proxy_route"
    )
    async def test_vllm_proxy_route_fallback_to_factory(
        self, mock_factory_route, mock_is_router, mock_get_body
    ):
        mock_request = MagicMock(spec=Request)
        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()
        mock_factory_route.return_value = "factory_success"

        result = await vllm_proxy_route(
            endpoint="/chat/completions",
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

        assert result == "factory_success"
        mock_factory_route.assert_awaited_once()


class TestForwardHeaders:
    """
    Test cases for _forward_headers parameter in passthrough endpoints
    """

    @pytest.mark.asyncio
    async def test_pass_through_request_with_forward_headers_true(self):
        """
        Test that when forward_headers=True, user headers from the main request
        are forwarded to the target endpoint (except content-length and host)
        """
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        # Create a mock request with custom headers
        mock_request = MagicMock(spec=Request)
        mock_request.state = (
            None  # Prevent MagicMock from returning a truthy _cached_headers
        )
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/test/endpoint"

        # User headers that should be forwarded
        user_headers = {
            "x-custom-header": "custom-value",
            "x-api-key": "user-api-key",
            "authorization": "Bearer user-token",
            "user-agent": "test-client/1.0",
            "content-type": "application/json",
            # These should NOT be forwarded
            "content-length": "123",
            "host": "original-host.com",
        }
        mock_request.headers = user_headers
        mock_request.query_params = {}

        # Mock the request body
        mock_request_body = {"test": "data"}

        mock_user_api_key_dict = MagicMock()

        # Custom headers that should be merged with user headers
        custom_headers = {
            "x-litellm-header": "litellm-value",
        }

        target_url = "https://api.example.com/v1/test"

        # Mock the httpx client and response
        mock_httpx_response = MagicMock()
        mock_httpx_response.status_code = 200
        mock_httpx_response.headers = {"content-type": "application/json"}
        mock_httpx_response.aiter_bytes = AsyncMock(
            return_value=[b'{"result": "success"}']
        )
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value=mock_request_body,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging_obj,
        ):
            # Setup mock httpx client
            mock_client = MagicMock()
            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            # Setup mock logging object
            mock_logging_obj.pre_call_hook = AsyncMock(return_value=mock_request_body)
            mock_logging_obj.post_call_success_hook = AsyncMock()
            mock_logging_obj.post_call_failure_hook = AsyncMock()
            mock_logging_obj.post_call_response_headers_hook = AsyncMock(
                return_value={}
            )

            # Call pass_through_request with forward_headers=True
            result = await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers=custom_headers,
                user_api_key_dict=mock_user_api_key_dict,
                forward_headers=True,  # Enable header forwarding
                stream=False,
            )

            # Verify the httpx client was called
            assert mock_client.send.called

            # Get the headers that were sent to the target
            call_args = mock_client.build_request.call_args
            sent_headers = call_args[1]["headers"]

            # Verify user headers were forwarded (except content-length and host)
            assert sent_headers["x-custom-header"] == "custom-value"
            assert sent_headers["x-api-key"] == "user-api-key"
            assert sent_headers["authorization"] == "Bearer user-token"
            assert sent_headers["user-agent"] == "test-client/1.0"
            assert sent_headers["content-type"] == "application/json"

            # Verify custom headers were included
            assert sent_headers["x-litellm-header"] == "litellm-value"

            # Verify content-length and host were NOT forwarded
            assert "content-length" not in sent_headers
            assert "host" not in sent_headers

    @pytest.mark.asyncio
    async def test_pass_through_request_with_forward_headers_false(self):
        """
        Test that when forward_headers=False (default), user headers are NOT forwarded,
        only custom_headers are sent
        """
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        # Create a mock request with custom headers
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/test/endpoint"

        # User headers that should NOT be forwarded
        user_headers = {
            "x-custom-header": "custom-value",
            "x-api-key": "user-api-key",
            "authorization": "Bearer user-token",
        }
        mock_request.headers = user_headers
        mock_request.query_params = {}

        mock_request_body = {"test": "data"}
        mock_user_api_key_dict = MagicMock()

        # Only these custom headers should be sent
        custom_headers = {
            "x-litellm-header": "litellm-value",
            "authorization": "Bearer litellm-token",
        }

        target_url = "https://api.example.com/v1/test"

        # Mock the httpx client and response
        mock_httpx_response = MagicMock()
        mock_httpx_response.status_code = 200
        mock_httpx_response.headers = {"content-type": "application/json"}
        mock_httpx_response.aiter_bytes = AsyncMock(
            return_value=[b'{"result": "success"}']
        )
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value=mock_request_body,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging_obj,
        ):
            # Setup mock httpx client
            mock_client = MagicMock()
            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            # Setup mock logging object
            mock_logging_obj.pre_call_hook = AsyncMock(return_value=mock_request_body)
            mock_logging_obj.post_call_success_hook = AsyncMock()
            mock_logging_obj.post_call_failure_hook = AsyncMock()
            mock_logging_obj.post_call_response_headers_hook = AsyncMock(
                return_value={}
            )

            # Call pass_through_request with forward_headers=False (default)
            result = await pass_through_request(
                request=mock_request,
                target=target_url,
                custom_headers=custom_headers,
                user_api_key_dict=mock_user_api_key_dict,
                forward_headers=False,  # Explicitly set to False
                stream=False,
            )

            # Verify the httpx client was called
            assert mock_client.send.called

            # Get the headers that were sent to the target
            call_args = mock_client.build_request.call_args
            sent_headers = call_args[1]["headers"]

            # Verify only custom headers were sent
            assert sent_headers["x-litellm-header"] == "litellm-value"
            assert sent_headers["authorization"] == "Bearer litellm-token"

            # Verify user headers were NOT forwarded
            assert "x-custom-header" not in sent_headers
            assert "x-api-key" not in sent_headers
            # Authorization is present but should be from custom_headers, not user headers
            assert sent_headers["authorization"] == "Bearer litellm-token"

    @pytest.mark.asyncio
    async def test_llm_passthrough_factory_with_forward_headers(self):
        """
        Test that _forward_headers works correctly in llm_passthrough_factory_proxy_route
        which is used in the code snippet provided by the user
        """
        from litellm.types.utils import LlmProviders

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = MagicMock()
        mock_request.url.path = "/openai/chat/completions"

        # User headers to be forwarded
        user_headers = {
            "x-custom-tracking-id": "tracking-123",
            "x-request-id": "req-456",
            "user-agent": "my-app/2.0",
        }
        mock_request.headers = user_headers
        mock_request.json = AsyncMock(return_value={"stream": False})

        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        # Mock the httpx response
        mock_httpx_response = MagicMock()
        mock_httpx_response.status_code = 200
        mock_httpx_response.headers = {"content-type": "application/json"}
        mock_httpx_response.aiter_bytes = AsyncMock(
            return_value=[b'{"result": "success"}']
        )
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with (
            patch(
                "litellm.utils.ProviderConfigManager.get_provider_model_info"
            ) as mock_get_provider,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
            ) as mock_get_creds,
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
                return_value={"messages": [{"role": "user", "content": "test"}]},
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
            ) as mock_get_client,
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging_obj,
        ):
            # Setup provider config
            mock_provider_config = MagicMock()
            mock_provider_config.get_api_base.return_value = "https://api.openai.com/v1"
            mock_provider_config.validate_environment.return_value = {
                "authorization": "Bearer sk-test"
            }
            mock_get_provider.return_value = mock_provider_config
            mock_get_creds.return_value = "sk-test"

            # Setup mock httpx client
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            # Setup mock logging object
            mock_logging_obj.pre_call_hook = AsyncMock(
                return_value={"messages": [{"role": "user", "content": "test"}]}
            )
            mock_logging_obj.post_call_success_hook = AsyncMock()

            # This is the key part - when create_pass_through_route is called with _forward_headers=True
            # it should forward the user headers
            with patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route:
                mock_endpoint_func = AsyncMock(return_value="success")
                mock_create_route.return_value = mock_endpoint_func

                result = await llm_passthrough_factory_proxy_route(
                    custom_llm_provider=LlmProviders.OPENAI,
                    endpoint="/chat/completions",
                    request=mock_request,
                    fastapi_response=mock_fastapi_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

                # Verify create_pass_through_route was called
                mock_create_route.assert_called_once()

                # Get the call arguments to verify _forward_headers parameter
                call_kwargs = mock_create_route.call_args[1]

                # Note: The current implementation doesn't explicitly pass _forward_headers
                # This test documents the current behavior. If _forward_headers should be
                # configurable in llm_passthrough_factory_proxy_route, it would need to be added


class TestMilvusProxyRoute:
    """
    Test cases for Milvus passthrough endpoint
    """

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_success(self):
        """
        Test successful Milvus proxy route with valid managed vector store index
        """

        collection_name = "dall-e-6"
        vector_store_name = "milvus-store-1"
        vector_store_index = "collection_123"
        api_base = "http://localhost:19530"

        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.url = MagicMock()
        mock_request.url.path = "/milvus/vectors/search"

        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        # Mock vector store index object
        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        # Mock vector store
        mock_vector_store = {
            "litellm_params": {
                "api_base": api_base,
                "api_key": "test-milvus-key",
            }
        }

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
                return_value={"collectionName": collection_name, "data": [[0.1, 0.2]]},
            ) as mock_get_body,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
            ) as mock_get_config,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
            ) as mock_is_allowed,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
            ) as mock_safe_set,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
            patch.object(litellm, "vector_store_index_registry") as mock_index_registry,
            patch.object(litellm, "vector_store_registry") as mock_vector_registry,
        ):
            # Setup mocks
            mock_provider_config = MagicMock()
            mock_provider_config.get_auth_credentials.return_value = {
                "headers": {"Authorization": "Bearer test-token"}
            }
            mock_provider_config.get_complete_url.return_value = api_base
            mock_get_config.return_value = mock_provider_config

            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )

            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                mock_vector_store
            )

            mock_endpoint_func = AsyncMock(
                return_value={"results": [{"id": 1, "distance": 0.5}]}
            )
            mock_create_route.return_value = mock_endpoint_func

            # Call the route
            result = await milvus_proxy_route(
                endpoint="vectors/search",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify calls
            mock_get_body.assert_called_once()
            mock_index_registry.is_vector_store_index.assert_called_once_with(
                vector_store_index_name=collection_name
            )
            mock_is_allowed.assert_called_once()
            mock_safe_set.assert_called_once()

            # Verify collection name was updated to the actual index
            set_body_call_args = mock_safe_set.call_args[0]
            assert set_body_call_args[1]["collectionName"] == vector_store_index

            # Verify create_pass_through_route was called with correct URL
            mock_create_route.assert_called_once()
            create_route_args = mock_create_route.call_args[1]
            assert "vectors/search" in create_route_args["target"]
            assert create_route_args["custom_headers"] == {
                "Authorization": "Bearer test-token"
            }

            # Verify endpoint function was called
            mock_endpoint_func.assert_awaited_once()
            assert result == {"results": [{"id": 1, "distance": 0.5}]}

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_missing_collection_name(self):
        """
        Test that missing collection name raises HTTPException
        """
        from fastapi import HTTPException


        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
                return_value={"data": [[0.1, 0.2]]},  # No collectionName
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
            ) as mock_get_config,
        ):
            mock_get_config.return_value = MagicMock()

            with pytest.raises(HTTPException) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 400
            assert "Collection name is required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_no_provider_config(self):
        """
        Test that missing provider config raises HTTPException
        """
        from fastapi import HTTPException


        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 500
            assert "Unable to find Milvus vector store config" in str(
                exc_info.value.detail
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_no_index_registry(self):
        """
        Test that missing index registry raises HTTPException
        """
        from fastapi import HTTPException


        collection_name = "test-collection"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
                return_value={"collectionName": collection_name},
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
            ) as mock_get_config,
            patch.object(litellm, "vector_store_index_registry", None),
        ):
            mock_get_config.return_value = MagicMock()

            with pytest.raises(HTTPException) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 500
            assert "Unable to find Milvus vector store index registry" in str(
                exc_info.value.detail
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_not_managed_index(self):
        """
        Test that non-managed vector store index raises HTTPException
        """
        from fastapi import HTTPException


        collection_name = "unmanaged-collection"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
                return_value={"collectionName": collection_name},
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
            ) as mock_get_config,
            patch.object(litellm, "vector_store_index_registry") as mock_index_registry,
            patch.object(litellm, "vector_store_registry", MagicMock()),
        ):
            mock_get_config.return_value = MagicMock()
            mock_index_registry.is_vector_store_index.return_value = False

            with pytest.raises(HTTPException) as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert exc_info.value.status_code == 400
            assert (
                f"Collection {collection_name} is not a litellm managed vector store index"
                in str(exc_info.value.detail)
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_vector_store_not_found(self):
        """
        Test that missing vector store raises Exception
        """

        collection_name = "test-collection"
        vector_store_name = "missing-store"
        vector_store_index = "collection_123"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
                return_value={"collectionName": collection_name},
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
            ) as mock_get_config,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
            ),
            patch.object(litellm, "vector_store_index_registry") as mock_index_registry,
            patch.object(litellm, "vector_store_registry") as mock_vector_registry,
        ):
            mock_get_config.return_value = MagicMock()
            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )
            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                None
            )

            with pytest.raises(Exception, match='Vector store not found for missing-store') as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert f"Vector store not found for {vector_store_name}" in str(
                exc_info.value
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_no_api_base(self):
        """
        Test that missing api_base raises Exception
        """

        collection_name = "test-collection"
        vector_store_name = "milvus-store-1"
        vector_store_index = "collection_123"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        mock_vector_store = {"litellm_params": {}}  # No api_base

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
                return_value={"collectionName": collection_name},
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
            ) as mock_get_config,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
            ),
            patch.object(litellm, "vector_store_index_registry") as mock_index_registry,
            patch.object(litellm, "vector_store_registry") as mock_vector_registry,
        ):
            mock_provider_config = MagicMock()
            mock_provider_config.get_auth_credentials.return_value = {"headers": {}}
            mock_provider_config.get_complete_url.return_value = None
            mock_get_config.return_value = mock_provider_config

            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )
            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                mock_vector_store
            )

            with pytest.raises(Exception, match='api_base not found in vector store configuration for') as exc_info:
                await milvus_proxy_route(
                    endpoint="vectors/search",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert (
                f"api_base not found in vector store configuration for {vector_store_name}"
                in str(exc_info.value)
            )

    @pytest.mark.asyncio
    async def test_milvus_proxy_route_endpoint_without_leading_slash(self):
        """
        Test that endpoint without leading slash is handled correctly
        """

        collection_name = "test-collection"
        vector_store_name = "milvus-store-1"
        vector_store_index = "collection_123"
        api_base = "http://localhost:19530"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        mock_vector_store = {"litellm_params": {"api_base": api_base}}

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
                return_value={"collectionName": collection_name},
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
            ) as mock_get_config,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
            patch.object(litellm, "vector_store_index_registry") as mock_index_registry,
            patch.object(litellm, "vector_store_registry") as mock_vector_registry,
        ):
            mock_provider_config = MagicMock()
            mock_provider_config.get_auth_credentials.return_value = {"headers": {}}
            mock_provider_config.get_complete_url.return_value = api_base
            mock_get_config.return_value = mock_provider_config

            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )
            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                mock_vector_store
            )

            mock_endpoint_func = AsyncMock(return_value={"status": "success"})
            mock_create_route.return_value = mock_endpoint_func

            # Call with endpoint without leading slash
            await milvus_proxy_route(
                endpoint="vectors/search",  # No leading slash
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify that the target URL has correct path
            create_route_args = mock_create_route.call_args[1]
            assert "/vectors/search" in create_route_args["target"]


class TestOpenAIPassthroughRoute:
    """
    Test cases for OpenAI passthrough endpoint (/openai_passthrough)
    """

    @pytest.mark.asyncio
    async def test_openai_passthrough_responses_api(self):
        """
        Test that /openai_passthrough endpoint correctly handles Responses API calls
        This verifies the fix for issue #18865 where /openai/v1/responses was being
        routed to LiteLLM's native implementation instead of passthrough
        """

        # Mock request for Responses API
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value="sk-test-key",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
        ):
            mock_endpoint_func = AsyncMock(
                return_value={"id": "resp_123", "status": "completed"}
            )
            mock_create_route.return_value = mock_endpoint_func

            # Call the route with /v1/responses endpoint
            result = await openai_proxy_route(
                endpoint="v1/responses",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify create_pass_through_route was called with correct target
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]

            # Should route to OpenAI's responses API
            assert call_args["target"] == "https://api.openai.com/v1/responses"
            assert call_args["endpoint"] == "v1/responses"

            # Verify headers contain API key
            assert "authorization" in call_args["custom_headers"]
            assert "Bearer sk-test-key" in call_args["custom_headers"]["authorization"]

            # Verify result
            assert result == {"id": "resp_123", "status": "completed"}

    @pytest.mark.asyncio
    async def test_openai_passthrough_chat_completions(self):
        """
        Test that /openai_passthrough works for chat completions
        """

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value="sk-test-key",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
        ):
            mock_endpoint_func = AsyncMock(
                return_value={"id": "chatcmpl-123", "choices": []}
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await openai_proxy_route(
                endpoint="v1/chat/completions",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify routing
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.openai.com/v1/chat/completions"

            # Verify result
            assert result == {"id": "chatcmpl-123", "choices": []}

    @pytest.mark.asyncio
    async def test_openai_passthrough_missing_api_key(self):
        """
        Test that missing OPENAI_API_KEY raises an exception
        """

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value=None,
        ):
            with pytest.raises(Exception, match="Required 'OPENAI_API_KEY' in environment to make") as exc_info:
                await openai_proxy_route(
                    endpoint="v1/chat/completions",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )

            assert "Required 'OPENAI_API_KEY'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openai_passthrough_assistants_api(self):
        """
        Test that /openai_passthrough works for Assistants API endpoints
        """

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/assistants"
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value="sk-test-key",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
        ):
            mock_endpoint_func = AsyncMock(
                return_value={"id": "asst_123", "object": "assistant"}
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await openai_proxy_route(
                endpoint="v1/assistants",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify routing
            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.openai.com/v1/assistants"

            # Verify headers contain API key and OpenAI-Beta header
            assert "authorization" in call_args["custom_headers"]

            # Verify result
            assert result == {"id": "asst_123", "object": "assistant"}


def _resolve_route_name(method: str, path: str) -> str | None:
    from starlette.routing import Match

    from litellm.proxy.proxy_server import app

    scope: Final = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "query_string": b"",
        "root_path": "",
    }
    for route in app.router.routes:
        if route.matches(scope)[0] == Match.FULL:
            return getattr(route, "name", None)
    return None


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/openai_passthrough/v1/files"),
        ("GET", "/openai_passthrough/v1/files"),
        ("GET", "/openai_passthrough/v1/files/file-abc123"),
        ("DELETE", "/openai_passthrough/v1/files/file-abc123"),
        ("GET", "/openai_passthrough/v1/files/file-abc123/content"),
        ("POST", "/openai_passthrough/v1/batches"),
        ("GET", "/openai_passthrough/v1/batches"),
        ("GET", "/openai_passthrough/v1/batches/batch_abc123"),
        ("POST", "/openai_passthrough/v1/batches/batch_abc123/cancel"),
        ("POST", "/openai_passthrough/v1/responses"),
    ],
)
def test_openai_passthrough_prefix_wins_over_native_provider_routes(method, path):
    """
    /openai_passthrough exists to guarantee passthrough, so the native
    /{provider}/v1/files and /{provider}/v1/batches routes must never capture it
    with provider="openai_passthrough" (which 500s on the LlmProviders lookup).
    """
    assert _resolve_route_name(method, path) == "openai_proxy_route"


@pytest.mark.parametrize(
    "method, path, expected_name",
    [
        ("POST", "/openai/v1/files", "create_file"),
        ("GET", "/azure/v1/files", "list_files"),
        ("POST", "/v1/files", "create_file"),
        ("POST", "/v1/batches", "create_batch"),
        ("POST", "/openai/v1/chat/completions", "openai_proxy_route"),
    ],
)
def test_native_provider_routes_are_unchanged(method, path, expected_name):
    assert _resolve_route_name(method, path) == expected_name


class TestCursorProxyRoute:
    """Tests for the Cursor Cloud Agents pass-through route."""

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_creates_pass_through_with_basic_auth(self):
        """should create a pass-through route with Basic Auth header for Cursor API"""
        import base64

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        test_api_key = "test-cursor-api-key-123"

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value=test_api_key,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
        ):
            mock_endpoint_func = AsyncMock(
                return_value={"agents": [], "nextCursor": None}
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await cursor_proxy_route(
                endpoint="v0/agents",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_create_route.assert_called_once()
            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.cursor.com/v0/agents"

            expected_auth = base64.b64encode(f"{test_api_key}:".encode("utf-8")).decode(
                "ascii"
            )
            assert (
                call_args["custom_headers"]["Authorization"] == f"Basic {expected_auth}"
            )

            assert result == {"agents": [], "nextCursor": None}

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_raises_on_missing_api_key(self):
        """should raise 401 when no Cursor API key is available"""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value=None,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.credential_list",
                [],
            ),
        ):
            with pytest.raises(Exception, match='Cursor API key not found\\. Add Cursor credentials via') as exc_info:
                await cursor_proxy_route(
                    endpoint="v0/agents",
                    request=mock_request,
                    fastapi_response=mock_response,
                    user_api_key_dict=mock_user_api_key_dict,
                )
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_uses_ui_credential(self):
        """should use credentials added via UI (litellm.credential_list) when env var is not set"""
        from litellm.types.utils import CredentialItem

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        ui_credential = CredentialItem(
            credential_name="my-cursor-key",
            credential_values={
                "api_key": "crsr_ui_test_key",
                "api_base": "https://api.cursor.com",
            },
            credential_info={"custom_llm_provider": "cursor"},
        )

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value=None,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.litellm.credential_list",
                [ui_credential],
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
        ):
            mock_endpoint_func = AsyncMock(return_value={"models": []})
            mock_create_route.return_value = mock_endpoint_func

            result = await cursor_proxy_route(
                endpoint="v0/models",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.cursor.com/v0/models"

            import base64

            expected_auth = base64.b64encode(b"crsr_ui_test_key:").decode("ascii")
            assert (
                call_args["custom_headers"]["Authorization"] == f"Basic {expected_auth}"
            )

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_custom_api_base(self):
        """should use CURSOR_API_BASE env var when set"""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch.dict(
                os.environ, {"CURSOR_API_BASE": "https://custom-cursor.example.com"}
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value="test-key",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
        ):
            mock_endpoint_func = AsyncMock(return_value={})
            mock_create_route.return_value = mock_endpoint_func

            await cursor_proxy_route(
                endpoint="v0/me",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://custom-cursor.example.com/v0/me"

    @pytest.mark.asyncio
    async def test_cursor_proxy_route_launch_agent(self):
        """should handle POST to launch an agent through the pass-through"""
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.headers = {"content-type": "application/json"}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value="test-key",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
            ) as mock_create_route,
        ):
            mock_endpoint_func = AsyncMock(
                return_value={
                    "id": "bc_abc123",
                    "name": "Test Agent",
                    "status": "CREATING",
                }
            )
            mock_create_route.return_value = mock_endpoint_func

            result = await cursor_proxy_route(
                endpoint="v0/agents",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            call_args = mock_create_route.call_args[1]
            assert call_args["target"] == "https://api.cursor.com/v0/agents"
            assert result["id"] == "bc_abc123"
            assert result["status"] == "CREATING"


class TestVertexRawPredictStreamingClassification:
    """
    Regression coverage for LIT-4761.

    `_base_vertex_proxy_route` classified any target URL containing "stream" as a
    streaming request. `:streamRawPredict` carries that substring, so a unary
    Anthropic-on-Vertex call (no `stream` field in the body) was sent with
    `?alt=sse` and logged through the streaming chunk collector, which parses
    Anthropic SSE deltas and finds no usage in a complete `"type": "message"`
    body; the spend log recorded 0 tokens and $0 cost.

    Streaming for the rawPredict family is decided by the request body, per the
    Anthropic Messages contract. The Gemini generateContent family stays
    URL-signalled because the Gemini REST body has no `stream` field.
    """

    RAW_PREDICT_ENDPOINT = (
        "v1/projects/test-project/locations/us-east5/publishers/anthropic/models/"
        "claude-sonnet-4-6:streamRawPredict"
    )
    GENERATE_CONTENT_ENDPOINT = (
        "v1/projects/test-project/locations/us-east5/publishers/google/models/"
        "gemini-2.5-flash:streamGenerateContent"
    )

    async def _capture_passthrough_kwargs(self, endpoint: str, body: object) -> dict:
        raw_body = json.dumps(body).encode("utf-8")

        async def receive():
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": f"/vertex_ai/{endpoint}",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"x-litellm-api-key", b"test-key"),
                    (b"authorization", b"Bearer ya29.byo-google-oauth"),
                ],
                "query_string": b"",
            },
            receive=receive,
        )

        captured: dict = {}

        def fake_create_pass_through_route(**kwargs):
            captured.update(kwargs)
            return AsyncMock(return_value={"status": "success"})

        mock_credentials = Mock()
        mock_credentials.token = "test-token"

        base_url = "https://us-east5-aiplatform.googleapis.com/"
        mock_handler = Mock()
        mock_handler.get_default_base_target_url.return_value = base_url
        mock_handler.update_base_target_url_with_credential_location = Mock(return_value=base_url)

        module = "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
        with (
            mock.patch(
                "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth",
                return_value=(mock_credentials, "test-project"),
            ),
            mock.patch(f"{module}.create_pass_through_route", side_effect=fake_create_pass_through_route),
            mock.patch(f"{module}.get_litellm_virtual_key", return_value="Bearer test-key"),
            mock.patch(f"{module}.user_api_key_auth", new=AsyncMock(return_value=UserAPIKeyAuth(api_key="test-key"))),
            mock.patch(f"{module}.get_vertex_pass_through_handler", return_value=mock_handler),
        ):
            await vertex_proxy_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            )

        assert captured, "create_pass_through_route was never called"
        return captured

    @pytest.mark.asyncio
    async def test_raw_predict_without_stream_field_is_not_streaming(self):
        captured = await self._capture_passthrough_kwargs(
            endpoint=self.RAW_PREDICT_ENDPOINT,
            body={
                "anthropic_version": "vertex-2023-10-16",
                "messages": [{"role": "user", "content": "Explain MLOps"}],
                "max_tokens": 5000,
            },
        )

        assert captured["is_streaming_request"] is False
        assert "alt=sse" not in captured["target"]

    @pytest.mark.asyncio
    async def test_raw_predict_with_stream_false_is_not_streaming(self):
        captured = await self._capture_passthrough_kwargs(
            endpoint=self.RAW_PREDICT_ENDPOINT,
            body={
                "anthropic_version": "vertex-2023-10-16",
                "stream": False,
                "messages": [{"role": "user", "content": "Explain MLOps"}],
                "max_tokens": 5000,
            },
        )

        assert captured["is_streaming_request"] is False
        assert "alt=sse" not in captured["target"]

    @pytest.mark.asyncio
    async def test_raw_predict_with_stream_true_still_streams(self):
        captured = await self._capture_passthrough_kwargs(
            endpoint=self.RAW_PREDICT_ENDPOINT,
            body={
                "anthropic_version": "vertex-2023-10-16",
                "stream": True,
                "messages": [{"role": "user", "content": "Explain MLOps"}],
                "max_tokens": 5000,
            },
        )

        assert captured["is_streaming_request"] is True
        assert captured["target"].endswith("?alt=sse")

    @pytest.mark.asyncio
    async def test_gemini_stream_generate_content_stays_url_signalled(self):
        captured = await self._capture_passthrough_kwargs(
            endpoint=self.GENERATE_CONTENT_ENDPOINT,
            body={"contents": [{"role": "user", "parts": [{"text": "Explain MLOps"}]}]},
        )

        assert captured["is_streaming_request"] is True
        assert captured["target"].endswith("?alt=sse")

    @pytest.mark.asyncio
    async def test_raw_predict_with_non_object_body_is_not_streaming(self):
        captured = await self._capture_passthrough_kwargs(
            endpoint=self.RAW_PREDICT_ENDPOINT,
            body=[{"role": "user", "content": "Explain MLOps"}],
        )

        assert captured["is_streaming_request"] is False
        assert "alt=sse" not in captured["target"]


@pytest.mark.parametrize(
    "request_body, expected",
    [
        ({"stream": True}, True),
        ({"stream": "true"}, True),
        ({"stream": False}, False),
        ({}, False),
        ([{"role": "user"}], False),
        ([], False),
        ("stream", False),
        (7, False),
        (None, False),
    ],
)
def test_is_passthrough_request_streaming_tolerates_non_object_bodies(request_body, expected):
    """
    A JSON request body is not required to be an object. Every passthrough
    streaming decision funnels through this predicate, so a list or scalar body
    must answer False instead of raising AttributeError.
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        is_passthrough_request_streaming,
    )

    assert is_passthrough_request_streaming(request_body) is expected


def _unsigned_jwt(claims: Mapping[str, str]) -> str:
    def segment(payload: Mapping[str, str]) -> str:
        return base64.urlsafe_b64encode(json.dumps(dict(payload)).encode()).rstrip(b"=").decode()

    return ".".join((segment({"alg": "RS256", "typ": "JWT"}), segment(claims), "c2lnbmF0dXJl"))


class TestVertexCredentiallessPassthroughVirtualKeyLeak:
    """Regression coverage for LIT-5997.

    With no Vertex credential configured, the passthrough took the
    bring-your-own-credentials branch and forwarded the whole incoming header set
    to Google, including whichever header carried the caller's LiteLLM virtual key.
    LiteLLM accepts that key from several headers (``Authorization``,
    ``x-litellm-api-key``, ``x-goog-api-key``, ``api-key``, ``x-api-key``), and
    ``x-goog-api-key`` doubles as a genuine Google credential, so any of them could
    leak the proxy's own secret to an upstream provider.

    A credential-less request that carries no upstream Google credential must now
    fail with a clean 401 and never reach ``create_pass_through_route``. The
    proxy-only auth headers Google never consumes (``x-litellm-api-key``,
    ``api-key``, ``x-api-key``) are dropped by name, and the virtual key is dropped
    by value from ``Authorization`` / ``x-goog-api-key``, which may instead carry a
    genuine bring-your-own Google credential that must still pass through. The
    by-value strip also covers a virtual key sent in the operator-configured
    ``general_settings.litellm_key_header_name``, whatever that header is named.

    The by-value strip keys off what actually authenticated the caller (the
    master key, or the LiteLLM key whose hash ``user_api_key_auth`` resolved as
    ``api_key``), never off header precedence: a custom auth or JWT that
    authenticated the caller without consuming ``Authorization`` leaves the
    caller's own Google token there, and it must keep flowing.
    """

    VKEY = "sk-litellm-victim-key"
    ENDPOINT = (
        "v1/projects/my-proj/locations/us-central1/publishers/google/models/"
        "gemini-2.5-flash:generateContent"
    )

    async def _run(
        self,
        monkeypatch,
        headers: list[tuple[bytes, bytes]],
        authenticated: UserAPIKeyAuth | None = None,
        master_key: str | None = "sk-master-1234",
    ) -> tuple[HTTPException | None, dict | None]:
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", master_key)
        caller: Final = authenticated if authenticated is not None else UserAPIKeyAuth(api_key=self.VKEY)
        from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
            PassthroughEndpointRouter,
        )

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": f"/vertex_ai/{self.ENDPOINT}",
                "headers": headers,
                "query_string": b"",
            },
            receive=receive,
        )

        captured: dict = {}

        def fake_create_pass_through_route(**kwargs):
            captured.update(kwargs)
            return AsyncMock(return_value={"status": "success"})

        mock_handler = Mock()
        mock_handler.get_default_base_target_url.return_value = "https://us-central1-aiplatform.googleapis.com/"

        module = "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
        monkeypatch.setattr(f"{module}.passthrough_endpoint_router", PassthroughEndpointRouter())
        raised: HTTPException | None = None
        with (
            mock.patch(f"{module}.create_pass_through_route", side_effect=fake_create_pass_through_route),
            mock.patch(f"{module}.user_api_key_auth", new=AsyncMock(return_value=caller)),
            mock.patch(f"{module}.get_vertex_pass_through_handler", return_value=mock_handler),
        ):
            try:
                await vertex_proxy_route(
                    endpoint=self.ENDPOINT,
                    request=request,
                    fastapi_response=Response(),
                    user_api_key_dict=caller,
                )
            except HTTPException as exc:
                raised = exc

        return raised, (captured.get("custom_headers") if captured else None)

    @pytest.mark.asyncio
    async def test_authorization_bearer_virtual_key_is_rejected_not_forwarded(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [(b"authorization", f"Bearer {self.VKEY}".encode()), (b"content-type", b"application/json")],
        )
        assert forwarded is None, "credential-less request must never reach the upstream forwarder"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_x_litellm_api_key_virtual_key_is_rejected_not_forwarded(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [(b"x-litellm-api-key", self.VKEY.encode()), (b"content-type", b"application/json")],
        )
        assert forwarded is None, "credential-less request must never reach the upstream forwarder"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_x_goog_api_key_carrying_virtual_key_is_rejected_not_forwarded(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"x-litellm-api-key", self.VKEY.encode()),
                (b"x-goog-api-key", self.VKEY.encode()),
                (b"content-type", b"application/json"),
            ],
        )
        assert forwarded is None, "the virtual key in x-goog-api-key must not satisfy the gate nor be forwarded"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_virtual_key_authenticated_solely_via_x_goog_api_key_is_rejected(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"x-goog-api-key", self.VKEY.encode()),
                (b"content-type", b"application/json"),
            ],
        )
        assert forwarded is None, "a virtual key that authenticated via x-goog-api-key must be stripped, not forwarded"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_byo_google_oauth_token_still_forwards_without_virtual_key(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"x-litellm-api-key", self.VKEY.encode()),
                (b"authorization", b"Bearer ya29.google-oauth-token"),
                (b"content-type", b"application/json"),
            ],
        )
        assert raised is None
        assert forwarded is not None
        assert forwarded.get("authorization") == "Bearer ya29.google-oauth-token"
        assert "x-litellm-api-key" not in forwarded
        assert self.VKEY not in " ".join(f"{name}:{value}" for name, value in forwarded.items())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "Basic"])
    async def test_virtual_key_echoed_in_authorization_with_any_scheme_is_stripped(self, monkeypatch, scheme):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"x-litellm-api-key", self.VKEY.encode()),
                (b"authorization", f"{scheme} {self.VKEY}".encode()),
                (b"content-type", b"application/json"),
            ],
        )
        assert forwarded is None, f"a virtual key echoed as '{scheme} <key>' in Authorization must be stripped, not forwarded"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_byo_x_goog_api_key_still_forwards_without_virtual_key(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"x-litellm-api-key", self.VKEY.encode()),
                (b"x-goog-api-key", b"AIza-google-api-key"),
                (b"content-type", b"application/json"),
            ],
        )
        assert raised is None
        assert forwarded is not None
        assert forwarded.get("x-goog-api-key") == "AIza-google-api-key"
        assert "x-litellm-api-key" not in forwarded
        assert self.VKEY not in " ".join(f"{name}:{value}" for name, value in forwarded.items())

    @pytest.mark.asyncio
    async def test_alternate_proxy_auth_headers_are_never_forwarded_to_google(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"x-litellm-api-key", self.VKEY.encode()),
                (b"authorization", b"Bearer ya29.google-oauth-token"),
                (b"api-key", b"azure-style-caller-secret"),
                (b"x-api-key", b"anthropic-style-caller-secret"),
                (b"content-type", b"application/json"),
            ],
        )
        assert raised is None
        assert forwarded is not None
        assert forwarded.get("authorization") == "Bearer ya29.google-oauth-token"
        assert "api-key" not in forwarded
        assert "x-api-key" not in forwarded
        assert "x-litellm-api-key" not in forwarded
        forwarded_blob = " ".join(f"{name}:{value}" for name, value in forwarded.items())
        assert self.VKEY not in forwarded_blob
        assert "azure-style-caller-secret" not in forwarded_blob
        assert "anthropic-style-caller-secret" not in forwarded_blob

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "credential_header",
        sorted(
            SpecialHeaders.litellm_credential_header_names()
            - {"authorization", "x-goog-api-key", "x-litellm-api-key"}
        ),
    )
    async def test_every_non_google_credential_header_is_dropped_by_name(self, monkeypatch, credential_header):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"x-litellm-api-key", self.VKEY.encode()),
                (b"x-goog-api-key", b"AIza-real-google-api-key"),
                (credential_header.encode(), b"some-distinct-caller-secret-value"),
                (b"content-type", b"application/json"),
            ],
        )
        assert raised is None
        assert forwarded is not None
        assert forwarded.get("x-goog-api-key") == "AIza-real-google-api-key"
        assert credential_header not in forwarded
        assert "x-litellm-api-key" not in forwarded
        forwarded_blob = " ".join(f"{name}:{value}" for name, value in forwarded.items())
        assert self.VKEY not in forwarded_blob
        assert "some-distinct-caller-secret-value" not in forwarded_blob

    @pytest.mark.asyncio
    async def test_virtual_key_in_operator_configured_header_is_stripped(self, monkeypatch):
        with mock.patch.dict(  # test-quality-ok: general_settings is the real proxy config surface for litellm_key_header_name; no injection seam exists on this route
            "litellm.proxy.proxy_server.general_settings",
            {"litellm_key_header_name": "x-company-key"},
        ):
            raised, forwarded = await self._run(
                monkeypatch,
                [
                    (b"x-company-key", f"Bearer {self.VKEY}".encode()),
                    (b"x-goog-api-key", b"AIza-real-google-api-key"),
                    (b"content-type", b"application/json"),
                ],
            )
        assert raised is None
        assert forwarded is not None
        assert forwarded.get("x-goog-api-key") == "AIza-real-google-api-key"
        assert "x-company-key" not in forwarded
        assert self.VKEY not in " ".join(f"{name}:{value}" for name, value in forwarded.items())

    @pytest.mark.asyncio
    async def test_virtual_key_in_operator_configured_header_alone_is_rejected(self, monkeypatch):
        with mock.patch.dict(  # test-quality-ok: general_settings is the real proxy config surface for litellm_key_header_name; no injection seam exists on this route
            "litellm.proxy.proxy_server.general_settings",
            {"litellm_key_header_name": "x-company-key"},
        ):
            raised, forwarded = await self._run(
                monkeypatch,
                [
                    (b"x-company-key", f"Bearer {self.VKEY}".encode()),
                    (b"content-type", b"application/json"),
                ],
            )
        assert forwarded is None, "a virtual key in the custom auth header must not satisfy the gate nor be forwarded"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_virtual_key_in_pass_through_configured_header_is_dropped_and_rejected(self, monkeypatch):
        with mock.patch.dict(  # test-quality-ok: general_settings is the real proxy config surface for pass_through_endpoints; no injection seam exists on this route
            "litellm.proxy.proxy_server.general_settings",
            {"pass_through_endpoints": [{"headers": {"litellm_user_api_key": "x-company-key"}}]},
        ):
            raised, forwarded = await self._run(
                monkeypatch,
                [
                    (b"x-company-key", f"Bearer {self.VKEY}".encode()),
                    (b"content-type", b"application/json"),
                ],
            )
        assert forwarded is None, "a virtual key in the pass-through key header must be dropped, not forwarded"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_authenticated_authorization_is_stripped_over_a_lower_precedence_pass_through_header(self, monkeypatch):
        with mock.patch.dict(  # test-quality-ok: general_settings is the real proxy config surface for pass_through_endpoints; no injection seam exists on this route
            "litellm.proxy.proxy_server.general_settings",
            {"pass_through_endpoints": [{"headers": {"litellm_user_api_key": "x-company-key"}}]},
        ):
            raised, forwarded = await self._run(
                monkeypatch,
                [
                    (b"authorization", f"Bearer {self.VKEY}".encode()),
                    (b"x-company-key", b"sk-decoy-lower-precedence-value"),
                    (b"x-goog-api-key", b"AIza-real-google-api-key"),
                    (b"content-type", b"application/json"),
                ],
            )
        assert raised is None
        assert forwarded is not None
        assert forwarded.get("x-goog-api-key") == "AIza-real-google-api-key"
        assert "authorization" not in forwarded, "Authorization authenticated (higher precedence) so its key must be stripped"
        assert "x-company-key" not in forwarded
        assert self.VKEY not in " ".join(f"{name}:{value}" for name, value in forwarded.items())

    @pytest.mark.asyncio
    async def test_virtual_key_in_mapped_route_litellm_user_api_key_header_is_stripped(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"litellm_user_api_key", self.VKEY.encode()),
                (b"authorization", b"Bearer ya29.byo-google-oauth"),
                (b"x-goog-api-key", b"AIza-real-google-api-key"),
                (b"content-type", b"application/json"),
            ],
        )
        assert raised is None
        assert forwarded is not None
        assert forwarded.get("x-goog-api-key") == "AIza-real-google-api-key"
        assert forwarded.get("authorization") == "Bearer ya29.byo-google-oauth"
        assert "litellm_user_api_key" not in forwarded
        assert self.VKEY not in " ".join(f"{name}:{value}" for name, value in forwarded.items())

    @pytest.mark.asyncio
    async def test_virtual_key_in_mapped_route_litellm_user_api_key_header_alone_is_rejected(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"litellm_user_api_key", self.VKEY.encode()),
                (b"content-type", b"application/json"),
            ],
        )
        assert forwarded is None, "a virtual key in the mapped-route litellm_user_api_key header must be dropped, not forwarded"
        assert raised is not None and raised.status_code == 401

    GOOGLE_OAUTH_TOKEN = "ya29.byo-google-oauth-token"

    LITELLM_JWT_CLAIMS = MappingProxyType({"sub": "jwt-subject", "iss": "https://idp.example.com"})
    LITELLM_JWT = _unsigned_jwt(LITELLM_JWT_CLAIMS)
    GOOGLE_SERVICE_ACCOUNT_JWT = _unsigned_jwt(
        {
            "sub": "vertex-caller@my-proj.iam.gserviceaccount.com",
            "iss": "vertex-caller@my-proj.iam.gserviceaccount.com",
            "aud": "https://aiplatform.googleapis.com/",
        }
    )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("master_key", "authenticated"),
        [
            pytest.param(
                "sk-master-1234",
                UserAPIKeyAuth(api_key="best-api-key-ever", user_role=LitellmUserRoles.PROXY_ADMIN),
                id="custom-auth-returning-its-own-identifier",
            ),
            pytest.param(
                "sk-master-1234",
                UserAPIKeyAuth(api_key=None, user_id="jwt-subject", jwt_claims=dict(LITELLM_JWT_CLAIMS)),
                id="jwt-auth",
            ),
            pytest.param(None, UserAPIKeyAuth(api_key=GOOGLE_OAUTH_TOKEN), id="no-master-key-echoes-raw-header"),
        ],
    )
    async def test_google_token_in_authorization_is_forwarded_when_auth_did_not_consume_it(
        self, monkeypatch, master_key: str | None, authenticated: UserAPIKeyAuth
    ):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"authorization", f"Bearer {self.GOOGLE_OAUTH_TOKEN}".encode()),
                (b"content-type", b"application/json"),
            ],
            authenticated=authenticated,
            master_key=master_key,
        )
        assert raised is None, f"the caller's own Google token must not be mistaken for a LiteLLM key: {raised}"
        assert forwarded is not None
        assert forwarded.get("authorization") == f"Bearer {self.GOOGLE_OAUTH_TOKEN}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("credential", "authenticated"),
        [
            pytest.param("modified_key", UserAPIKeyAuth(api_key="modified_key"), id="custom-auth-echoing-opaque-credential"),
            pytest.param(
                LITELLM_JWT,
                UserAPIKeyAuth(api_key=LITELLM_JWT, user_id="jwt-subject"),
                id="custom-auth-echoing-jwt",
            ),
            pytest.param(
                LITELLM_JWT,
                UserAPIKeyAuth(api_key=None, user_id="jwt-subject", jwt_claims=dict(LITELLM_JWT_CLAIMS)),
                id="jwt-auth",
            ),
            pytest.param(
                LITELLM_JWT,
                UserAPIKeyAuth(
                    api_key=None,
                    user_id="jwt-subject",
                    jwt_claims={
                        **LITELLM_JWT_CLAIMS,
                        JWTHandler.LITELLM_JWT_ISSUER_CLAIM: "https://idp.example.com",
                        JWTHandler.LITELLM_USER_ID_CLAIM: "jwt-subject",
                    },
                ),
                id="multi-issuer-jwt-auth-normalized-claims",
            ),
        ],
    )
    async def test_non_sk_litellm_credential_that_authenticated_is_rejected_not_forwarded(
        self, monkeypatch, credential: str, authenticated: UserAPIKeyAuth
    ):
        raised, forwarded = await self._run(
            monkeypatch,
            [(b"authorization", f"Bearer {credential}".encode()), (b"content-type", b"application/json")],
            authenticated=authenticated,
        )
        assert forwarded is None, "the credential that authenticated the caller must never reach the upstream forwarder"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_authenticated_caller_keeps_a_different_byo_google_jwt(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"x-litellm-api-key", self.LITELLM_JWT.encode()),
                (b"authorization", f"Bearer {self.GOOGLE_SERVICE_ACCOUNT_JWT}".encode()),
                (b"content-type", b"application/json"),
            ],
            authenticated=UserAPIKeyAuth(api_key=None, user_id="jwt-subject", jwt_claims=dict(self.LITELLM_JWT_CLAIMS)),
        )
        assert raised is None, f"a Google JWT that is not the one that authenticated must keep flowing: {raised}"
        assert forwarded is not None
        assert forwarded.get("authorization") == f"Bearer {self.GOOGLE_SERVICE_ACCOUNT_JWT}"
        assert "x-litellm-api-key" not in forwarded

    @pytest.mark.asyncio
    async def test_master_key_in_authorization_alone_is_rejected(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [(b"authorization", b"Bearer sk-master-1234"), (b"content-type", b"application/json")],
            authenticated=UserAPIKeyAuth(api_key=LITELLM_PROXY_MASTER_KEY_ALIAS, user_role=LitellmUserRoles.PROXY_ADMIN),
        )
        assert forwarded is None, "the master key must never reach the upstream forwarder"
        assert raised is not None and raised.status_code == 401

    @pytest.mark.asyncio
    async def test_master_key_is_stripped_and_byo_x_goog_api_key_forwards(self, monkeypatch):
        raised, forwarded = await self._run(
            monkeypatch,
            [
                (b"authorization", b"Bearer sk-master-1234"),
                (b"x-goog-api-key", b"AIza-real-google-api-key"),
                (b"content-type", b"application/json"),
            ],
            authenticated=UserAPIKeyAuth(api_key=LITELLM_PROXY_MASTER_KEY_ALIAS, user_role=LitellmUserRoles.PROXY_ADMIN),
        )
        assert raised is None
        assert forwarded is not None
        assert forwarded.get("x-goog-api-key") == "AIza-real-google-api-key"
        assert "authorization" not in forwarded
        assert "sk-master-1234" not in " ".join(f"{name}:{value}" for name, value in forwarded.items())


class TestGetAzureAISearchIndexFromEndpoint:
    """The operable index is only the segment right after ``indexes``.

    A doc-write path ends in ``.../docs/index``; the trailing ``index`` must not
    be mistaken for the target, otherwise a caller could be authorized on one
    index while Azure applies the write to another.
    """

    @pytest.mark.parametrize(
        "endpoint, expected",
        [
            ("indexes/my-index/docs/index", "my-index"),
            ("indexes/my-index/docs/search", "my-index"),
            ("indexes/my-index", "my-index"),
            ("indexes/my-index?api-version=2024-07-01", "my-index"),
            ("/indexes/my-index/docs/index", "my-index"),
            ("indexes/victim/docs/index", "victim"),
            ("openai/deployments/gpt-4o/chat/completions", None),
            ("indexes", None),
            ("indexes/", None),
        ],
    )
    def test_extracts_positional_index_only(self, endpoint, expected):
        assert get_azure_ai_search_index_from_endpoint(endpoint) == expected


class TestAzureProxyRouteCrossIndexAuthorization:
    """Regression tests: the passthrough must authorize the index that the request
    actually targets (the ``/indexes/{name}`` segment), never a different segment
    that merely happens to match a managed index the caller can access.
    """

    def _request(self, method: str, path: str) -> MagicMock:
        request = MagicMock(spec=Request)
        request.method = method
        request.headers = {"content-type": "application/json"}
        request.url = MagicMock()
        request.url.path = path
        return request

    @pytest.mark.asyncio
    async def test_authorizes_the_targeted_index(self):
        index_object = MagicMock()
        index_object.litellm_params.vector_store_name = "my-store"
        vector_store = {"litellm_params": {"api_base": "https://svc.search.windows.net"}}

        with (
            patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_passthrough_request_using_router_model",
                return_value=False,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
            ) as mock_get_config,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
            ) as mock_is_allowed,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.assert_user_can_access_vector_store",
                new=AsyncMock(),
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.BaseOpenAIPassThroughHandler._base_openai_pass_through_handler",
                new=AsyncMock(return_value=Response()),
            ),
            patch.object(litellm, "vector_store_index_registry") as mock_index_registry,
            patch.object(litellm, "vector_store_registry") as mock_vector_registry,
        ):
            mock_get_config.return_value.get_auth_credentials.return_value = {"headers": {"api-key": "k"}}
            mock_index_registry.is_vector_store_index.side_effect = lambda vector_store_index_name: (
                vector_store_index_name == "my-index"
            )
            mock_index_registry.get_vector_store_index_by_name.return_value = index_object
            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = vector_store

            await azure_proxy_route(
                endpoint="indexes/my-index/docs/index",
                request=self._request("POST", "/azure_ai/indexes/my-index/docs/index"),
                fastapi_response=MagicMock(spec=Response),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            )

            mock_is_allowed.assert_called_once()
            assert mock_is_allowed.call_args.kwargs["index_name"] == "my-index"
            mock_index_registry.get_vector_store_index_by_name.assert_called_once_with(
                vector_store_index_name="my-index"
            )

    @pytest.mark.asyncio
    async def test_trailing_index_segment_does_not_authorize_a_different_index(self):
        with (
            patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_passthrough_request_using_router_model",
                return_value=False,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
            ) as mock_is_allowed,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_secret_str",
                return_value="https://azure-openai.example.com",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value="azure-key",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.BaseOpenAIPassThroughHandler._base_openai_pass_through_handler",
                new=AsyncMock(return_value=Response()),
            ) as mock_handler,
            patch.object(litellm, "vector_store_index_registry") as mock_index_registry,
        ):
            mock_index_registry.is_vector_store_index.side_effect = lambda vector_store_index_name: (
                vector_store_index_name == "index"
            )

            await azure_proxy_route(
                endpoint="indexes/victim/docs/index",
                request=self._request("POST", "/azure_ai/indexes/victim/docs/index"),
                fastapi_response=MagicMock(spec=Response),
                user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
            )

            mock_is_allowed.assert_not_called()
            mock_handler.assert_awaited_once()
            assert mock_handler.await_args.kwargs["custom_llm_provider"] == litellm.LlmProviders.AZURE


class TestAzureProxyRouteServiceLevelIndexCreate:
    """``POST /indexes`` carries no index name, so the managed-index branch cannot
    claim it and it would otherwise reach the generic Azure passthrough on the
    proxy's own credential. The admin-only index management guard has to be
    enforced on the route itself, not just on the permission gate the route skips.
    """

    def _request(self, method: str, path: str) -> MagicMock:
        request = MagicMock(spec=Request)
        request.method = method
        request.headers = {"content-type": "application/json"}
        request.url = MagicMock()
        request.url.path = path
        return request

    @pytest.mark.parametrize(
        "method, endpoint, expected",
        [
            ("POST", "indexes", True),
            ("POST", "indexes?api-version=2024-07-01", True),
            ("POST", "/indexes/", True),
            ("POST", "indexes/my-index", False),
            ("POST", "indexes/my-index/docs/index", False),
            ("GET", "indexes", False),
            ("POST", "openai/deployments/gpt-4o/chat/completions", False),
        ],
    )
    def test_recognizes_service_level_create(self, method, endpoint, expected):
        assert is_azure_ai_search_service_level_index_create(method=method, endpoint=endpoint) is expected

    @pytest.mark.asyncio
    async def test_non_admin_cannot_create_an_index(self):
        with (
            patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_secret_str",
                return_value="https://svc.search.windows.net",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.BaseOpenAIPassThroughHandler._base_openai_pass_through_handler",
                new=AsyncMock(return_value=Response()),
            ) as mock_handler,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await azure_proxy_route(
                    endpoint="indexes?api-version=2024-07-01",
                    request=self._request("POST", "/azure_ai/indexes"),
                    fastapi_response=MagicMock(spec=Response),
                    user_api_key_dict=UserAPIKeyAuth(
                        token="sk-team-token",
                        user_role=LitellmUserRoles.INTERNAL_USER,
                    ),
                )

            assert exc_info.value.status_code == 403
            assert "Only proxy admins can create" in exc_info.value.detail
            mock_handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_admin_can_still_create_an_index(self):
        with (
            patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_secret_str",
                return_value="https://svc.search.windows.net",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value="azure-key",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.BaseOpenAIPassThroughHandler._base_openai_pass_through_handler",
                new=AsyncMock(return_value=Response()),
            ) as mock_handler,
        ):
            await azure_proxy_route(
                endpoint="indexes?api-version=2024-07-01",
                request=self._request("POST", "/azure_ai/indexes"),
                fastapi_response=MagicMock(spec=Response),
                user_api_key_dict=UserAPIKeyAuth(
                    token="sk-admin-token",
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                ),
            )

            mock_handler.assert_awaited_once()


class TestComprehendMedicalProxyRoute:
    def _mock_request(self, body: object) -> Mock:
        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.json = AsyncMock(return_value=body)
        return mock_request

    @pytest.mark.asyncio
    async def test_signs_and_forwards_detect_entities_v2(self):
        from botocore.credentials import Credentials

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            comprehend_medical_proxy_route,
        )
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY,
            LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY,
        )

        request_body = {"Text": "Patient was prescribed 40mg atorvastatin daily."}
        mock_request = self._mock_request(request_body)
        mock_endpoint_func = AsyncMock(return_value={"Entities": []})

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_secret_str",
                side_effect=lambda secret_name: "us-east-1" if secret_name == "AWS_REGION_NAME" else None,
            ),
            patch(
                "litellm.llms.bedrock.base_aws_llm.BaseAWSLLM.get_credentials",
                return_value=Credentials("test-access-key", "test-secret-key"),
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ) as mock_create_route,
        ):
            result = await comprehend_medical_proxy_route(
                operation="DetectEntitiesV2",
                request=mock_request,
                fastapi_response=Mock(),
                user_api_key_dict=Mock(),
            )

        assert result == {"Entities": []}
        call_kwargs = mock_create_route.call_args.kwargs
        assert call_kwargs["target"] == "https://comprehendmedical.us-east-1.amazonaws.com/"
        assert call_kwargs["custom_llm_provider"] == "comprehendmedical"
        assert "_forward_headers" not in call_kwargs
        signed_headers = dict(call_kwargs["custom_headers"])
        assert signed_headers["X-Amz-Target"] == "ComprehendMedical_20181030.DetectEntitiesV2"
        assert signed_headers["Content-Type"] == "application/x-amz-json-1.1"
        assert signed_headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "/comprehendmedical/aws4_request" in signed_headers["Authorization"]
        assert getattr(mock_request.state, LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY) == request_body
        assert json.loads(getattr(mock_request.state, LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY)) == request_body
        mock_endpoint_func.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "operation",
        [
            "Detect-Entities",
            "Detect/../secrets",
            "",
            "a" * 200,
            "DetectEntities",
            "StartEntitiesDetectionV2Job",
        ],
    )
    async def test_rejects_unsupported_operations(self, operation):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            comprehend_medical_proxy_route,
        )

        with pytest.raises(HTTPException) as exc_info:
            await comprehend_medical_proxy_route(
                operation=operation,
                request=self._mock_request({"Text": "hi"}),
                fastapi_response=Mock(),
                user_api_key_dict=Mock(),
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [{"Text": "hi", "stream": True}, {"Text": "hi", "stream": False}, ["Text"]])
    async def test_rejects_stream_key_and_non_object_bodies(self, body):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            comprehend_medical_proxy_route,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_secret_str",
            return_value="us-east-1",
        ):
            with pytest.raises(HTTPException) as exc_info:
                await comprehend_medical_proxy_route(
                    operation="DetectEntitiesV2",
                    request=self._mock_request(body),
                    fastapi_response=Mock(),
                    user_api_key_dict=Mock(),
                )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_region_returns_400(self):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            comprehend_medical_proxy_route,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await comprehend_medical_proxy_route(
                    operation="DetectPHI",
                    request=self._mock_request({"Text": "hi"}),
                    fastapi_response=Mock(),
                    user_api_key_dict=Mock(),
                )
        assert exc_info.value.status_code == 400

    def test_comprehendmedical_is_a_mapped_pass_through_route(self):
        from litellm.proxy._types import LiteLLMRoutes

        assert "/comprehendmedical" in LiteLLMRoutes.mapped_pass_through_routes.value

    @pytest.mark.asyncio
    async def test_sdk_route_reads_operation_from_x_amz_target(self):
        from botocore.credentials import Credentials

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            comprehend_medical_sdk_proxy_route,
        )

        mock_request = self._mock_request({"Text": "hi"})
        mock_request.headers = {"x-amz-target": "ComprehendMedical_20181030.DetectPHI"}
        mock_endpoint_func = AsyncMock(return_value={"Entities": []})

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_secret_str",
                side_effect=lambda secret_name: "us-east-1" if secret_name == "AWS_REGION_NAME" else None,
            ),
            patch(
                "litellm.llms.bedrock.base_aws_llm.BaseAWSLLM.get_credentials",
                return_value=Credentials("test-access-key", "test-secret-key"),
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route",
                return_value=mock_endpoint_func,
            ) as mock_create_route,
        ):
            result = await comprehend_medical_sdk_proxy_route(
                request=mock_request,
                fastapi_response=Mock(),
                user_api_key_dict=Mock(),
            )

        assert result == {"Entities": []}
        signed_headers = dict(mock_create_route.call_args.kwargs["custom_headers"])
        assert signed_headers["X-Amz-Target"] == "ComprehendMedical_20181030.DetectPHI"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "target_header",
        ["", "ComprehendMedical_20181030", "WrongService.DetectPHI", "ComprehendMedical_20181030."],
    )
    async def test_sdk_route_rejects_bad_x_amz_target(self, target_header):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            comprehend_medical_sdk_proxy_route,
        )

        mock_request = self._mock_request({"Text": "hi"})
        mock_request.headers = {"x-amz-target": target_header}

        with pytest.raises(HTTPException) as exc_info:
            await comprehend_medical_sdk_proxy_route(
                request=mock_request,
                fastapi_response=Mock(),
                user_api_key_dict=Mock(),
            )
        assert exc_info.value.status_code == 400


LIVE_RESOURCE_PATH = "projects/proj-db/locations/global/publishers/google/models/gemini-live-2.5-flash"


class TestVertexAILiveWebsocketPassthrough:
    def _websocket(self):
        from starlette.websockets import WebSocketState

        websocket = MagicMock()
        websocket.accept = AsyncMock()
        websocket.close = AsyncMock()
        websocket.headers = {}
        websocket.client_state = WebSocketState.CONNECTED
        return websocket

    def _clear_vertex_env(self, monkeypatch):
        monkeypatch.delenv("DEFAULT_VERTEXAI_PROJECT", raising=False)
        monkeypatch.delenv("DEFAULT_VERTEXAI_LOCATION", raising=False)
        monkeypatch.delenv("DEFAULT_GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    @pytest.mark.asyncio
    async def test_uses_db_deployment_credentials_without_query_params(self, monkeypatch):
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as passthrough_module,
        )

        llm_router = litellm.Router(
            model_list=[
                {
                    "model_name": "gemini-live",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-live-2.5-flash",
                        "use_in_pass_through": True,
                        "vertex_project": "proj-db",
                        "vertex_location": "global",
                        "vertex_credentials": '{"type": "service_account"}',
                    },
                }
            ]
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        monkeypatch.setattr(
            passthrough_module.passthrough_endpoint_router, "default_vertex_config", None
        )
        self._clear_vertex_env(monkeypatch)
        websocket = self._websocket()
        ensure_token = AsyncMock(return_value=("token-abc", "proj-db"))
        ws_passthrough = AsyncMock()

        with (
            patch.object(passthrough_module.vertex_llm_base, "_ensure_access_token_async", ensure_token),
            patch.object(passthrough_module, "websocket_passthrough_request", ws_passthrough),
        ):
            await passthrough_module.vertex_ai_live_websocket_passthrough(
                websocket=websocket,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        ensure_token.assert_awaited_once_with(
            credentials='{"type": "service_account"}',
            project_id="proj-db",
            custom_llm_provider="vertex_ai_beta",
        )
        passthrough_kwargs = ws_passthrough.await_args.kwargs
        assert passthrough_kwargs["target"] == (
            "wss://aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
        )
        assert passthrough_kwargs["custom_headers"]["Authorization"] == "Bearer token-abc"
        rewriter = passthrough_kwargs["setup_model_rewriter"]
        assert rewriter("gemini-live") == (
            "projects/proj-db/locations/global/publishers/google/models/gemini-live-2.5-flash"
        )
        websocket.close.assert_not_awaited()

    @pytest.mark.parametrize(
        "setup_model, expected",
        [
            ("gemini-live-2.5-flash", LIVE_RESOURCE_PATH),
            ("models/gemini-live-2.5-flash", LIVE_RESOURCE_PATH),
            ("vertex_ai/gemini-live-2.5-flash", LIVE_RESOURCE_PATH),
            ("gemini-live", LIVE_RESOURCE_PATH),
            ("models/gemini-live", LIVE_RESOURCE_PATH),
            (
                "publishers/meta/models/llama-3.3-70b-instruct-maas",
                "projects/proj-db/locations/global/publishers/meta/models/llama-3.3-70b-instruct-maas",
            ),
            (
                "projects/other/locations/us-central1/publishers/google/models/gemini-2.0-flash",
                "projects/other/locations/us-central1/publishers/google/models/gemini-2.0-flash",
            ),
        ],
    )
    def test_setup_model_rewriter_normalises_the_forms_clients_send(self, setup_model, expected):
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as passthrough_module,
        )

        llm_router = litellm.Router(
            model_list=[
                {
                    "model_name": "gemini-live",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-live-2.5-flash",
                        "use_in_pass_through": True,
                        "vertex_project": "proj-db",
                        "vertex_location": "global",
                    },
                }
            ]
        )

        rewriter = passthrough_module._build_vertex_live_setup_model_rewriter(
            vertex_project="proj-db",
            vertex_location="global",
            llm_router=llm_router,
        )

        assert rewriter is not None
        assert rewriter(setup_model) == expected

    @pytest.mark.asyncio
    async def test_default_vertex_config_outranks_db_deployment(self, monkeypatch):
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as passthrough_module,
        )
        from litellm.types.passthrough_endpoints.vertex_ai import (
            VertexPassThroughCredentials,
        )

        llm_router = litellm.Router(
            model_list=[
                {
                    "model_name": "gemini-live",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-live-2.5-flash",
                        "use_in_pass_through": True,
                        "vertex_project": "proj-db",
                        "vertex_location": "global",
                        "vertex_credentials": '{"type": "db_account"}',
                    },
                }
            ]
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        monkeypatch.setattr(
            passthrough_module.passthrough_endpoint_router,
            "default_vertex_config",
            VertexPassThroughCredentials(
                vertex_project="proj-env",
                vertex_location="global",
                vertex_credentials='{"type": "env_account"}',
            ),
        )
        self._clear_vertex_env(monkeypatch)
        websocket = self._websocket()
        ensure_token = AsyncMock(return_value=("token-abc", "proj-env"))
        ws_passthrough = AsyncMock()

        with (
            patch.object(passthrough_module.vertex_llm_base, "_ensure_access_token_async", ensure_token),
            patch.object(passthrough_module, "websocket_passthrough_request", ws_passthrough),
        ):
            await passthrough_module.vertex_ai_live_websocket_passthrough(
                websocket=websocket,
                model="gemini-live",
                user_api_key_dict=UserAPIKeyAuth(),
            )

        ensure_token.assert_awaited_once_with(
            credentials='{"type": "env_account"}',
            project_id="proj-env",
            custom_llm_provider="vertex_ai_beta",
        )
        rewriter = ws_passthrough.await_args.kwargs["setup_model_rewriter"]
        assert rewriter("gemini-live") == (
            "projects/proj-env/locations/global/publishers/google/models/gemini-live-2.5-flash"
        )

    @pytest.mark.asyncio
    async def test_credential_failure_close_names_configuration_options(self, monkeypatch):
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as passthrough_module,
        )

        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
        monkeypatch.setattr(
            passthrough_module.passthrough_endpoint_router, "default_vertex_config", None
        )
        self._clear_vertex_env(monkeypatch)
        websocket = self._websocket()
        ensure_token = AsyncMock(side_effect=Exception("Unable to find your credentials"))

        with (
            patch.object(passthrough_module.vertex_llm_base, "_ensure_access_token_async", ensure_token),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging,
        ):
            mock_proxy_logging.post_call_failure_hook = AsyncMock()
            await passthrough_module.vertex_ai_live_websocket_passthrough(
                websocket=websocket,
                user_api_key_dict=UserAPIKeyAuth(),
            )

        close_kwargs = websocket.close.await_args.kwargs
        assert close_kwargs["code"] == 1011
        assert "use_in_pass_through" in close_kwargs["reason"]
        assert "default_vertex_config" in close_kwargs["reason"]
        assert len(close_kwargs["reason"].encode("utf-8")) <= 123


class TestPassthroughRouterModelBudgetReservation:
    """
    Router-model passthrough on /vllm and /azure must thread the calling key's
    metadata into ``allm_passthrough_route``. Without ``user_api_key`` the spend
    is attributed to nobody, and without ``user_api_key_budget_reservation`` the
    pre-call reservation is never released, so the shared spend counter drifts up
    until the key falsely trips a 429 BudgetExceededError (LIT-5470).
    """

    def _key_with_reservation(self) -> UserAPIKeyAuth:
        reservation = {
            "reserved_cost": 0.5,
            "entries": [{"counter_key": "spend:key:hashed-token", "reserved_cost": 0.5}],
        }
        return UserAPIKeyAuth(
            api_key="hashed-token",
            user_id="u1",
            team_id="t1",
            budget_reservation=reservation,
            agent_id="agent-xyz",
            end_user_max_budget=42.0,
        )

    def _request(self) -> MagicMock:
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.query_params = {}
        return request

    def _install_recording_router(self, monkeypatch, body: dict) -> list[dict]:
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as ep
        import litellm.proxy.proxy_server as proxy_server

        captured: list[dict] = []

        class RecordingRouter:
            async def allm_passthrough_route(self, **kwargs):
                captured.append(kwargs)
                return httpx.Response(200, json={"ok": True})

        async def fake_get_request_body(_request):
            return body

        monkeypatch.setattr(proxy_server, "llm_router", RecordingRouter())
        monkeypatch.setattr(ep, "get_request_body", fake_get_request_body)
        monkeypatch.setattr(ep, "is_passthrough_request_using_router_model", lambda *a, **k: True)
        return captured

    def _assert_metadata_carries_attribution(self, captured: list[dict], user_api_key_dict: UserAPIKeyAuth) -> None:
        assert len(captured) == 1, "the router-model branch must dispatch exactly once"
        assert captured[0].get("metadata") is None, (
            "attribution must ride the litellm_metadata bucket the router canonicalizes on; "
            "the plain metadata bucket is dropped for every non-user_api_key field"
        )
        litellm_metadata = captured[0]["litellm_metadata"]
        assert litellm_metadata["user_api_key"] == user_api_key_dict.api_key
        assert litellm_metadata["user_api_key_budget_reservation"] is user_api_key_dict.budget_reservation
        assert litellm_metadata["user_api_key_user_id"] == user_api_key_dict.user_id
        assert litellm_metadata["user_api_key_team_id"] == user_api_key_dict.team_id
        assert litellm_metadata["agent_id"] == user_api_key_dict.agent_id
        assert litellm_metadata["user_api_end_user_max_budget"] == user_api_key_dict.end_user_max_budget

    @pytest.mark.asyncio
    async def test_vllm_router_model_threads_key_metadata(self, monkeypatch):
        user_api_key_dict = self._key_with_reservation()
        captured = self._install_recording_router(monkeypatch, {"model": "router-model", "stream": False})

        await vllm_proxy_route(
            endpoint="/chat/completions",
            request=self._request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=user_api_key_dict,
        )

        self._assert_metadata_carries_attribution(captured, user_api_key_dict)

    @pytest.mark.asyncio
    async def test_azure_router_model_threads_key_metadata(self, monkeypatch):
        user_api_key_dict = self._key_with_reservation()
        captured = self._install_recording_router(monkeypatch, {"model": "gpt-5", "stream": False})

        await azure_proxy_route(
            endpoint="openai/deployments/gpt-5/chat/completions",
            request=self._request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=user_api_key_dict,
        )

        self._assert_metadata_carries_attribution(captured, user_api_key_dict)
