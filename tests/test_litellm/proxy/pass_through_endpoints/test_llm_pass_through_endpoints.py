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
    llm_passthrough_factory_proxy_route,
    vllm_proxy_route,
    vertex_discovery_proxy_route,
    vertex_proxy_route,
    bedrock_llm_proxy_route,
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
                target=f"https://aiplatform.googleapis.com/v1/projects/{test_project}/locations/{test_location}/publishers/google/models/gemini-1.5-flash:generateContent",
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
                            "endOffsetSec": 5
                        }
                    ]
                }
            ]
        }

        # Create mock httpx.Response
        mock_httpx_response = Mock()
        mock_httpx_response.json.return_value = multimodal_response_data
        mock_httpx_response.status_code = 200

        # Create mock logging object
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        mock_logging_obj.model_call_details = {}

        # Test URL with multimodal embedding model
        url_route = "/v1/projects/test-project/locations/us-central1/publishers/google/models/multimodalembedding@001:predict"
        
        start_time = datetime.datetime.now()
        end_time = datetime.datetime.now()
        
        with patch("litellm.llms.vertex_ai.multimodal_embeddings.transformation.VertexAIMultimodalEmbeddingConfig") as mock_multimodal_config:
            # Mock the multimodal config instance and its methods
            mock_config_instance = Mock()
            mock_multimodal_config.return_value = mock_config_instance
            
            # Create a mock embedding response that would be returned by the transformation
            from litellm.types.utils import Embedding, EmbeddingResponse, Usage
            mock_embedding_response = EmbeddingResponse(
                object="list",
                data=[
                    Embedding(embedding=[0.1, 0.2, 0.3, 0.4, 0.5], index=0, object="embedding"),
                    Embedding(embedding=[0.6, 0.7, 0.8, 0.9, 1.0], index=1, object="embedding"),
                ],
                model="multimodalembedding@001",
                usage=Usage(prompt_tokens=0, total_tokens=0, completion_tokens=0)
            )
            mock_config_instance.transform_embedding_response.return_value = mock_embedding_response
            
            # Call the handler
            result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                httpx_response=mock_httpx_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="test-result",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False
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
            "predictions": [
                {
                    "textEmbedding": [0.1, 0.2, 0.3]
                }
            ]
        }
        assert VertexPassthroughLoggingHandler._is_multimodal_embedding_response(response_with_text_embedding) is True

        # Test case 2: Response with imageEmbedding should be detected as multimodal
        response_with_image_embedding = {
            "predictions": [
                {
                    "imageEmbedding": [0.4, 0.5, 0.6]
                }
            ]
        }
        assert VertexPassthroughLoggingHandler._is_multimodal_embedding_response(response_with_image_embedding) is True

        # Test case 3: Response with videoEmbeddings should be detected as multimodal
        response_with_video_embeddings = {
            "predictions": [
                {
                    "videoEmbeddings": [
                        {
                            "embedding": [0.7, 0.8, 0.9],
                            "startOffsetSec": 0,
                            "endOffsetSec": 5
                        }
                    ]
                }
            ]
        }
        assert VertexPassthroughLoggingHandler._is_multimodal_embedding_response(response_with_video_embeddings) is True

        # Test case 4: Regular text embedding response should NOT be detected as multimodal
        regular_embedding_response = {
            "predictions": [
                {
                    "embeddings": {
                        "values": [0.1, 0.2, 0.3]
                    }
                }
            ]
        }
        assert VertexPassthroughLoggingHandler._is_multimodal_embedding_response(regular_embedding_response) is False

        # Test case 5: Non-embedding response should NOT be detected as multimodal
        non_embedding_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello world"}]
                    }
                }
            ]
        }
        assert VertexPassthroughLoggingHandler._is_multimodal_embedding_response(non_embedding_response) is False

        # Test case 6: Empty response should NOT be detected as multimodal
        empty_response = {}
        assert VertexPassthroughLoggingHandler._is_multimodal_embedding_response(empty_response) is False


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

class TestBedrockLLMProxyRoute:
    @pytest.mark.asyncio
    async def test_bedrock_llm_proxy_route_application_inference_profile(self):
        mock_request = Mock()
        mock_request.method = "POST"
        mock_response = Mock()
        mock_user_api_key_dict = Mock()
        mock_request_body = {"messages": [{"role": "user", "content": "test"}]}
        mock_processor = Mock()
        mock_processor.base_passthrough_process_llm_request = AsyncMock(return_value="success")
        
        with patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._read_request_body", return_value=mock_request_body), \
             patch("litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing", return_value=mock_processor):
            
            # Test application-inference-profile endpoint
            endpoint = "model/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/r742sbn2zckd/converse"
            
            result = await bedrock_llm_proxy_route(
                endpoint=endpoint,
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )
            
            mock_processor.base_passthrough_process_llm_request.assert_called_once()
            call_kwargs = mock_processor.base_passthrough_process_llm_request.call_args.kwargs
            
            # For application-inference-profile, model should be "arn:aws:bedrock:us-east-1:026090525607:application-inference-profile/r742sbn2zckd"
            assert call_kwargs["model"] == "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/r742sbn2zckd"
            assert result == "success"
            
    @pytest.mark.asyncio
    async def test_bedrock_llm_proxy_route_regular_model(self):
        mock_request = Mock()
        mock_request.method = "POST"
        mock_response = Mock()
        mock_user_api_key_dict = Mock()
        mock_request_body = {"messages": [{"role": "user", "content": "test"}]}
        mock_processor = Mock()
        mock_processor.base_passthrough_process_llm_request = AsyncMock(return_value="success")
        
        with patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._read_request_body", return_value=mock_request_body), \
             patch("litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing", return_value=mock_processor):
            
            # Test regular model endpoint
            endpoint = "model/anthropic.claude-3-sonnet-20240229-v1:0/converse"
            
            result = await bedrock_llm_proxy_route(
                endpoint=endpoint,
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )
            mock_processor.base_passthrough_process_llm_request.assert_called_once()
            call_kwargs = mock_processor.base_passthrough_process_llm_request.call_args.kwargs
            
            # For regular models, model should be just the model ID
            assert call_kwargs["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert result == "success"


class TestLLMPassthroughFactoryProxyRoute:
    @pytest.mark.asyncio
    async def test_llm_passthrough_factory_proxy_route_success(self):
        from litellm.types.utils import LlmProviders
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.json = AsyncMock(return_value={"stream": False})
        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.utils.ProviderConfigManager.get_provider_model_info"
        ) as mock_get_provider, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
        ) as mock_get_creds, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
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
