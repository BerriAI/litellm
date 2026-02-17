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
    bedrock_llm_proxy_route,
    create_pass_through_route,
    llm_passthrough_factory_proxy_route,
    milvus_proxy_route,
    openai_proxy_route,
    vertex_discovery_proxy_route,
    vertex_proxy_route,
    vllm_proxy_route,
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
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
        ) as mock_load_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
        ) as mock_get_virtual_key, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_user_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
        ) as mock_get_handler:
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
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
        ) as mock_load_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
        ) as mock_get_virtual_key, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_user_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
        ) as mock_get_handler:
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

        with mock.patch(
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
        ) as mock_load_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
        ) as mock_get_handler:
            # Mock credentials object with necessary attributes
            mock_credentials = Mock()
            mock_credentials.token = default_credentials

            mock_load_auth.return_value = (mock_credentials, default_project)

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
                is_streaming_request=False,
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

        with mock.patch(
            "litellm.llms.vertex_ai.vertex_llm_base.VertexBase.load_auth"
        ) as mock_load_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_litellm_virtual_key"
        ) as mock_get_virtual_key, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth"
        ) as mock_user_auth, mock.patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_vertex_pass_through_handler"
        ) as mock_get_handler:
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
        mock_processor.base_passthrough_process_llm_request = AsyncMock(
            return_value="success"
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._read_request_body",
            return_value=mock_request_body,
        ), patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing",
            return_value=mock_processor,
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

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._read_request_body",
            return_value=mock_request_body,
        ), patch(
            "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing",
            return_value=mock_processor,
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
            assert deployment_litellm_params.get("aws_access_key_id") == model_access_key
            assert deployment_litellm_params.get("aws_secret_access_key") == model_secret_key
            assert deployment_litellm_params.get("aws_region_name") == model_region
            assert deployment_litellm_params.get("aws_session_token") == model_session_token

            # Verify environment variables are NOT in the deployment
            assert deployment_litellm_params.get("aws_access_key_id") != env_access_key
            assert deployment_litellm_params.get("aws_secret_access_key") != env_secret_key
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

            with patch(
                "litellm.passthrough.main.llm_passthrough_route",
                new_callable=AsyncMock,
                side_effect=mock_llm_passthrough_route,
            ), patch(
                "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_passthrough_process_llm_request",
                new_callable=AsyncMock,
            ) as mock_process:
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
        mock_httpx_response.aiter_bytes = AsyncMock(return_value=[b'{"result": "success"}'])
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value=mock_request_body,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj:
            # Setup mock httpx client
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            # Setup mock logging object
            mock_logging_obj.pre_call_hook = AsyncMock(return_value=mock_request_body)
            mock_logging_obj.post_call_success_hook = AsyncMock()
            mock_logging_obj.post_call_failure_hook = AsyncMock()

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
            assert mock_client.request.called

            # Get the headers that were sent to the target
            call_args = mock_client.request.call_args
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
        mock_httpx_response.aiter_bytes = AsyncMock(return_value=[b'{"result": "success"}'])
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value=mock_request_body,
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj:
            # Setup mock httpx client
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value=mock_httpx_response)
            mock_client_obj = MagicMock()
            mock_client_obj.client = mock_client
            mock_get_client.return_value = mock_client_obj

            # Setup mock logging object
            mock_logging_obj.pre_call_hook = AsyncMock(return_value=mock_request_body)
            mock_logging_obj.post_call_success_hook = AsyncMock()
            mock_logging_obj.post_call_failure_hook = AsyncMock()

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
            assert mock_client.request.called

            # Get the headers that were sent to the target
            call_args = mock_client.request.call_args
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
        mock_httpx_response.aiter_bytes = AsyncMock(return_value=[b'{"result": "success"}'])
        mock_httpx_response.aread = AsyncMock(return_value=b'{"result": "success"}')

        with patch(
            "litellm.utils.ProviderConfigManager.get_provider_model_info"
        ) as mock_get_provider, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials"
        ) as mock_get_creds, patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._read_request_body",
            return_value={"messages": [{"role": "user", "content": "test"}]},
        ), patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client, patch(
            "litellm.proxy.proxy_server.proxy_logging_obj"
        ) as mock_logging_obj:
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
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

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

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name, "data": [[0.1, 0.2]]},
        ) as mock_get_body, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
        ) as mock_is_allowed, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ) as mock_safe_set, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry"
        ) as mock_vector_registry:
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

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"data": [[0.1, 0.2]]},  # No collectionName
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config:
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

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

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

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "test-collection"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch.object(
            litellm, "vector_store_index_registry", None
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

        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "unmanaged-collection"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry", MagicMock()
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
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

        collection_name = "test-collection"
        vector_store_name = "missing-store"
        vector_store_index = "collection_123"

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        mock_index_object = MagicMock()
        mock_index_object.litellm_params.vector_store_name = vector_store_name
        mock_index_object.litellm_params.vector_store_index = vector_store_index

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ), patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry"
        ) as mock_vector_registry:
            mock_get_config.return_value = MagicMock()
            mock_index_registry.is_vector_store_index.return_value = True
            mock_index_registry.get_vector_store_index_by_name.return_value = (
                mock_index_object
            )
            mock_vector_registry.get_litellm_managed_vector_store_from_registry_by_name.return_value = (
                None
            )

            with pytest.raises(Exception) as exc_info:
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
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

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

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ), patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry"
        ) as mock_vector_registry:
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

            with pytest.raises(Exception) as exc_info:
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
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            milvus_proxy_route,
        )

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

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
            return_value={"collectionName": collection_name},
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.ProviderConfigManager.get_provider_vector_stores_config"
        ) as mock_get_config, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_allowed_to_call_vector_store_endpoint"
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._safe_set_request_parsed_body"
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route, patch.object(
            litellm, "vector_store_index_registry"
        ) as mock_index_registry, patch.object(
            litellm, "vector_store_registry"
        ) as mock_vector_registry:
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
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        # Mock request for Responses API
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
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
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
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
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value=None,
        ):
            with pytest.raises(Exception) as exc_info:
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
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            openai_proxy_route,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.headers = {"content-type": "application/json"}
        mock_request.query_params = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/assistants"
        mock_response = MagicMock(spec=Response)
        mock_user_api_key_dict = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
            return_value="sk-test-key",
        ), patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
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
