# Test the following scenarios:
# 1. Generate a Key, and use it to make a call


import json
import logging
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from fastapi import HTTPException, Request

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.proxy._types import ProxyException, TokenCountRequest
from litellm.proxy.anthropic_endpoints.endpoints import (
    count_tokens as anthropic_count_tokens,
)
from litellm.proxy.proxy_server import token_counter
from litellm.types.utils import TokenCountResponse

verbose_proxy_logger.setLevel(level=logging.DEBUG)


def get_vertex_ai_creds_json() -> dict:
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"
    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    return service_account_key_data


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


@pytest.mark.asyncio
async def test_vLLM_token_counting():
    """
    Test Token counter for vLLM models
    - User passes model="special-alias"
    - token_counter should infer that special_alias -> maps to wolfram/miquliz-120b-v2.0
    -> token counter should use hugging face tokenizer
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "special-alias",
                "litellm_params": {
                    "model": "openai/wolfram/miquliz-120b-v2.0",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(
            model="special-alias",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert (
        response.tokenizer_type == "openai_tokenizer"
    )  # SHOULD use the default tokenizer
    assert response.model_used == "wolfram/miquliz-120b-v2.0"


@pytest.mark.asyncio
async def test_token_counting_model_not_in_model_list():
    """
    Test Token counter - when a model is not in model_list
    -> should use the default OpenAI tokenizer
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(
            model="special-alias",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert (
        response.tokenizer_type == "openai_tokenizer"
    )  # SHOULD use the OpenAI tokenizer
    assert response.model_used == "special-alias"


@pytest.mark.asyncio
async def test_gpt_token_counting():
    """
    Test Token counter
    -> should work for gpt-4
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    print("response: ", response)

    assert (
        response.tokenizer_type == "openai_tokenizer"
    )  # SHOULD use the OpenAI tokenizer
    assert response.request_model == "gpt-4"


@pytest.mark.asyncio
async def test_anthropic_messages_count_tokens_endpoint():
    """
    Test /v1/messages/count_tokens endpoint with Anthropic model
    - Should return response in Anthropic format: {"input_tokens": <count>}
    - Should work as wrapper around internal token_counter function
    """
    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens
    from fastapi import Request
    from unittest.mock import MagicMock

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello Claude!"}],
    }

    # Mock the _read_request_body function
    async def mock_read_request_body(request):
        return mock_request_data

    # Mock UserAPIKeyAuth
    mock_user_api_key_dict = MagicMock()

    # Patch the _read_request_body function
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints._read_request_body
    anthropic_endpoints._read_request_body = mock_read_request_body

    # Mock the internal token_counter function to return a controlled response
    async def mock_token_counter(request, call_endpoint=False):
        assert (
            call_endpoint == True
        ), "Should be called with call_endpoint=True for Anthropic endpoint"
        assert request.model == "claude-3-sonnet-20240229"
        assert request.messages == [{"role": "user", "content": "Hello Claude!"}]

        from litellm.types.utils import TokenCountResponse

        return TokenCountResponse(
            total_tokens=15,
            request_model="claude-3-sonnet-20240229",
            model_used="claude-3-sonnet-20240229",
            tokenizer_type="openai_tokenizer",
        )

    # Patch the imported token_counter function from proxy_server
    import litellm.proxy.proxy_server as proxy_server

    original_token_counter = proxy_server.token_counter
    proxy_server.token_counter = mock_token_counter

    try:
        # Call the endpoint
        response = await count_tokens(mock_request, mock_user_api_key_dict)

        # Verify response format matches Anthropic spec
        assert isinstance(response, dict)
        assert "input_tokens" in response
        assert response["input_tokens"] == 15
        assert len(response) == 1  # Should only contain input_tokens

        print("✅ Anthropic endpoint test passed!")

    finally:
        # Restore original functions
        anthropic_endpoints._read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_messages_count_tokens_with_non_anthropic_model():
    """
    Test /v1/messages/count_tokens endpoint with non-Anthropic model (GPT-4)
    - Should still work and return Anthropic format
    - Should call internal token_counter with from_anthropic_endpoint=True
    """
    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens
    from fastapi import Request
    from unittest.mock import MagicMock

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello GPT!"}],
    }

    # Mock the _read_request_body function
    async def mock_read_request_body(request):
        return mock_request_data

    # Mock UserAPIKeyAuth
    mock_user_api_key_dict = MagicMock()

    # Patch the _read_request_body function
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints._read_request_body
    anthropic_endpoints._read_request_body = mock_read_request_body

    # Mock the internal token_counter function to return a controlled response
    async def mock_token_counter(request, call_endpoint=True):
        assert (
            call_endpoint == True
        ), "Should be called with call_endpoint=True for Anthropic endpoint"
        assert request.model == "gpt-4"
        assert request.messages == [{"role": "user", "content": "Hello GPT!"}]

        from litellm.types.utils import TokenCountResponse

        return TokenCountResponse(
            total_tokens=12,
            request_model="gpt-4",
            model_used="gpt-4",
            tokenizer_type="openai_tokenizer",
        )

    # Patch the imported token_counter function from proxy_server
    import litellm.proxy.proxy_server as proxy_server

    original_token_counter = proxy_server.token_counter
    proxy_server.token_counter = mock_token_counter

    try:
        # Call the endpoint
        response = await count_tokens(mock_request, mock_user_api_key_dict)

        # Verify response format matches Anthropic spec
        assert isinstance(response, dict)
        assert "input_tokens" in response
        assert response["input_tokens"] == 12
        assert len(response) == 1  # Should only contain input_tokens

        print("✅ Non-Anthropic model test passed!")

    finally:
        # Restore original functions
        anthropic_endpoints._read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


@pytest.mark.asyncio
async def test_internal_token_counter_anthropic_provider_detection():
    """
    Test that the internal token_counter correctly detects Anthropic providers
    and handles the from_anthropic_endpoint flag appropriately
    """

    # Test with Anthropic provider
    llm_router = Router(
        model_list=[
            {
                "model_name": "claude-test",
                "litellm_params": {
                    "model": "anthropic/claude-3-sonnet-20240229",
                    "api_key": "test-key",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test with is_direct_request=False (simulating call from Anthropic endpoint)
    response = await token_counter(
        request=TokenCountRequest(
            model="claude-test",
            messages=[{"role": "user", "content": "hello"}],
        ),
        call_endpoint=True,
    )

    print("Anthropic provider test response:", response)

    # Verify response structure
    assert response.request_model == "claude-test"
    assert response.model_used == "claude-3-sonnet-20240229"
    assert response.total_tokens > 0

    # Test with non-Anthropic provider
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-test",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test with is_direct_request=False but non-Anthropic provider
    response = await token_counter(
        request=TokenCountRequest(
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
        ),
        call_endpoint=True,
    )

    print("Non-Anthropic provider test response:", response)

    # Verify response structure
    assert response.request_model == "gpt-test"
    assert response.model_used == "gpt-4"
    assert response.total_tokens > 0
    assert response.tokenizer_type == "openai_tokenizer"  # Should use LiteLLM tokenizer


@pytest.mark.asyncio
async def test_anthropic_endpoint_error_handling():
    """
    Test error handling in the /v1/messages/count_tokens endpoint
    """
    from litellm.proxy.anthropic_endpoints.endpoints import count_tokens
    from fastapi import Request, HTTPException
    from unittest.mock import MagicMock

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_user_api_key_dict = MagicMock()

    # Test missing model parameter
    mock_request_data = {
        "messages": [{"role": "user", "content": "Hello!"}]
        # Missing "model" key
    }

    async def mock_read_request_body(request):
        return mock_request_data

    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints

    original_read_request_body = anthropic_endpoints._read_request_body
    anthropic_endpoints._read_request_body = mock_read_request_body

    try:
        # Should raise HTTPException for missing model
        with pytest.raises(HTTPException) as exc_info:
            await count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "model parameter is required" in str(exc_info.value.detail)

        print("✅ Error handling test passed!")

    finally:
        anthropic_endpoints._read_request_body = original_read_request_body


@pytest.mark.asyncio
async def test_factory_anthropic_endpoint_calls_anthropic_counter():
    """Test that /v1/messages/count_tokens with Anthropic model uses Anthropic counter."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(
        return_value={"input_tokens": 42}
    )

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler
    ):
        # Mock router to return Anthropic deployment
        with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
            mock_router.model_list = [
                {
                    "model_name": "claude-3-5-sonnet",
                    "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                    "model_info": {},
                }
            ]

            # Mock the async method properly
            mock_router.async_get_available_deployment = AsyncMock(
                return_value={
                    "model_name": "claude-3-5-sonnet",
                    "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                    "model_info": {},
                }
            )

            # Set ANTHROPIC_API_KEY for the test
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                client = TestClient(app)

                response = client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "claude-3-5-sonnet",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["input_tokens"] == 42

                # Verify that Anthropic handler was called
                mock_handler.handle_count_tokens_request.assert_called_once()


@pytest.mark.asyncio
async def test_factory_gpt4_endpoint_does_not_call_anthropic_counter():
    """Test that /v1/messages/count_tokens with GPT-4 does NOT use Anthropic counter."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(
        return_value={"input_tokens": 42}
    )

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler
    ):
        # Mock litellm token counter
        with patch("litellm.token_counter") as mock_litellm_counter:
            mock_litellm_counter.return_value = 50

            # Mock router to return GPT-4 deployment
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                mock_router.model_list = [
                    {
                        "model_name": "gpt-4",
                        "litellm_params": {"model": "openai/gpt-4"},
                        "model_info": {},
                    }
                ]

                # Mock the async method properly
                mock_router.async_get_available_deployment = AsyncMock(
                    return_value={
                        "model_name": "gpt-4",
                        "litellm_params": {"model": "openai/gpt-4"},
                        "model_info": {},
                    }
                )

                client = TestClient(app)

                response = client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["input_tokens"] == 50

                # Verify that Anthropic handler was NOT called
                mock_handler.handle_count_tokens_request.assert_not_called()


@pytest.mark.asyncio
async def test_factory_normal_token_counter_endpoint_does_not_call_anthropic():
    """Test that /utils/token_counter does NOT use Anthropic counter even with Anthropic model."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app

    # Mock the global handler instance in token_counter module
    mock_handler = MagicMock()
    mock_handler.handle_count_tokens_request = AsyncMock(
        return_value={"input_tokens": 42}
    )

    with patch(
        "litellm.llms.anthropic.count_tokens.token_counter.anthropic_count_tokens_handler",
        mock_handler
    ):
        # Mock litellm token counter
        with patch("litellm.token_counter") as mock_litellm_counter:
            mock_litellm_counter.return_value = 35

            # Mock router to return Anthropic deployment
            with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
                mock_router.model_list = [
                    {
                        "model_name": "claude-3-5-sonnet",
                        "litellm_params": {
                            "model": "anthropic/claude-3-5-sonnet-20241022"
                        },
                        "model_info": {},
                    }
                ]

                # Mock the async method properly
                mock_router.async_get_available_deployment = AsyncMock(
                    return_value={
                        "model_name": "claude-3-5-sonnet",
                        "litellm_params": {
                            "model": "anthropic/claude-3-5-sonnet-20241022"
                        },
                        "model_info": {},
                    }
                )

                client = TestClient(app)

                response = client.post(
                    "/utils/token_counter",
                    json={
                        "model": "claude-3-5-sonnet",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["total_tokens"] == 35

                # Verify that Anthropic handler was NOT called (since call_endpoint=False)
                mock_handler.handle_count_tokens_request.assert_not_called()


@pytest.mark.asyncio
async def test_factory_registration():
    """Test that the new factory pattern correctly provides counters."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    # Test Anthropic ModelInfo provides token counter
    anthropic_model_info = AnthropicModelInfo()
    counter = anthropic_model_info.get_token_counter()
    assert counter is not None

    # Create test deployments
    anthropic_deployment = {
        "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"}
    }

    non_anthropic_deployment = {"litellm_params": {"model": "openai/gpt-4"}}

    # Test Anthropic counter supports provider
    assert counter.should_use_token_counting_api(custom_llm_provider="anthropic")
    assert not counter.should_use_token_counting_api(custom_llm_provider="openai")

    # Test non-Anthropic provider
    assert not counter.should_use_token_counting_api(custom_llm_provider="openai")

    # Test None deployment
    assert not counter.should_use_token_counting_api(custom_llm_provider=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gemini-2.5-pro", "vertex-ai-gemini-2.5-pro"])
async def test_vertex_ai_gemini_token_counting_with_contents(model_name):
    """
    Test token counting for Vertex AI Gemini model using contents format with call_endpoint=True
    """
    load_vertex_ai_credentials()
    llm_router = Router(
        model_list=[
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "gemini/gemini-2.5-pro",
                },
            },
            {
                "model_name": "vertex-ai-gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                },
            },
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test with contents format and call_endpoint=True
    response = await token_counter(
        request=TokenCountRequest(
            model=model_name,
            contents=[
                {"parts": [{"text": "Hello world, how are you doing today? i am ij"}]}
            ],
        ),
        call_endpoint=True,
    )

    print("Vertex AI Gemini token counting response:", response)

    # validate we have original response
    assert response.original_response is not None
    assert response.original_response.get("totalTokens") is not None
    assert response.original_response.get("promptTokensDetails") is not None

    prompt_tokens_details = response.original_response.get("promptTokensDetails")
    assert prompt_tokens_details is not None


@pytest.mark.asyncio
async def test_bedrock_count_tokens_endpoint():
    """
    Test that Bedrock CountTokens endpoint correctly extracts model from request body.
    """
    from litellm.router import Router

    # Mock the Bedrock CountTokens handler
    async def mock_count_tokens_handler(request_data, litellm_params, resolved_model):
        # Verify the correct model was resolved
        assert resolved_model == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert request_data["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert request_data["messages"] == [{"role": "user", "content": "Hello!"}]

        return {"input_tokens": 25}

    # Set up router with Bedrock model
    llm_router = Router(
        model_list=[
            {
                "model_name": "claude-bedrock",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test the mock handler directly to verify correct parameter extraction
    request_data = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    # Test the mock handler directly to verify correct parameter extraction
    await mock_count_tokens_handler(
        request_data, {}, "anthropic.claude-3-sonnet-20240229-v1:0"
    )


@pytest.mark.asyncio
async def test_vertex_ai_anthropic_token_counting():
    """
    Unit test for Vertex AI Anthropic token counting with mocked API calls.

    This tests the token counting implementation for Vertex AI partner models
    without making actual API calls. Mocks at the handler level to test the full flow.
    """
    from unittest.mock import AsyncMock, patch, MagicMock

    # Mock the Vertex AI partner models token counter response
    mock_token_response = {
        "input_tokens": 15,
        "tokenizer_used": "vertex_ai_partner_models",
    }

    llm_router = Router(
        model_list=[
            {
                "model_name": "vertex_ai/claude-3-5-sonnet-20241022",
                "litellm_params": {
                    "model": "vertex_ai/claude-3-5-sonnet-20241022",
                    "vertex_project": "test-project",
                    "vertex_location": "us-east5",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Mock the lower level handler method
    with patch(
        "litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler.VertexAIPartnerModelsTokenCounter.handle_count_tokens_request"
    ) as mock_handle_count_tokens:
        mock_handle_count_tokens.return_value = mock_token_response

        # Test with messages format and call_endpoint=True
        response = await token_counter(
            request=TokenCountRequest(
                model="vertex_ai/claude-3-5-sonnet-20241022",
                messages=[
                    {
                        "role": "user",
                        "content": "Hello Claude on Vertex AI! How are you?",
                    }
                ],
            ),
            call_endpoint=True,
        )

        # Validate that handle_count_tokens_request was called
        assert mock_handle_count_tokens.called

        # Verify the call arguments
        call_args = mock_handle_count_tokens.call_args
        assert call_args is not None
        assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert "messages" in call_args.kwargs["request_data"]
        assert (
            call_args.kwargs["request_data"]["messages"][0]["content"]
            == "Hello Claude on Vertex AI! How are you?"
        )

        # Validate response structure
        assert response.model_used == "claude-3-5-sonnet-20241022"
        assert response.request_model == "vertex_ai/claude-3-5-sonnet-20241022"
        assert response.total_tokens == 15
        assert response.tokenizer_type == "vertex_ai_partner_models"

        # Validate original response contains input_tokens
        assert response.original_response is not None
        assert "input_tokens" in response.original_response
        assert response.original_response["input_tokens"] == 15

@pytest.mark.parametrize("vertex_location", ["global", "us-central1"])
def test_vertex_ai_partner_models_token_counting_endpoint(vertex_location):
    """
    Test that the VertexAIPartnerModelsTokenCounter builds the correct endpoint URL
    for different vertex locations, including the special 'global' location.
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import (
        VertexAIPartnerModelsTokenCounter,
    )

    endpoint = VertexAIPartnerModelsTokenCounter()._build_count_tokens_endpoint(
        model="claude-3-5-sonnet-20241022",
        project_id="test-project",
        vertex_location=vertex_location,
        api_base=None,
    )
    if vertex_location == "global":
        assert endpoint.startswith("https://aiplatform.googleapis.com")
    else:
        assert endpoint.startswith(f"https://{vertex_location}-aiplatform.googleapis.com")


@pytest.mark.asyncio
async def test_bedrock_token_counter_error_propagation_bedrock_error():
    """
    Test that BedrockTokenCounter properly returns error response when BedrockError is raised.
    Verifies that the status code and error message are preserved.
    """
    counter = BedrockTokenCounter()

    # Mock the handler to raise BedrockError with specific status code
    with patch.object(
        counter, "count_tokens", wraps=counter.count_tokens
    ) as mock_count:
        # We need to patch at the handler level
        with patch(
            "litellm.llms.bedrock.count_tokens.bedrock_token_counter.BedrockCountTokensHandler"
        ) as MockHandler:
            mock_handler_instance = MockHandler.return_value
            mock_handler_instance.handle_count_tokens_request = AsyncMock(
                side_effect=BedrockError(
                    status_code=429, message="Rate limit exceeded"
                )
            )

            result = await counter.count_tokens(
                model_to_use="anthropic.claude-3-sonnet",
                messages=[{"role": "user", "content": "hello"}],
                contents=None,
                deployment={"litellm_params": {}},
                request_model="bedrock/anthropic.claude-3-sonnet",
            )

            assert result is not None
            assert result.error is True
            assert result.status_code == 429
            assert "Rate limit exceeded" in result.error_message
            assert result.tokenizer_type == "bedrock_api"
            assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_bedrock_token_counter_error_propagation_generic_exception():
    """
    Test that BedrockTokenCounter returns error response with 500 status for generic exceptions.
    """
    counter = BedrockTokenCounter()

    with patch(
        "litellm.llms.bedrock.count_tokens.bedrock_token_counter.BedrockCountTokensHandler"
    ) as MockHandler:
        mock_handler_instance = MockHandler.return_value
        mock_handler_instance.handle_count_tokens_request = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        result = await counter.count_tokens(
            model_to_use="anthropic.claude-3-sonnet",
            messages=[{"role": "user", "content": "hello"}],
            contents=None,
            deployment={"litellm_params": {}},
            request_model="bedrock/anthropic.claude-3-sonnet",
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 500
        assert "Unexpected error" in result.error_message


@pytest.mark.asyncio
async def test_bedrock_handler_httpx_error_status_code_propagation():
    """
    Test that BedrockCountTokensHandler properly extracts status code from httpx.HTTPStatusError.
    """
    handler = BedrockCountTokensHandler()

    # Create a mock httpx response with 403 status
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden - Invalid credentials"

    # Create HTTPStatusError
    http_error = httpx.HTTPStatusError(
        message="Client error '403 Forbidden'",
        request=MagicMock(),
        response=mock_response,
    )

    with patch.object(handler, "validate_count_tokens_request"):
        with patch.object(handler, "_get_aws_region_name", return_value="us-west-2"):
            with patch.object(
                handler, "transform_anthropic_to_bedrock_count_tokens", return_value={}
            ):
                with patch.object(
                    handler,
                    "get_bedrock_count_tokens_endpoint",
                    return_value="https://example.com",
                ):
                    with patch.object(handler, "_sign_request", return_value=({}, "{}")):
                        with patch(
                            "litellm.llms.bedrock.count_tokens.handler.get_async_httpx_client"
                        ) as mock_client:
                            mock_async_client = AsyncMock()
                            mock_async_client.post = AsyncMock(side_effect=http_error)
                            mock_client.return_value = mock_async_client

                            with pytest.raises(BedrockError) as exc_info:
                                await handler.handle_count_tokens_request(
                                    request_data={
                                        "model": "test",
                                        "messages": [
                                            {"role": "user", "content": "hello"}
                                        ],
                                    },
                                    litellm_params={},
                                    resolved_model="anthropic.claude-3-sonnet",
                                )

                            assert exc_info.value.status_code == 403
                            # Message should be the raw response text
                            assert exc_info.value.message == "Forbidden - Invalid credentials"


@pytest.mark.asyncio
async def test_proxy_token_counter_error_raises_exception_when_disabled():
    """
    Test that proxy token_counter raises ProxyException when disable_token_counter=True
    and provider returns an error response.
    """
    # Create error response
    error_response = TokenCountResponse(
        total_tokens=0,
        request_model="bedrock/anthropic.claude-3-sonnet",
        model_used="anthropic.claude-3-sonnet",
        tokenizer_type="bedrock_api",
        error=True,
        error_message="Rate limit exceeded",
        status_code=429,
    )

    # Create mock router that returns a deployment
    mock_deployment = {
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet",
        },
        "model_info": {},
    }

    mock_router = MagicMock()
    mock_router.async_get_available_deployment = AsyncMock(return_value=mock_deployment)

    setattr(litellm.proxy.proxy_server, "llm_router", mock_router)

    # Save original value and function
    original_disable = litellm.disable_token_counter
    original_get_provider_token_counter = litellm.proxy.proxy_server._get_provider_token_counter

    try:
        litellm.disable_token_counter = True

        # Create a mock counter that returns an error response
        mock_counter = MagicMock(spec=BedrockTokenCounter)
        mock_counter.should_use_token_counting_api.return_value = True
        mock_counter.count_tokens = AsyncMock(return_value=error_response)

        # Replace the function directly
        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "anthropic.claude-3-sonnet", "bedrock")

        litellm.proxy.proxy_server._get_provider_token_counter = mock_get_provider_token_counter

        with pytest.raises(ProxyException) as exc_info:
            await token_counter(
                request=TokenCountRequest(
                    model="claude-bedrock",
                    messages=[{"role": "user", "content": "hello"}],
                ),
                call_endpoint=True,
            )

        assert exc_info.value.code == "429"
        assert "Rate limit exceeded" in exc_info.value.message
    finally:
        litellm.disable_token_counter = original_disable
        litellm.proxy.proxy_server._get_provider_token_counter = original_get_provider_token_counter


@pytest.mark.asyncio
async def test_proxy_token_counter_error_falls_back_when_enabled():
    """
    Test that proxy token_counter falls back to local tokenizer when disable_token_counter=False
    and provider returns an error response.
    """
    # Create error response
    error_response = TokenCountResponse(
        total_tokens=0,
        request_model="bedrock/anthropic.claude-3-sonnet",
        model_used="anthropic.claude-3-sonnet",
        tokenizer_type="bedrock_api",
        error=True,
        error_message="Rate limit exceeded",
        status_code=429,
    )

    # Create mock router that returns a deployment
    mock_deployment = {
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet",
        },
        "model_info": {},
    }

    mock_router = MagicMock()
    mock_router.async_get_available_deployment = AsyncMock(return_value=mock_deployment)

    setattr(litellm.proxy.proxy_server, "llm_router", mock_router)

    # Save original value and function
    original_disable = litellm.disable_token_counter
    original_get_provider_token_counter = litellm.proxy.proxy_server._get_provider_token_counter

    try:
        litellm.disable_token_counter = False

        # Create a mock counter that returns an error response
        mock_counter = MagicMock(spec=BedrockTokenCounter)
        mock_counter.should_use_token_counting_api.return_value = True
        mock_counter.count_tokens = AsyncMock(return_value=error_response)

        # Replace the function directly
        def mock_get_provider_token_counter(deployment, model_to_use):
            return (mock_counter, "anthropic.claude-3-sonnet", "bedrock")

        litellm.proxy.proxy_server._get_provider_token_counter = mock_get_provider_token_counter

        # Should not raise, should fall back to local tokenizer
        result = await token_counter(
            request=TokenCountRequest(
                model="claude-bedrock",
                messages=[{"role": "user", "content": "hello"}],
            ),
            call_endpoint=True,
        )

        # Should have used the fallback tokenizer
        assert result.error is False
        assert result.total_tokens > 0
        assert result.tokenizer_type != "bedrock_api"
    finally:
        litellm.disable_token_counter = original_disable
        litellm.proxy.proxy_server._get_provider_token_counter = original_get_provider_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_returns_anthropic_error_format():
    """
    Test that /v1/messages/count_tokens returns errors in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints._read_request_body
    anthropic_endpoints._read_request_body = mock_read_request_body

    original_token_counter = proxy_server.token_counter

    # Mock token_counter to raise ProxyException with Bedrock-style error
    async def mock_token_counter_error(request, call_endpoint=False):
        raise ProxyException(
            message='{"detail":{"message":"Input is too long for requested model."}}',
            type="token_counting_error",
            param="model",
            code=400,
        )

    proxy_server.token_counter = mock_token_counter_error

    try:
        with pytest.raises(HTTPException) as exc_info:
            await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        # Verify HTTP status code is correct
        assert exc_info.value.status_code == 400

        # Verify error is in Anthropic format
        detail = exc_info.value.detail
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "invalid_request_error"
        assert detail["error"]["message"] == "Input is too long for requested model."
    finally:
        anthropic_endpoints._read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_403_permission_error_format():
    """
    Test that 403 errors are returned as permission_error in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints._read_request_body
    anthropic_endpoints._read_request_body = mock_read_request_body

    original_token_counter = proxy_server.token_counter

    # Mock token_counter to raise ProxyException with 403 error
    async def mock_token_counter_error(request, call_endpoint=False):
        raise ProxyException(
            message='{"Message":"Bearer Token has expired"}',
            type="token_counting_error",
            param="model",
            code=403,
        )

    proxy_server.token_counter = mock_token_counter_error

    try:
        with pytest.raises(HTTPException) as exc_info:
            await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 403

        detail = exc_info.value.detail
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "permission_error"
        assert detail["error"]["message"] == "Bearer Token has expired"
    finally:
        anthropic_endpoints._read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


@pytest.mark.asyncio
async def test_anthropic_endpoint_429_rate_limit_error_format():
    """
    Test that 429 errors are returned as rate_limit_error in Anthropic format.
    """
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints
    import litellm.proxy.proxy_server as proxy_server

    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-bedrock",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    async def mock_read_request_body(request):
        return mock_request_data

    mock_user_api_key_dict = MagicMock()

    original_read_request_body = anthropic_endpoints._read_request_body
    anthropic_endpoints._read_request_body = mock_read_request_body

    original_token_counter = proxy_server.token_counter

    # Mock token_counter to raise ProxyException with 429 error
    async def mock_token_counter_error(request, call_endpoint=False):
        raise ProxyException(
            message="Rate limit exceeded",
            type="token_counting_error",
            param="model",
            code=429,
        )

    proxy_server.token_counter = mock_token_counter_error

    try:
        with pytest.raises(HTTPException) as exc_info:
            await anthropic_count_tokens(mock_request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 429

        detail = exc_info.value.detail
        assert detail["type"] == "error"
        assert detail["error"]["type"] == "rate_limit_error"
        assert detail["error"]["message"] == "Rate limit exceeded"
    finally:
        anthropic_endpoints._read_request_body = original_read_request_body
        proxy_server.token_counter = original_token_counter


