# Test the following scenarios:
# 1. Generate a Key, and use it to make a call


import sys, os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

load_dotenv()
import os, io, time

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm, asyncio
from litellm.proxy.proxy_server import token_counter
from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token, update_spend
from litellm._logging import verbose_proxy_logger

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from litellm.proxy._types import TokenCountRequest, TokenCountResponse


from litellm import Router


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
    from unittest.mock import AsyncMock, MagicMock
    
    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello Claude!"}]
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
    async def mock_token_counter(request, is_direct_request=True):
        assert is_direct_request == False, "Should be called with is_direct_request=False for Anthropic endpoint"
        assert request.model == "claude-3-sonnet-20240229"
        assert request.messages == [{"role": "user", "content": "Hello Claude!"}]
        
        from litellm.proxy._types import TokenCountResponse
        return TokenCountResponse(
            total_tokens=15,
            request_model="claude-3-sonnet-20240229",
            model_used="claude-3-sonnet-20240229",
            tokenizer_type="openai_tokenizer"
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
    from unittest.mock import AsyncMock, MagicMock
    
    # Mock request object
    mock_request = MagicMock(spec=Request)
    mock_request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello GPT!"}]
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
    async def mock_token_counter(request, is_direct_request=True):
        assert is_direct_request == False, "Should be called with is_direct_request=False for Anthropic endpoint"
        assert request.model == "gpt-4"
        assert request.messages == [{"role": "user", "content": "Hello GPT!"}]
        
        from litellm.proxy._types import TokenCountResponse
        return TokenCountResponse(
            total_tokens=12,
            request_model="gpt-4", 
            model_used="gpt-4",
            tokenizer_type="openai_tokenizer"
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
                    "api_key": "test-key"
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
        is_direct_request=False
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
        is_direct_request=False
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
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app
    
    # Mock the anthropic token counting function
    with patch('litellm.proxy.utils.count_tokens_with_anthropic_api') as mock_anthropic_count:
        mock_anthropic_count.return_value = {
            "total_tokens": 42,
            "tokenizer_used": "anthropic"
        }
        
        # Mock router to return Anthropic deployment
        with patch('litellm.proxy.proxy_server.llm_router') as mock_router:
            mock_router.model_list = [{
                "model_name": "claude-3-5-sonnet",
                "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                "model_info": {}
            }]
            
            client = TestClient(app)
            
            response = client.post(
                "/v1/messages/count_tokens",
                json={
                    "model": "claude-3-5-sonnet",
                    "messages": [{"role": "user", "content": "Hello"}]
                },
                headers={"Authorization": "Bearer test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["input_tokens"] == 42
            
            # Verify that Anthropic API was called
            mock_anthropic_count.assert_called_once()


@pytest.mark.asyncio
async def test_factory_gpt4_endpoint_does_not_call_anthropic_counter():
    """Test that /v1/messages/count_tokens with GPT-4 does NOT use Anthropic counter."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app
    
    # Mock the anthropic token counting function
    with patch('litellm.proxy.utils.count_tokens_with_anthropic_api') as mock_anthropic_count:
        # Mock litellm token counter
        with patch('litellm.token_counter') as mock_litellm_counter:
            mock_litellm_counter.return_value = 50
            
            # Mock router to return GPT-4 deployment
            with patch('litellm.proxy.proxy_server.llm_router') as mock_router:
                mock_router.model_list = [{
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {}
                }]
                
                client = TestClient(app)
                
                response = client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "Hello"}]
                    },
                    headers={"Authorization": "Bearer test-key"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["input_tokens"] == 50
                
                # Verify that Anthropic API was NOT called
                mock_anthropic_count.assert_not_called()


@pytest.mark.asyncio
async def test_factory_normal_token_counter_endpoint_does_not_call_anthropic():
    """Test that /utils/token_counter does NOT use Anthropic counter even with Anthropic model."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app
    
    # Mock the anthropic token counting function
    with patch('litellm.proxy.utils.count_tokens_with_anthropic_api') as mock_anthropic_count:
        # Mock litellm token counter
        with patch('litellm.token_counter') as mock_litellm_counter:
            mock_litellm_counter.return_value = 35
            
            # Mock router to return Anthropic deployment
            with patch('litellm.proxy.proxy_server.llm_router') as mock_router:
                mock_router.model_list = [{
                    "model_name": "claude-3-5-sonnet",
                    "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"},
                    "model_info": {}
                }]
                
                client = TestClient(app)
                
                response = client.post(
                    "/utils/token_counter",
                    json={
                        "model": "claude-3-5-sonnet",
                        "messages": [{"role": "user", "content": "Hello"}]
                    },
                    headers={"Authorization": "Bearer test-key"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["total_tokens"] == 35
                
                # Verify that Anthropic API was NOT called (since is_direct_request=True)
                mock_anthropic_count.assert_not_called()


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
    
    non_anthropic_deployment = {
        "litellm_params": {"model": "openai/gpt-4"}
    }
    
    # Test Anthropic counter supports provider
    assert counter.supports_provider(anthropic_deployment, from_endpoint=True)
    assert not counter.supports_provider(anthropic_deployment, from_endpoint=False)
    
    # Test non-Anthropic provider
    assert not counter.supports_provider(non_anthropic_deployment, from_endpoint=True)
    assert not counter.supports_provider(non_anthropic_deployment, from_endpoint=False)
    
    # Test None deployment
    assert not counter.supports_provider(None, from_endpoint=True)
    assert not counter.supports_provider(None, from_endpoint=False)


@pytest.mark.asyncio 
async def test_factory_anthropic_counter_supports_provider():
    """Test AnthropicTokenCounter provider detection logic."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    
    anthropic_model_info = AnthropicModelInfo()
    counter = anthropic_model_info.get_token_counter()
    
    # Test Anthropic provider detection
    anthropic_deployment = {
        "litellm_params": {"model": "anthropic/claude-3-5-sonnet-20241022"}
    }
    assert counter.supports_provider(anthropic_deployment, from_endpoint=True)
    assert not counter.supports_provider(anthropic_deployment, from_endpoint=False)
    
    # Test non-Anthropic provider
    openai_deployment = {
        "litellm_params": {"model": "openai/gpt-4"}
    }
    assert not counter.supports_provider(openai_deployment, from_endpoint=True)
    assert not counter.supports_provider(openai_deployment, from_endpoint=False)
    
    # Test None deployment
    assert not counter.supports_provider(None, from_endpoint=True)
    assert not counter.supports_provider(None, from_endpoint=False)
