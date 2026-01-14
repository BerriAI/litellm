"""
Test to verify that custom headers are correctly forwarded to Bedrock rerank API calls.

This test verifies the fix for the issue where headers configured via
forward_client_headers_to_llm_api were not being passed to Bedrock rerank provider.
"""

import json
import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

# Mock response for Bedrock rerank
# Format based on Bedrock rerank API response structure
bedrock_rerank_response = {
    "results": [
        {
            "index": 2,
            "relevanceScore": 0.95
        },
        {
            "index": 0,
            "relevanceScore": 0.1
        },
        {
            "index": 1,
            "relevanceScore": 0.05
        }
    ],
    "usage": {
        "search_units": 1
    }
}

# Test data
test_query = "What is the capital of the United States?"
test_documents = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Washington, D.C. is the capital of the United States.",
]


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
        "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
    ],
)
def test_bedrock_rerank_header_forwarding_sync(model):
    """
    Test that custom headers are correctly forwarded to Bedrock rerank API calls (sync).
    
    This test verifies the fix for the issue where headers configured via
    forward_client_headers_to_llm_api were not being passed to Bedrock rerank provider.
    """
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    
    # Headers that would be set by the proxy when forwarding client headers
    # Using x- prefix headers as those are the ones that get forwarded
    custom_headers = {
        "X-Custom-Header": "CustomValue",
        "X-BYOK-Token": "secret-token",
        "X-Test-Header": "test-value",
    }
    
    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(bedrock_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response
        
        try:
            # Call rerank with custom headers via kwargs
            # This simulates what the proxy does when forward_client_headers_to_llm_api is set
            response = litellm.rerank(
                model=model,
                query=test_query,
                documents=test_documents,
                top_n=3,
                client=client,
                headers=custom_headers,  # This is how proxy passes forwarded headers
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
            )
            
            assert isinstance(response, litellm.RerankResponse)
            
            # Verify that the request was made
            assert mock_post.called, "HTTP client post should be called"
            
            # Get the actual call arguments
            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})
            
            # Verify our custom headers are present in the request headers
            # Note: AWS SigV4 signing may modify header names to lowercase
            for header_key, header_value in custom_headers.items():
                header_found = (
                    header_key in headers
                    or header_key.lower() in headers
                    or any(k.lower() == header_key.lower() for k in headers.keys())
                )
                assert header_found, (
                    f"Header {header_key} should be in request headers. "
                    f"Found headers: {list(headers.keys())}"
                )
                
            print(f"✓ Test passed for {model} (sync)")
            print(f"  Headers correctly forwarded: {list(headers.keys())}")
            
        except Exception as e:
            pytest.fail(f"Failed to forward headers to {model}: {str(e)}")


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
        "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
    ],
)
@pytest.mark.asyncio
async def test_bedrock_rerank_header_forwarding_async(model):
    """
    Test that custom headers are correctly forwarded to Bedrock rerank API calls (async).
    
    This test verifies the fix for the issue where headers configured via
    forward_client_headers_to_llm_api were not being passed to Bedrock rerank provider.
    """
    litellm.set_verbose = True
    client = AsyncHTTPHandler()
    test_api_key = "test-bearer-token-12345"
    
    # Headers that would be set by the proxy when forwarding client headers
    # Using x- prefix headers as those are the ones that get forwarded
    custom_headers = {
        "X-Custom-Header": "CustomValue",
        "X-BYOK-Token": "secret-token",
        "X-Test-Header": "test-value",
    }
    
    from unittest.mock import AsyncMock
    
    with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(bedrock_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response
        
        try:
            # Call rerank with custom headers via kwargs
            response = await litellm.arerank(
                model=model,
                query=test_query,
                documents=test_documents,
                top_n=3,
                client=client,
                headers=custom_headers,  # This is how proxy passes forwarded headers
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
            )
            
            assert isinstance(response, litellm.RerankResponse)
            
            # Verify that the request was made
            assert mock_post.called, "HTTP client post should be called"
            
            # Get the actual call arguments
            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})
            
            # Verify our custom headers are present in the request headers
            # Note: AWS SigV4 signing may modify header names to lowercase
            for header_key, header_value in custom_headers.items():
                header_found = (
                    header_key in headers
                    or header_key.lower() in headers
                    or any(k.lower() == header_key.lower() for k in headers.keys())
                )
                assert header_found, (
                    f"Header {header_key} should be in request headers. "
                    f"Found headers: {list(headers.keys())}"
                )
                
            print(f"✓ Test passed for {model} (async)")
            print(f"  Headers correctly forwarded: {list(headers.keys())}")
            
        except Exception as e:
            pytest.fail(f"Failed to forward headers to {model}: {str(e)}")


def test_bedrock_rerank_extra_headers_and_headers_merge():
    """
    Test that both extra_headers and headers parameters are correctly merged for Bedrock rerank.
    
    This ensures that headers from kwargs (forwarded by proxy) and extra_headers
    (passed explicitly) are both included in the final headers sent to the provider.
    """
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    model = "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
    
    # Headers from proxy (via kwargs["headers"])
    proxy_headers = {"X-Forwarded-Header": "ProxyValue"}
    
    # Explicit extra_headers
    explicit_headers = {"X-Explicit-Header": "ExplicitValue"}
    
    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(bedrock_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response
        
        try:
            response = litellm.rerank(
                model=model,
                query=test_query,
                documents=test_documents,
                top_n=3,
                client=client,
                headers=proxy_headers,  # From proxy forwarding
                extra_headers=explicit_headers,  # Explicitly passed
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
            )
            
            assert isinstance(response, litellm.RerankResponse)
            
            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})
            
            # Both sets of headers should be present
            # Note: AWS SigV4 signing may modify header names to lowercase
            proxy_header_found = any(
                k.lower() == "x-forwarded-header" for k in headers.keys()
            )
            assert proxy_header_found, (
                "Proxy forwarded header should be present. "
                f"Found headers: {list(headers.keys())}"
            )
            
            explicit_header_found = any(
                k.lower() == "x-explicit-header" for k in headers.keys()
            )
            assert explicit_header_found, (
                "Explicitly passed header should be present. "
                f"Found headers: {list(headers.keys())}"
            )
            
            print("✓ Both header sources correctly merged and forwarded")
            print(f"  Final headers: {list(headers.keys())}")
            
        except Exception as e:
            pytest.fail(f"Failed to merge and forward headers: {str(e)}")

