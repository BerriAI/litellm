import json
import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

# Mock responses for different embedding models
titan_embedding_response = {
    "embedding": [0.1, 0.2, 0.3],
    "inputTextTokenCount": 10
}

cohere_embedding_response = {
    "embeddings": [[0.1, 0.2, 0.3]],
    "inputTextTokenCount": 10
}

twelvelabs_embedding_response = {
    "embedding": [0.1, 0.2, 0.3],
    "embeddingOption": "visual-text",
    "startSec": 0.0,
    "endSec": 1.0
}

# Test data
test_input = "Hello world from litellm"
test_image_base64 = "data:image/png,test_image_base64_data"


@pytest.mark.parametrize(
    "model,input_type,embed_response",
    [
        ("bedrock/amazon.titan-embed-text-v1", "text", titan_embedding_response),
        ("bedrock/amazon.titan-embed-text-v2:0", "text", titan_embedding_response),
        ("bedrock/amazon.titan-embed-image-v1", "image", titan_embedding_response),
        ("bedrock/cohere.embed-english-v3", "text", cohere_embedding_response),
        ("bedrock/cohere.embed-multilingual-v3", "text", cohere_embedding_response),
        ("bedrock/twelvelabs.marengo-embed-2-7-v1:0", "text", twelvelabs_embedding_response),
        ("bedrock/twelvelabs.marengo-embed-2-7-v1:0", "image", twelvelabs_embedding_response),
    ],
)
def test_bedrock_embedding_with_api_key_bearer_token(model, input_type, embed_response):
    """Test embedding functionality with bearer token authentication"""
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(embed_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        input_data = test_image_base64 if input_type == "image" else test_input

        # Add inputType parameter for TwelveLabs Marengo models
        kwargs = {
            "model": model,
            "input": input_data,
            "client": client,
            "aws_region_name": "us-east-1",
            "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "api_key": test_api_key
        }
        
        # Add input_type parameter for TwelveLabs Marengo models (maps to inputType)
        if "twelvelabs.marengo-embed" in model:
            kwargs["input_type"] = input_type
            
        response = litellm.embedding(**kwargs)

        assert isinstance(response, litellm.EmbeddingResponse)
        assert isinstance(response.data[0]['embedding'], list)
        assert len(response.data[0]['embedding']) == 3  # Based on mock response

        headers = mock_post.call_args.kwargs.get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {test_api_key}"


@pytest.mark.parametrize(
    "model,input_type,embed_response",
    [
        ("bedrock/amazon.titan-embed-text-v1", "text", titan_embedding_response),
    ],
)
def test_bedrock_embedding_with_env_variable_bearer_token(model, input_type, embed_response):
    """Test embedding functionality with bearer token from environment variable"""
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "env-bearer-token-12345"
    
    with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": test_api_key}), \
         patch.object(client, "post") as mock_post:
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(embed_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        response = litellm.embedding(
            model=model,
            input=test_input,
            client=client,
            aws_region_name="us-west-2",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-west-2.amazonaws.com",
        )

        assert isinstance(response, litellm.EmbeddingResponse)
        headers = mock_post.call_args.kwargs.get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {test_api_key}"


@pytest.mark.asyncio
async def test_async_bedrock_embedding_with_bearer_token():
    """Test async embedding functionality with bearer token authentication"""
    litellm.set_verbose = True
    client = AsyncHTTPHandler()
    test_api_key = "async-bearer-token-12345"
    model = "bedrock/amazon.titan-embed-text-v1"

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(titan_embedding_response)
        mock_response.json = Mock(return_value=titan_embedding_response)
        mock_post.return_value = mock_response

        response = await litellm.aembedding(
            model=model,
            input=test_input,
            client=client,
            aws_region_name="us-west-2",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-west-2.amazonaws.com",
            api_key=test_api_key
        )

        assert isinstance(response, litellm.EmbeddingResponse)
        
        headers = mock_post.call_args.kwargs.get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {test_api_key}"


def test_bedrock_embedding_with_sigv4():
    """Test embedding falls back to SigV4 auth when no bearer token is provided"""
    litellm.set_verbose = True
    model = "bedrock/amazon.titan-embed-text-v1"

    with patch("litellm.llms.bedrock.embed.embedding.BedrockEmbedding.embeddings") as mock_bedrock_embed:
        mock_embedding_response = litellm.EmbeddingResponse()
        mock_embedding_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_bedrock_embed.return_value = mock_embedding_response

        response = litellm.embedding(
            model=model,
            input=test_input,
            aws_region_name="us-west-2",
        )

        assert isinstance(response, litellm.EmbeddingResponse)
        mock_bedrock_embed.assert_called_once()


def test_bedrock_titan_v2_encoding_format_float():
    """Test amazon.titan-embed-text-v2:0 with encoding_format=float parameter"""
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    model = "bedrock/amazon.titan-embed-text-v2:0"

    # Mock response with embeddingsByType for binary format (addressing issue #14680)
    titan_v2_response = {
        "embedding": [0.1, 0.2, 0.3],
        "inputTextTokenCount": 10
    }

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(titan_v2_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        response = litellm.embedding(
            model=model,
            input=test_input,
            encoding_format="float",  # This should work but currently throws UnsupportedParamsError
            client=client,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=test_api_key
        )

        assert isinstance(response, litellm.EmbeddingResponse)
        assert isinstance(response.data[0]['embedding'], list)
        assert len(response.data[0]['embedding']) == 3

        # Verify that the request contains embeddingTypes: ["float"] instead of encoding_format
        request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
        assert "embeddingTypes" in request_body
        assert request_body["embeddingTypes"] == ["float"]
        assert "encoding_format" not in request_body


def test_bedrock_titan_v2_encoding_format_base64():
    """Test amazon.titan-embed-text-v2:0 with encoding_format=base64 parameter (maps to binary)"""
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    model = "bedrock/amazon.titan-embed-text-v2:0"

    # Mock response with embeddingsByType for binary format
    titan_v2_binary_response = {
        "embeddingsByType": {
            "binary": "YmluYXJ5X2VtYmVkZGluZ19kYXRh"  # base64 encoded binary data
        },
        "inputTextTokenCount": 10
    }

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(titan_v2_binary_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        response = litellm.embedding(
            model=model,
            input=test_input,
            encoding_format="base64",  # This should map to embeddingTypes: ["binary"]
            client=client,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=test_api_key
        )

        assert isinstance(response, litellm.EmbeddingResponse)

        # Verify that the request contains embeddingTypes: ["binary"] for base64 encoding
        request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
        assert "embeddingTypes" in request_body
        assert request_body["embeddingTypes"] == ["binary"]


def test_twelvelabs_input_type_parameter_mapping():
    """Test that input_type parameter is correctly mapped to inputType for TwelveLabs models"""
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    model = "bedrock/twelvelabs.marengo-embed-2-7-v1:0"

    twelvelabs_response = {
        "data": [{
            "embedding": [0.1, 0.2, 0.3],
            "inputTextTokenCount": 10
        }]
    }

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(twelvelabs_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        # Test with input_type parameter (new LiteLLM parameter)
        response = litellm.embedding(
            model=model,
            input=test_input,
            client=client,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=test_api_key,
            input_type="text"  # New parameter that should map to inputType
        )

        assert isinstance(response, litellm.EmbeddingResponse)
        assert isinstance(response.data[0]['embedding'], list)
        assert len(response.data[0]['embedding']) == 3

        # Verify that the request contains inputType (mapped from input_type)
        request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
        assert "inputType" in request_body
        assert request_body["inputType"] == "text"
        assert "input_type" not in request_body  # Should be mapped, not passed through


def test_twelvelabs_input_type_parameter_mapping_async_invoke():
    """Test that input_type parameter is correctly mapped to inputType for TwelveLabs async invoke models"""
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    model = "bedrock/async_invoke/twelvelabs.marengo-embed-2-7-v1:0"

    async_invoke_response = {
        "invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"
    }

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(async_invoke_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        # Test with input_type parameter for async invoke
        response = litellm.embedding(
            model=model,
            input=test_input,
            client=client,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=test_api_key,
            output_s3_uri="s3://test-bucket/async-invoke-output/",
            input_type="text"  # New parameter that should map to inputType
        )

        assert isinstance(response, litellm.EmbeddingResponse)
        assert hasattr(response, '_hidden_params')
        assert response._hidden_params is not None
        assert hasattr(response._hidden_params, '_invocation_arn')

        # Verify that the request contains inputType in modelInput (mapped from input_type)
        request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
        assert "modelInput" in request_body
        assert "inputType" in request_body["modelInput"]
        assert request_body["modelInput"]["inputType"] == "text"
        assert "input_type" not in request_body  # Should be mapped, not passed through


def test_twelvelabs_missing_input_type_error():
    """Test that missing input_type parameter defaults to 'text' for TwelveLabs models"""
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    
    # Test TwelveLabs model - should default to 'text' when input_type is missing
    twelvelabs_model = "bedrock/twelvelabs.marengo-embed-2-7-v1:0"
    twelvelabs_response = {
        "data": [{
            "embedding": [0.1, 0.2, 0.3],
            "inputTextTokenCount": 10
        }]
    }
    
    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(twelvelabs_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        # Test that missing input_type defaults to "text" for TwelveLabs
        response = litellm.embedding(
            model=twelvelabs_model,
            input=test_input,
            client=client,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=test_api_key
            # No input_type parameter - should default to "text"
        )
        
        # Verify the response is successful
        assert isinstance(response, litellm.EmbeddingResponse)
        
        # Verify that the request contains inputType: "text" by default
        request_body = json.loads(mock_post.call_args.kwargs.get("data", "{}"))
        assert "inputType" in request_body
        assert request_body["inputType"] == "text"
    
    # Test Amazon Titan model - should NOT throw error (input_type not required)
    titan_model = "bedrock/amazon.titan-embed-text-v1"
    titan_response = {
        "embedding": [0.1, 0.2, 0.3],
        "inputTextTokenCount": 10
    }
    
    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(titan_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        # Test that missing input_type does NOT throw an error for Amazon Titan
        response = litellm.embedding(
            model=titan_model,
            input=test_input,
            client=client,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=test_api_key
            # No input_type parameter - should work fine
        )
        
        # Should succeed without input_type
        assert isinstance(response, litellm.EmbeddingResponse)


@pytest.mark.parametrize(
    "model,embed_response",
    [
        ("bedrock/amazon.titan-embed-text-v1", titan_embedding_response),
        ("bedrock/amazon.titan-embed-text-v2:0", titan_embedding_response),
        ("bedrock/cohere.embed-english-v3", cohere_embedding_response),
    ],
)
def test_bedrock_embedding_header_forwarding(model, embed_response):
    """
    Test that custom headers are correctly forwarded to Bedrock embedding API calls.
    
    This test verifies the fix for the issue where headers configured via
    forward_client_headers_to_llm_api were not being passed to Bedrock embedding provider.
    
    Relevant Issue: https://github.com/BerriAI/litellm/pull/16042
    """
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    
    # Headers that would be set by the proxy when forwarding client headers
    custom_headers = {
        "X-Custom-Header": "CustomValue",
        "X-BYOK-Token": "secret-token",
        "Extra-Header": "foobar",
    }
    
    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(embed_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        
        try:
            # Call embedding with custom headers via kwargs
            # This simulates what the proxy does when forward_client_headers_to_llm_api is set
            response = litellm.embedding(
                model=model,
                input=test_input,
                client=client,
                headers=custom_headers,  # This is how proxy passes forwarded headers
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
            )
            
            assert isinstance(response, litellm.EmbeddingResponse)
            
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
                
            print(f"✓ Test passed for {model}")
            print(f"  Headers correctly forwarded: {list(headers.keys())}")
            
        except Exception as e:
            pytest.fail(f"Failed to forward headers to {model}: {str(e)}")


def test_bedrock_embedding_extra_headers_and_headers_merge():
    """
    Test that both extra_headers and headers parameters are correctly merged for Bedrock embeddings.
    
    This ensures that headers from kwargs (forwarded by proxy) and extra_headers
    (passed explicitly) are both included in the final headers sent to the provider.
    """
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    model = "bedrock/amazon.titan-embed-text-v1"
    
    # Headers from proxy (via kwargs["headers"])
    proxy_headers = {"X-Forwarded-Header": "ProxyValue"}
    
    # Explicit extra_headers
    explicit_headers = {"X-Explicit-Header": "ExplicitValue"}
    
    # Mock response
    embed_response = {
        "embedding": [0.1, 0.2, 0.3],
        "inputTextTokenCount": 10
    }
    
    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(embed_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        
        try:
            response = litellm.embedding(
                model=model,
                input=test_input,
                client=client,
                headers=proxy_headers,  # From proxy forwarding
                extra_headers=explicit_headers,  # Explicitly passed
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
            )
            
            assert isinstance(response, litellm.EmbeddingResponse)
            
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


def test_bedrock_cohere_v4_embedding_response_parsing():
    """
    Test parsing of Bedrock Cohere v4 embedding response which returns a dictionary of embeddings
    keyed by type (e.g. 'float', 'int8') instead of a direct list.
    """
    litellm.set_verbose = True
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    model = "bedrock/cohere.embed-v4:0"

    # Mock response for Cohere v4 with multiple embedding types
    cohere_v4_response = {
        "embeddings": {
            "float": [[0.1, 0.2, 0.3]],
            "int8": [[1, 2, 3]]
        },
        "response_type": "embeddings_by_type",
        "id": "test-id",
        "texts": ["test input"]
    }

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(cohere_v4_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        response = litellm.embedding(
            model=model,
            input=["test input"],
            client=client,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=test_api_key
        )

        assert isinstance(response, litellm.EmbeddingResponse)
        
        # Verify we get two embedding objects back (one for float, one for int8)
        assert len(response.data) == 2
        
        # Check first embedding (float)
        assert response.data[0]['object'] == 'embedding'
        assert response.data[0]['embedding'] == [0.1, 0.2, 0.3]
        assert response.data[0]['type'] == 'float'
        
        # Check second embedding (int8)
        assert response.data[1]['object'] == 'embedding'
        assert response.data[1]['embedding'] == [1, 2, 3]
        assert response.data[1]['type'] == 'int8'


def test_bedrock_embedding_custom_headers_with_iam_role_and_custom_api_base():
    """
    Test that custom headers are correctly forwarded when using IAM role credentials
    (with session token) and a custom api_base.
    
    This test verifies the fix for the issue where custom headers were not being
    forwarded to Bedrock embeddings endpoint when using:
    - IAM role authentication (session tokens)
    - Custom api_base (proxy endpoint)
    
    The fix converts HeadersDict to regular dict before passing to httpx, ensuring
    headers are properly forwarded even with IAM roles and custom endpoints.
    
    Relevant Issue: Custom headers not forwarded with IAM roles + custom api_base
    """
    litellm.set_verbose = True
    client = HTTPHandler()
    
    # Simulate IAM role credentials with session token
    aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
    aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    aws_session_token = "AQoEXAMPLEH4aoAH0gNCAPyJxz4BlCFFxWNE1OPTgk5TthT+FvwqnKwRcOIfrRh3c/LTo6UDdyJwOOvEVPvLXCrrrUtdnniCEXAMPLE/IvU1dYUg2RVAJBanLiHb4IgRmpV3ZXrzoB348V+jZfXvYhEXAMPLEEXAMPLE"
    
    # Custom api_base (simulating a proxy endpoint)
    custom_api_base = "https://gateway.example.com/v1/bedrock-runtime/us-east-1"
    
    # Custom headers that need to be forwarded
    custom_headers = {
        "X-Custom-Header-1": "test-value-1",
        "X-Custom-Header-2": "test-value-2",
        "X-Forwarded-For": "192.168.1.1",
        "X-BYOK-Token": "secret-token-12345",
    }
    
    # Mock response
    embed_response = {
        "embedding": [0.1, 0.2, 0.3],
        "inputTextTokenCount": 10
    }
    
    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(embed_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        
        try:
            response = litellm.embedding(
                model="bedrock/amazon.titan-embed-text-v1",
                input=test_input,
                client=client,
                extra_headers=custom_headers,
                api_base=custom_api_base,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,  # IAM role session token
                aws_region_name="us-east-1",
            )
            
            assert isinstance(response, litellm.EmbeddingResponse)
            
            # Verify that the request was made
            assert mock_post.called, "HTTP client post should be called"
            
            # Get the actual call arguments
            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})
            
            # Verify custom headers are present in the request
            # Note: HeadersDict should be converted to regular dict, so headers should be accessible
            for header_key, header_value in custom_headers.items():
                # Check if header exists (case-insensitive for HTTP headers)
                header_found = any(
                    k.lower() == header_key.lower() for k in headers.keys()
                )
                assert header_found, (
                    f"Custom header {header_key} should be in request headers. "
                    f"Found headers: {list(headers.keys())}"
                )
                
                # Verify the value matches
                header_value_found = None
                for k, v in headers.items():
                    if k.lower() == header_key.lower():
                        header_value_found = v
                        break
                
                assert header_value_found == header_value, (
                    f"Header {header_key} should have value {header_value}, "
                    f"but found {header_value_found}"
                )
            
            # Verify AWS signature headers are also present
            assert "Authorization" in headers, "AWS signature should be present"
            assert "X-Amz-Date" in headers, "AWS date header should be present"
            assert "X-Amz-Security-Token" in headers, "Session token header should be present"
            assert headers["X-Amz-Security-Token"] == aws_session_token, (
                "Session token should match the provided token"
            )
            
            # Verify the custom api_base was used
            called_url = call_kwargs.get("url", "")
            assert custom_api_base in str(called_url), (
                f"Custom api_base {custom_api_base} should be used. "
                f"Got URL: {called_url}"
            )
            
            print("✓ Test passed: Custom headers forwarded with IAM role + custom api_base")
            print(f"  Custom headers found: {[k for k in headers.keys() if k.lower().startswith('x-custom') or k.lower().startswith('x-forwarded')]}")
            print(f"  AWS headers found: {[k for k in headers.keys() if k.lower().startswith('x-amz') or k.lower() == 'authorization']}")
            
        except Exception as e:
            pytest.fail(f"Failed to forward headers with IAM role + custom api_base: {str(e)}")


@pytest.mark.asyncio
async def test_bedrock_embedding_custom_headers_with_iam_role_and_custom_api_base_async():
    """
    Test that custom headers are correctly forwarded in async mode when using IAM role
    credentials (with session token) and a custom api_base.
    
    This is the async version of the test above, verifying the fix works for both
    sync and async embedding calls.
    """
    litellm.set_verbose = True
    client = AsyncHTTPHandler()
    
    # Simulate IAM role credentials with session token
    aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
    aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    aws_session_token = "AQoEXAMPLEH4aoAH0gNCAPyJxz4BlCFFxWNE1OPTgk5TthT+FvwqnKwRcOIfrRh3c/LTo6UDdyJwOOvEVPvLXCrrrUtdnniCEXAMPLE/IvU1dYUg2RVAJBanLiHb4IgRmpV3ZXrzoB348V+jZfXvYhEXAMPLEEXAMPLE"
    
    # Custom api_base (simulating a proxy endpoint)
    custom_api_base = "https://gateway.example.com/v1/bedrock-runtime/us-west-2"
    
    # Custom headers that need to be forwarded
    custom_headers = {
        "X-Custom-Header-1": "test-value-1",
        "X-Custom-Header-2": "test-value-2",
        "X-Forwarded-For": "192.168.1.1",
        "X-BYOK-Token": "secret-token-12345",
    }
    
    # Mock response
    embed_response = {
        "embedding": [0.1, 0.2, 0.3],
        "inputTextTokenCount": 10
    }
    
    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(embed_response)
        mock_response.json = Mock(return_value=embed_response)
        mock_post.return_value = mock_response
        
        try:
            response = await litellm.aembedding(
                model="bedrock/amazon.titan-embed-text-v1",
                input=test_input,
                client=client,
                extra_headers=custom_headers,
                api_base=custom_api_base,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,  # IAM role session token
                aws_region_name="us-west-2",
            )
            
            assert isinstance(response, litellm.EmbeddingResponse)
            
            # Verify that the request was made
            assert mock_post.called, "HTTP client post should be called"
            
            # Get the actual call arguments
            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})
            
            # Verify custom headers are present in the request
            for header_key, header_value in custom_headers.items():
                # Check if header exists (case-insensitive for HTTP headers)
                header_found = any(
                    k.lower() == header_key.lower() for k in headers.keys()
                )
                assert header_found, (
                    f"Custom header {header_key} should be in request headers. "
                    f"Found headers: {list(headers.keys())}"
                )
                
                # Verify the value matches
                header_value_found = None
                for k, v in headers.items():
                    if k.lower() == header_key.lower():
                        header_value_found = v
                        break
                
                assert header_value_found == header_value, (
                    f"Header {header_key} should have value {header_value}, "
                    f"but found {header_value_found}"
                )
            
            # Verify AWS signature headers are also present
            assert "Authorization" in headers, "AWS signature should be present"
            assert "X-Amz-Date" in headers, "AWS date header should be present"
            assert "X-Amz-Security-Token" in headers, "Session token header should be present"
            assert headers["X-Amz-Security-Token"] == aws_session_token, (
                "Session token should match the provided token"
            )
            
            # Verify the custom api_base was used
            called_url = call_kwargs.get("url", "")
            assert custom_api_base in str(called_url), (
                f"Custom api_base {custom_api_base} should be used. "
                f"Got URL: {called_url}"
            )
            
            print("✓ Test passed (async): Custom headers forwarded with IAM role + custom api_base")
            print(f"  Custom headers found: {[k for k in headers.keys() if k.lower().startswith('x-custom') or k.lower().startswith('x-forwarded')]}")
            print(f"  AWS headers found: {[k for k in headers.keys() if k.lower().startswith('x-amz') or k.lower() == 'authorization']}")
            
        except Exception as e:
            pytest.fail(f"Failed to forward headers with IAM role + custom api_base (async): {str(e)}")
