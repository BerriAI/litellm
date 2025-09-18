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

        response = litellm.embedding(
            model=model,
            input=input_data,
            client=client,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key=test_api_key
        )

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