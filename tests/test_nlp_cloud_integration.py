
import sys
import os
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import litellm
from litellm.llms.nlp_cloud.chat.transformation import NLPCloudConfig
from litellm.llms.nlp_cloud.embed.transformation import NLPCloudEmbeddingConfig
from litellm.types.utils import ModelResponse, EmbeddingResponse

# Mock response data
MOCK_EMBEDDING_RESPONSE = {
    "embeddings": [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
}

MOCK_GENERATION_RESPONSE = {
    "generated_text": "Hello world from NLP Cloud",
    "nb_input_tokens": 5,
    "nb_generated_tokens": 10
}

@pytest.fixture
def mock_httpx_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_GENERATION_RESPONSE
    mock_response.text = "Mock response text"
    mock_response.headers = MagicMock()
    
    # Setup for sync calls
    mock_client.post.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_async_httpx_client():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_GENERATION_RESPONSE
    mock_response.text = "Mock response text"
    mock_response.headers = MagicMock()
    
    # Setup for async calls
    mock_client.post.return_value = mock_response
    
    return mock_client

def test_nlp_cloud_config_transformation():
    """Test NLPCloudConfig transformations."""
    config = NLPCloudConfig()
    
    # Test request transformation
    messages = [{"role": "user", "content": "Hello"}]
    transformed_request = config.transform_request(
        model="llama-3-70b",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={}
    )
    assert transformed_request["text"] == "Hello"
    
    # Test response transformation
    mock_raw_response = MagicMock()
    mock_raw_response.json.return_value = MOCK_GENERATION_RESPONSE
    
    model_response = ModelResponse()
    transformed_response = config.transform_response(
        model="llama-3-70b",
        raw_response=mock_raw_response,
        model_response=model_response,
        logging_obj=MagicMock(),
        request_data=transformed_request,
        messages=messages,
        optional_params={},
        litellm_params={},
        encoding=None
    )
    
    assert transformed_response.choices[0].message.content == "Hello world from NLP Cloud"
    assert transformed_response.usage.prompt_tokens == 5
    assert transformed_response.usage.completion_tokens == 10

def test_nlp_cloud_embedding_config_transformation():
    """Test NLPCloudEmbeddingConfig transformations."""
    config = NLPCloudEmbeddingConfig()
    
    # Test request transformation
    input_text = ["sentence 1", "sentence 2"]
    transformed_request = config.transform_embedding_request(
        model="paraphrase-multilingual-mpnet-base-v2",
        input=input_text,
        optional_params={},
        headers={}
    )
    assert transformed_request["sentences"] == input_text
    
    # Test response transformation
    mock_raw_response = MagicMock()
    mock_raw_response.json.return_value = MOCK_EMBEDDING_RESPONSE
    
    model_response = EmbeddingResponse()
    transformed_response = config.transform_embedding_response(
        model="paraphrase-multilingual-mpnet-base-v2",
        raw_response=mock_raw_response,
        model_response=model_response,
        logging_obj=MagicMock(),
        api_key="test-key",
        request_data=transformed_request,
        optional_params={},
        litellm_params={}
    )
    
    assert len(transformed_response.data) == 2
    assert transformed_response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert transformed_response.data[1]["embedding"] == [0.4, 0.5, 0.6]

@pytest.mark.asyncio
async def test_nlp_cloud_completion_routing(mock_httpx_client):
    """Test that litellm.completion routes correctly and uses the new config."""
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", side_effect=mock_httpx_client.post):
        response = litellm.completion(
            model="nlp_cloud/finetuned-llama-3-70b",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key"
        )
        assert response.choices[0].message.content == "Hello world from NLP Cloud"

@pytest.mark.asyncio
async def test_nlp_cloud_embedding_routing(mock_httpx_client):
    """Test that litellm.embedding routes correctly and uses the new config."""
    mock_httpx_client.post.return_value.json.return_value = MOCK_EMBEDDING_RESPONSE
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", side_effect=mock_httpx_client.post):
        response = litellm.embedding(
            model="nlp_cloud/paraphrase-multilingual-mpnet-base-v2",
            input=["test"],
            api_key="test-key"
        )
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
