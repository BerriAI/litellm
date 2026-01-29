
import sys
import os
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import litellm
from litellm.llms.nlp_cloud.chat.handler import NLPCloudChatHandler
from litellm.utils import ModelResponse

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
    
    # Setup for async calls
    mock_client.post.return_value = mock_response
    
    return mock_client

def test_nlp_cloud_embedding(mock_httpx_client):
    """Test the embedding function directly."""
    
    # Mock return value for embedding specifically
    mock_httpx_client.post.return_value.json.return_value = MOCK_EMBEDDING_RESPONSE
    
    logging_obj = MagicMock()
    handler = NLPCloudChatHandler()
    
    response = handler.embedding(
        model="paraphrase-multilingual-mpnet-base-v2",
        input=["sentence 1", "sentence 2"],
        api_key="test_key",
        api_base="https://api.nlpcloud.io/v1/",
        logging_obj=logging_obj,
        client=mock_httpx_client
    )
    
    # Verify API call
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args[0][0] == "https://api.nlpcloud.io/v1/paraphrase-multilingual-mpnet-base-v2/embeddings"
    assert call_args[1]["headers"]["Authorization"] == "Token test_key"
    assert call_args[1]["headers"]["content-type"] == "application/json"
    
    # Verify payload
    import json
    payload = json.loads(call_args[1]["data"])
    assert payload["sentences"] == ["sentence 1", "sentence 2"]
    
    # Verify response object
    assert len(response.data) == 2
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert response.data[1]["embedding"] == [0.4, 0.5, 0.6]
    assert response.model == "paraphrase-multilingual-mpnet-base-v2"

@pytest.mark.asyncio
async def test_nlp_cloud_acompletion(mock_async_httpx_client):
    """Test the acompletion function directly."""
    
    logging_obj = MagicMock()
    model_response = ModelResponse()
    handler = NLPCloudChatHandler()
    
    response = await handler.acompletion(
        model="finetuned-llama-3-70b",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://api.nlpcloud.io/v1/",
        model_response=model_response,
        print_verbose=print,
        encoding=None,
        api_key="test_key",
        logging_obj=logging_obj,
        optional_params={},
        litellm_params={},
        client=mock_async_httpx_client
    )
    
    # Verify API call
    mock_async_httpx_client.post.assert_called_once()
    call_args = mock_async_httpx_client.post.call_args
    assert call_args[0][0] == "https://api.nlpcloud.io/v1/finetuned-llama-3-70b/generation"
    
    # Verify response
    assert response.choices[0].message.content == "Hello world from NLP Cloud"
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 10

def test_nlp_cloud_completion_streaming(mock_httpx_client):
    """Test the completion function with streaming (sync)."""
    
    # Mock streaming response
    mock_response = mock_httpx_client.post.return_value
    mock_response.iter_lines.return_value = iter([
        b'{"generated_text": "Hello "}',
        b'{"generated_text": "world "}',
        b'{"generated_text": "from NLP Cloud [DONE]"}'
    ])
    
    logging_obj = MagicMock()
    # Need to mock logging_obj.model_call_details
    logging_obj.model_call_details = {"litellm_params": {}}
    
    model_response = ModelResponse()
    handler = NLPCloudChatHandler()
    
    response = handler.completion(
        model="finetuned-llama-3-70b",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://api.nlpcloud.io/v1/",
        model_response=model_response,
        print_verbose=print,
        encoding=None,
        api_key="test_key",
        logging_obj=logging_obj,
        optional_params={"stream": True},
        litellm_params={},
        client=mock_httpx_client
    )
    
    # Iterate through chunks
    content = ""
    for chunk in response:
        content += chunk.choices[0].delta.content or ""
    
    assert "Hello" in content
    assert "world" in content
    assert "NLP Cloud" in content

@pytest.mark.asyncio
async def test_nlp_cloud_acompletion_streaming(mock_async_httpx_client):
    """Test the acompletion function with streaming (async)."""
    
    # Mock streaming response
    mock_response = mock_async_httpx_client.post.return_value
    
    async def async_iter(items):
        for item in items:
            yield item
            
    mock_response.aiter_lines.return_value = async_iter([
        b'{"generated_text": "Hello "}',
        b'{"generated_text": "world "}',
        b'{"generated_text": "from NLP Cloud [DONE]"}'
    ])
    
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.async_success_handler = AsyncMock()
    logging_obj._update_completion_start_time = MagicMock()
    
    model_response = ModelResponse()
    handler = NLPCloudChatHandler()
    
    response = await handler.acompletion(
        model="finetuned-llama-3-70b",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://api.nlpcloud.io/v1/",
        model_response=model_response,
        print_verbose=print,
        encoding=None,
        api_key="test_key",
        logging_obj=logging_obj,
        optional_params={"stream": True},
        litellm_params={},
        client=mock_async_httpx_client
    )
    
    # Iterate through chunks
    content = ""
    async for chunk in response:
        content += chunk.choices[0].delta.content or ""
    
    assert "Hello" in content
    assert "world" in content
    assert "NLP Cloud" in content

def test_nlp_cloud_integration_embedding_main_routing():
    """Test that litellm.embedding routes to NLP Cloud correctly."""
    
    with patch("litellm.llms.nlp_cloud.chat.handler.NLPCloudChatHandler.embedding") as mock_embedding_fn:
        # Mock the embedding function to return a dummy response
        mock_embedding_fn.return_value = litellm.EmbeddingResponse(data=[])
        
        litellm.embedding(
            model="nlp_cloud/paraphrase-multilingual-mpnet-base-v2",
            input=["test"],
            api_key="test_key"
        )
        
        mock_embedding_fn.assert_called_once()
        call_args = mock_embedding_fn.call_args
        # Note: args are (model, input, api_key, ...)
        # call_args[1] is kwargs. Check if passed as kwargs or args.
        # litellm.embedding passes some likely as args or kwargs depending on internal logic.
        # let's inspect call_args generally
        assert call_args.kwargs["model"] == "paraphrase-multilingual-mpnet-base-v2"
        assert call_args.kwargs["input"] == ["test"]
        assert call_args.kwargs["api_key"] == "test_key"

@pytest.mark.asyncio
async def test_nlp_cloud_integration_acompletion_main_routing():
    """Test that litellm.acompletion routes to NLP Cloud correctly."""
    
    with patch("litellm.llms.nlp_cloud.chat.handler.NLPCloudChatHandler.acompletion", new_callable=AsyncMock) as mock_acompletion_fn:
        # Mock the acompletion function
        mock_acompletion_fn.return_value = ModelResponse()
        
        await litellm.acompletion(
            model="nlp_cloud/finetuned-llama-3-70b",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test_key"
        )
        
        mock_acompletion_fn.assert_called_once()
        # Similarly verify kwargs
        assert mock_acompletion_fn.call_args.kwargs["model"] == "finetuned-llama-3-70b"
        assert mock_acompletion_fn.call_args.kwargs["api_key"] == "test_key"
