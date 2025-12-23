import json
import os
import sys
from unittest.mock import Mock, patch
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

isaacus_embedding_response = {
    "embeddings": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}],
    "model": "kanon-2-embedder",
    "usage": {
        "input_tokens": 5,
    }
}


@pytest.mark.parametrize(
    "model,input_data,expected_request_field",
    [
        ("isaacus/kanon-2-embedder", "Hello world", "texts"),
        ("kanon-2-embedder", ["Hello", "World"], "texts"),
    ],
)
def test_isaacus_embedding_models(model, input_data, expected_request_field):
    """Test embedding functionality for Isaacus models with different input types"""
    litellm.set_verbose = True
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(isaacus_embedding_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        try:
            response = litellm.embedding(
                model=model,
                input=input_data,
                client=client,
            )

            # Verify response structure
            assert isinstance(response, litellm.EmbeddingResponse)
            assert isinstance(response.data[0]['embedding'], list)
            assert len(response.data[0]['embedding']) == 5  # Based on mock response

            # Fetch request body
            request_data = json.loads(mock_post.call_args.kwargs["data"])

            # Verify the request uses 'texts' instead of 'input'
            assert expected_request_field in request_data, f"Request should have '{expected_request_field}' field"
            assert "input" not in request_data, "Request should not have 'input' field (should be transformed to 'texts')"

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


def test_isaacus_embedding_with_optional_params():
    """Test that optional Isaacus-specific parameters are properly included"""
    litellm.set_verbose = True
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(isaacus_embedding_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        try:
            response = litellm.embedding(
                model="kanon-2-embedder",
                input="Hello world",
                client=client,
                task="retrieval/query",
                dimensions=768,
                overflow_strategy="truncate",
            )

            # Verify response
            assert isinstance(response, litellm.EmbeddingResponse)

            # Fetch request body
            request_data = json.loads(mock_post.call_args.kwargs["data"])

            # Verify optional parameters are included
            assert request_data.get("task") == "retrieval/query"
            assert request_data.get("dimensions") == 768
            assert request_data.get("overflow_strategy") == "truncate"

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


def test_e2e_isaacus_embedding_basic():
    """
    Test basic text embedding with Isaacus kanon-2-embedder.
    Validates that the integration properly transforms requests and responses.
    """
    print("Testing basic Isaacus embedding...")

    # Set API key from environment
    api_key = os.environ.get("ISAACUS_API_KEY")
    if not api_key:
        pytest.skip("ISAACUS_API_KEY not set in environment")

    litellm._turn_on_debug()
    response = litellm.embedding(
        model="kanon-2-embedder",
        input="Hello world from LiteLLM!",
        api_key=api_key,
    )

    # Validate response structure
    assert isinstance(response, litellm.EmbeddingResponse), "Response should be EmbeddingResponse type"
    assert hasattr(response, 'data'), "Response should have 'data' attribute"
    assert len(response.data) > 0, "Response data should not be empty"

    # Validate first embedding
    embedding_obj = response.data[0]
    assert 'embedding' in embedding_obj, "Embedding object should have 'embedding' key"
    assert isinstance(embedding_obj['embedding'], list), "Embedding should be a list of floats"
    assert len(embedding_obj['embedding']) > 0, "Embedding vector should not be empty"
    assert all(isinstance(x, (int, float)) for x in embedding_obj['embedding']), "All embedding values should be numeric"

    # Validate embedding properties
    assert embedding_obj['index'] == 0, "First embedding should have index 0"
    assert embedding_obj['object'] == "embedding", "Embedding object type should be 'embedding'"

    # Validate usage information
    assert hasattr(response, 'usage'), "Response should have usage information"
    assert response.usage is not None, "Usage should not be None"
    assert response.usage.total_tokens >= 0, "Total tokens should be non-negative"

    print(f"Basic embedding successful! Vector size: {len(embedding_obj['embedding'])}")


def test_e2e_isaacus_embedding_batch():
    """
    Test batch text embedding with Isaacus.
    Validates that multiple inputs are properly handled.
    """
    print("Testing batch Isaacus embedding...")

    # Set API key from environment
    api_key = os.environ.get("ISAACUS_API_KEY")
    if not api_key:
        pytest.skip("ISAACUS_API_KEY not set in environment")

    litellm._turn_on_debug()
    inputs = [
        "First text to embed",
        "Second text to embed",
        "Third text to embed",
    ]

    response = litellm.embedding(
        model="kanon-2-embedder",
        input=inputs,
        api_key=api_key,
    )

    # Validate response structure
    assert isinstance(response, litellm.EmbeddingResponse), "Response should be EmbeddingResponse type"
    assert len(response.data) == len(inputs), f"Should have {len(inputs)} embeddings"

    # Validate each embedding
    for i, embedding_obj in enumerate(response.data):
        assert 'embedding' in embedding_obj, f"Embedding {i} should have 'embedding' key"
        assert isinstance(embedding_obj['embedding'], list), f"Embedding {i} should be a list"
        assert len(embedding_obj['embedding']) > 0, f"Embedding {i} should not be empty"
        assert embedding_obj['index'] == i, f"Embedding {i} should have correct index"

    # All embeddings should have the same dimension
    dimensions = [len(emb['embedding']) for emb in response.data]
    assert len(set(dimensions)) == 1, "All embeddings should have the same dimension"

    print(f"Batch embedding successful! Processed {len(inputs)} texts, vector size: {dimensions[0]}")


def test_e2e_isaacus_embedding_with_optional_params():
    """
    Test Isaacus embedding with optional parameters (task, dimensions, overflow_strategy).
    Validates that Isaacus-specific parameters are properly handled.
    """
    print("Testing Isaacus embedding with optional parameters...")

    # Set API key from environment
    api_key = os.environ.get("ISAACUS_API_KEY")
    if not api_key:
        pytest.skip("ISAACUS_API_KEY not set in environment")

    litellm._turn_on_debug()
    response = litellm.embedding(
        model="kanon-2-embedder",
        input="Hello world with optional params!",
        api_key=api_key,
        task="retrieval/query",
        dimensions=768,
        overflow_strategy="drop_end",
    )

    # Validate response structure
    assert isinstance(response, litellm.EmbeddingResponse), "Response should be EmbeddingResponse type"
    assert len(response.data) == 1, "Should have one embedding"

    # Validate embedding
    embedding_obj = response.data[0]
    assert isinstance(embedding_obj['embedding'], list), "Embedding should be a list"

    # Validate that the requested dimensions parameter is respected (if API supports it)
    # Note: This may depend on the model's capabilities
    print(f"Optional params embedding successful! Vector size: {len(embedding_obj['embedding'])}")


@pytest.mark.asyncio
async def test_e2e_isaacus_async_embedding():
    """
    Test async embedding with Isaacus.
    Validates that async embedding calls work correctly.
    """
    print("Testing async Isaacus embedding...")

    # Set API key from environment
    api_key = os.environ.get("ISAACUS_API_KEY")
    if not api_key:
        pytest.skip("ISAACUS_API_KEY not set in environment")

    litellm._turn_on_debug()
    response = await litellm.aembedding(
        model="kanon-2-embedder",
        input="Hello world from async!",
        api_key=api_key,
    )

    # Validate response structure
    assert isinstance(response, litellm.EmbeddingResponse), "Response should be EmbeddingResponse type"
    assert len(response.data) == 1, "Should have one embedding"

    # Validate embedding
    embedding_obj = response.data[0]
    assert isinstance(embedding_obj['embedding'], list), "Embedding should be a list"
    assert len(embedding_obj['embedding']) > 0, "Embedding should not be empty"

    print(f"Async embedding successful! Vector size: {len(embedding_obj['embedding'])}")
