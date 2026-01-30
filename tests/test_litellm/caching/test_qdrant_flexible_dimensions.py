import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def test_qdrant_semantic_cache_auto_detect_dimensions():
    """
    Test QDRANT semantic cache auto-detection of embedding dimensions.
    """
    # Mock the httpx clients and API calls
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client, \
         patch("litellm.embedding") as mock_embedding:
        
        # Mock the embedding response for dimension detection
        mock_embedding.return_value = {
            "data": [{"embedding": [0.1] * 768}],  # 768-dimensional embedding
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": False}}
        
        # Mock collection creation response
        mock_create_response = MagicMock()
        mock_create_response.json.return_value = {"result": True}
        
        # Mock collection details response
        mock_details_response = MagicMock()
        mock_details_response.json.return_value = {"result": {"config": {"vectors": {"size": 768}}}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.side_effect = [mock_response, mock_details_response]
        mock_sync_client_instance.put.return_value = mock_create_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize the cache without specifying dimensions (should auto-detect)
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
            embedding_model="custom/768d-model",  # Custom model with 768 dimensions
        )

        # Verify the cache was initialized with auto-detected dimensions
        assert qdrant_cache.embedding_dimensions == 768
        assert qdrant_cache.distance_metric == "Cosine"  # Default value
        
        # Verify embedding was called for dimension detection
        mock_embedding.assert_called_with(
            model="custom/768d-model",
            input="Hello World",
            cache={"no-store": True, "no-cache": True},
        )


def test_qdrant_semantic_cache_custom_dimensions_and_metric():
    """
    Test QDRANT semantic cache with custom dimensions and distance metric.
    """
    # Mock the httpx clients and API calls
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": False}}
        
        # Mock collection creation response
        mock_create_response = MagicMock()
        mock_create_response.json.return_value = {"result": True}
        
        # Mock collection details response
        mock_details_response = MagicMock()
        mock_details_response.json.return_value = {"result": {"config": {"vectors": {"size": 4096}}}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.side_effect = [mock_response, mock_details_response]
        mock_sync_client_instance.put.return_value = mock_create_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize the cache with custom dimensions and distance metric
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
            embedding_model="custom/4096d-model",
            embedding_dimensions=4096,  # Manually specified
            distance_metric="Dot",  # Custom distance metric
        )

        # Verify the cache was initialized with custom parameters
        assert qdrant_cache.embedding_dimensions == 4096
        assert qdrant_cache.distance_metric == "Dot"
        
        # Verify the collection was created with correct parameters
        create_call_args = mock_sync_client_instance.put.call_args
        assert create_call_args[1]['json']['vectors']['size'] == 4096
        assert create_call_args[1]['json']['vectors']['distance'] == "Dot"


def test_qdrant_semantic_cache_fallback_dimensions():
    """
    Test QDRANT semantic cache fallback to default dimensions when auto-detection fails.
    """
    # Mock the httpx clients and API calls
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client, \
         patch("litellm.embedding") as mock_embedding:
        
        # Mock the embedding to raise an exception (simulating failure)
        mock_embedding.side_effect = Exception("Model not found")
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": False}}
        
        # Mock collection creation response
        mock_create_response = MagicMock()
        mock_create_response.json.return_value = {"result": True}
        
        # Mock collection details response
        mock_details_response = MagicMock()
        mock_details_response.json.return_value = {"result": {"config": {"vectors": {"size": 1536}}}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.side_effect = [mock_response, mock_details_response]
        mock_sync_client_instance.put.return_value = mock_create_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize the cache (should fallback to default dimensions)
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
            embedding_model="unknown/model",
        )

        # Verify the cache fell back to default dimensions
        assert qdrant_cache.embedding_dimensions == 1536  # Default from QDRANT_VECTOR_SIZE
        assert qdrant_cache.distance_metric == "Cosine"


if __name__ == "__main__":
    test_qdrant_semantic_cache_auto_detect_dimensions()
    test_qdrant_semantic_cache_custom_dimensions_and_metric()
    test_qdrant_semantic_cache_fallback_dimensions()
    print("All tests passed!")
