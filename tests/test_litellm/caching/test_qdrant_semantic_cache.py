import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def test_qdrant_semantic_cache_initialization(monkeypatch):
    """
    Test QDRANT semantic cache initialization with proper parameters.
    Verifies that the cache is initialized correctly with given configuration.
    """
    # Mock the httpx clients and API calls
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize the cache with similarity threshold
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
        )

        # Verify the cache was initialized with correct parameters
        assert qdrant_cache.collection_name == "test_collection"
        assert qdrant_cache.qdrant_api_base == "http://test.qdrant.local"
        assert qdrant_cache.qdrant_api_key == "test_key"
        assert qdrant_cache.similarity_threshold == 0.8

    # Test initialization with missing similarity_threshold
    with pytest.raises(Exception, match="similarity_threshold must be provided"):
        QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
        )


def test_qdrant_semantic_cache_get_cache_hit():
    """
    Test QDRANT semantic cache get method when there's a cache hit.
    Verifies that cached results are properly retrieved and parsed.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize cache
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
        )

        # Mock a cache hit result from search API
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "result": [
                {
                    "payload": {
                        "text": "What is the capital of France?",  # Original prompt
                        "response": '{"id": "test-123", "choices": [{"message": {"content": "Paris is the capital of France."}}]}'
                    },
                    "score": 0.9
                }
            ]
        }
        qdrant_cache.sync_client.post = MagicMock(return_value=mock_search_response)

        # Mock the embedding function
        with patch(
            "litellm.embedding", 
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        ):
            # Test get_cache with a message
            result = qdrant_cache.get_cache(
                key="test_key", 
                messages=[{"content": "What is the capital of France?"}]
            )

            # Verify result is properly parsed
            expected_result = {
                "id": "test-123", 
                "choices": [{"message": {"content": "Paris is the capital of France."}}]
            }
            assert result == expected_result

            # Verify search was called
            qdrant_cache.sync_client.post.assert_called()


def test_qdrant_semantic_cache_get_cache_miss():
    """
    Test QDRANT semantic cache get method when there's a cache miss.
    Verifies that None is returned when no similar cached results are found.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize cache
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
        )

        # Mock a cache miss (no results)
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {"result": []}
        qdrant_cache.sync_client.post = MagicMock(return_value=mock_search_response)

        # Mock the embedding function
        with patch(
            "litellm.embedding", 
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        ):
            # Test get_cache with a message
            result = qdrant_cache.get_cache(
                key="test_key", 
                messages=[{"content": "What is the capital of Spain?"}]
            )

            # Verify None is returned for cache miss
            assert result is None

            # Verify search was called
            qdrant_cache.sync_client.post.assert_called()


@pytest.mark.asyncio
async def test_qdrant_semantic_cache_async_get_cache_hit():
    """
    Test QDRANT semantic cache async get method when there's a cache hit.
    Verifies that cached results are properly retrieved and parsed asynchronously.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        # Mock async client
        mock_async_client_instance = AsyncMock()
        mock_async_client.return_value = mock_async_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize cache
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
        )

        # Mock a cache hit result from async search API
        # Note: .json() should be sync even for async responses
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "result": [
                {
                    "payload": {
                        "text": "What is the capital of Spain?",  # Original prompt
                        "response": '{"id": "test-456", "choices": [{"message": {"content": "Madrid is the capital of Spain."}}]}'
                    },
                    "score": 0.85
                }
            ]
        }
        qdrant_cache.async_client.post = AsyncMock(return_value=mock_search_response)

        # Mock the async embedding function
        with patch(
            "litellm.aembedding", 
            return_value={"data": [{"embedding": [0.4, 0.5, 0.6]}]}
        ):
            # Test async_get_cache with a message
            result = await qdrant_cache.async_get_cache(
                key="test_key",
                messages=[{"content": "What is the capital of Spain?"}],
                metadata={},
            )

            # Verify result is properly parsed
            expected_result = {
                "id": "test-456", 
                "choices": [{"message": {"content": "Madrid is the capital of Spain."}}]
            }
            assert result == expected_result

            # Verify async search was called
            qdrant_cache.async_client.post.assert_called()


@pytest.mark.asyncio
async def test_qdrant_semantic_cache_async_get_cache_miss():
    """
    Test QDRANT semantic cache async get method when there's a cache miss.
    Verifies that None is returned when no similar cached results are found.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        # Mock async client
        mock_async_client_instance = AsyncMock()
        mock_async_client.return_value = mock_async_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize cache
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local", 
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
        )

        # Mock a cache miss (no results)
        mock_search_response = MagicMock()  # Note: .json() should be sync
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {"result": []}
        qdrant_cache.async_client.post = AsyncMock(return_value=mock_search_response)

        # Mock the async embedding function
        with patch(
            "litellm.aembedding", 
            return_value={"data": [{"embedding": [0.7, 0.8, 0.9]}]}
        ):
            # Test async_get_cache with a message
            result = await qdrant_cache.async_get_cache(
                key="test_key",
                messages=[{"content": "What is the capital of Italy?"}],
                metadata={},
            )

            # Verify None is returned for cache miss
            assert result is None

            # Verify async search was called
            qdrant_cache.async_client.post.assert_called()


def test_qdrant_semantic_cache_set_cache():
    """
    Test QDRANT semantic cache set method.
    Verifies that responses are properly stored in the cache.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize cache
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
        )

        # Mock the upsert method
        mock_upsert_response = MagicMock()
        mock_upsert_response.status_code = 200
        qdrant_cache.sync_client.put = MagicMock(return_value=mock_upsert_response)

        # Mock response to cache
        response_to_cache = {
            "id": "test-789",
            "choices": [{"message": {"content": "Rome is the capital of Italy."}}]
        }

        # Mock the embedding function
        with patch(
            "litellm.embedding", 
            return_value={"data": [{"embedding": [0.1, 0.1, 0.1]}]}
        ):
            # Test set_cache
            qdrant_cache.set_cache(
                key="test_key",
                value=response_to_cache,
                messages=[{"content": "What is the capital of Italy?"}]
            )

            # Verify upsert was called
            qdrant_cache.sync_client.put.assert_called()


@pytest.mark.asyncio
async def test_qdrant_semantic_cache_async_set_cache():
    """
    Test QDRANT semantic cache async set method.
    Verifies that responses are properly stored in the cache asynchronously.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:
        
        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        # Mock async client
        mock_async_client_instance = AsyncMock()
        mock_async_client.return_value = mock_async_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize cache
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key", 
            similarity_threshold=0.8,
        )

        # Mock the async upsert method
        mock_upsert_response = MagicMock()  # Note: .json() should be sync
        mock_upsert_response.status_code = 200
        qdrant_cache.async_client.put = AsyncMock(return_value=mock_upsert_response)

        # Mock response to cache
        response_to_cache = {
            "id": "test-999",
            "choices": [{"message": {"content": "Berlin is the capital of Germany."}}]
        }

        # Mock the async embedding function
        with patch(
            "litellm.aembedding", 
            return_value={"data": [{"embedding": [0.2, 0.2, 0.2]}]}
        ):
            # Test async_set_cache
            await qdrant_cache.async_set_cache(
                key="test_key",
                value=response_to_cache,
                messages=[{"content": "What is the capital of Germany?"}],
                metadata={}
            )

            # Verify async upsert was called
            qdrant_cache.async_client.put.assert_called() 

def test_qdrant_semantic_cache_custom_vector_size():
    """
    Test that QdrantSemanticCache uses a custom vector_size when creating a new collection.
    Verifies that the vector size passed to the constructor is used in the Qdrant collection
    creation payload instead of the default 1536.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:

        # Mock the collection does NOT exist (so it will be created)
        mock_exists_response = MagicMock()
        mock_exists_response.status_code = 200
        mock_exists_response.json.return_value = {"result": {"exists": False}}

        # Mock the collection creation response
        mock_create_response = MagicMock()
        mock_create_response.status_code = 200
        mock_create_response.json.return_value = {"result": True}

        # Mock the collection details response after creation
        mock_details_response = MagicMock()
        mock_details_response.status_code = 200
        mock_details_response.json.return_value = {"result": {"status": "ok"}}

        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.side_effect = [mock_exists_response, mock_details_response]
        mock_sync_client_instance.put.return_value = mock_create_response
        mock_sync_client.return_value = mock_sync_client_instance

        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize with custom vector_size of 768
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection_768",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
            vector_size=768,
        )

        # Verify the vector_size attribute is set correctly
        assert qdrant_cache.vector_size == 768

        # Verify the PUT call to create the collection used vector_size=768
        put_call = mock_sync_client_instance.put.call_args
        assert put_call is not None
        create_payload = put_call.kwargs.get("json") or put_call[1].get("json")
        assert create_payload["vectors"]["size"] == 768
        assert create_payload["vectors"]["distance"] == "Cosine"


def test_qdrant_semantic_cache_default_vector_size():
    """
    Test that QdrantSemanticCache defaults to QDRANT_VECTOR_SIZE (1536) when vector_size
    is not provided, and stores it as self.vector_size.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:

        # Mock the collection exists check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}

        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client.return_value = mock_sync_client_instance

        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache
        from litellm.constants import QDRANT_VECTOR_SIZE

        # Initialize without vector_size
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
        )

        # Verify it falls back to the default QDRANT_VECTOR_SIZE constant
        assert qdrant_cache.vector_size == QDRANT_VECTOR_SIZE


def test_qdrant_semantic_cache_large_vector_size():
    """
    Test that QdrantSemanticCache supports large embedding dimensions (e.g. 4096, 8192)
    for models like Stella, bge-en-icl, etc.
    """
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_async_client:

        # Mock the collection does NOT exist (so it will be created)
        mock_exists_response = MagicMock()
        mock_exists_response.status_code = 200
        mock_exists_response.json.return_value = {"result": {"exists": False}}

        mock_create_response = MagicMock()
        mock_create_response.status_code = 200
        mock_create_response.json.return_value = {"result": True}

        mock_details_response = MagicMock()
        mock_details_response.status_code = 200
        mock_details_response.json.return_value = {"result": {"status": "ok"}}

        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.side_effect = [mock_exists_response, mock_details_response]
        mock_sync_client_instance.put.return_value = mock_create_response
        mock_sync_client.return_value = mock_sync_client_instance

        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache

        # Initialize with a large vector_size of 4096
        qdrant_cache = QdrantSemanticCache(
            collection_name="test_collection_4096",
            qdrant_api_base="http://test.qdrant.local",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
            vector_size=4096,
        )

        assert qdrant_cache.vector_size == 4096

        # Verify the collection was created with 4096
        put_call = mock_sync_client_instance.put.call_args
        create_payload = put_call.kwargs.get("json") or put_call[1].get("json")
        assert create_payload["vectors"]["size"] == 4096
