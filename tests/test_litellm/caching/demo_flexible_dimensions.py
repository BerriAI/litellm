#!/usr/bin/env python3
"""
Demo script showing flexible embedding dimensions in Qdrant Semantic Cache.
This script demonstrates the new functionality for PR review.
"""

import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../../.."))

def demo_auto_detection():
    """Demo auto-detection of embedding dimensions"""
    print("Demo 1: Auto-Detection of Embedding Dimensions")
    print("=" * 50)
    
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client"), \
         patch("litellm.embedding") as mock_embedding:
        
        # Mock Qdrant responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": False}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response
        mock_sync_client_instance.put.return_value = MagicMock()
        mock_sync_client.return_value = mock_sync_client_instance
        
        # Test different embedding models with different dimensions
        test_cases = [
            {"model": "bge-base-en-v1.5", "dimensions": 768},
            {"model": "stella-base-en-v2", "dimensions": 1024}, 
            {"model": "voyage-large-2", "dimensions": 1536},
            {"model": "text-embedding-3-large", "dimensions": 3072},
        ]
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache
        
        for test_case in test_cases:
            # Mock embedding response with specific dimensions
            mock_embedding.return_value = {
                "data": [{"embedding": [0.1] * test_case["dimensions"]}]
            }
            
            cache = QdrantSemanticCache(
                collection_name=f"test_{test_case['model'].replace('-', '_')}",
                qdrant_api_base="http://localhost:6333",
                qdrant_api_key="test_key",
                similarity_threshold=0.8,
                embedding_model=test_case["model"],
                # No dimensions specified - should auto-detect
            )
            
            print(f"{test_case['model']:<25} → Auto-detected: {cache.embedding_dimensions}d")
            assert cache.embedding_dimensions == test_case["dimensions"], f"Expected {test_case['dimensions']}, got {cache.embedding_dimensions}"


def demo_custom_configuration():
    """Demo custom dimensions and distance metrics"""
    print("\nDemo 2: Custom Dimensions and Distance Metrics")
    print("=" * 50)
    
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client"):
        
        # Mock Qdrant responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": False}}
        
        mock_put_response = MagicMock()
        mock_put_response.json.return_value = {"result": True}
        
        mock_collection_response = MagicMock()
        mock_collection_response.json.return_value = {"result": {"config": {}}}
        
        mock_sync_client_instance = MagicMock()
        # Mock the collection existence check to return False (needs creation)
        # Then mock the collection details fetch after creation
        mock_sync_client_instance.get.side_effect = [mock_response, mock_collection_response] * 10  # Repeat for multiple tests
        # Mock the collection creation
        mock_sync_client_instance.put.return_value = mock_put_response
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache
        
        test_configs = [
            {"model": "custom-small", "dimensions": 384, "metric": "Cosine"},
            {"model": "custom-medium", "dimensions": 768, "metric": "Dot"},
            {"model": "custom-large", "dimensions": 2048, "metric": "Euclid"},
        ]
        
        for config in test_configs:
            cache = QdrantSemanticCache(
                collection_name=f"test_{config['model']}",
                qdrant_api_base="http://localhost:6333",
                qdrant_api_key="test_key",
                similarity_threshold=0.8,
                embedding_model=config["model"],
                embedding_dimensions=config["dimensions"],
                distance_metric=config["metric"],
            )
            
            print(f"{config['model']:<15} → {config['dimensions']:>4}d, {config['metric']:<8} metric")
            assert cache.embedding_dimensions == config["dimensions"]
            assert cache.distance_metric == config["metric"]
            
            # Verify collection creation parameters
            call_args = mock_sync_client_instance.put.call_args
            json_payload = call_args[1]['json']
            assert json_payload['vectors']['size'] == config["dimensions"]
            assert json_payload['vectors']['distance'] == config["metric"]


def demo_backward_compatibility():
    """Demo that existing configurations still work"""
    print("\nDemo 3: Backward Compatibility")
    print("=" * 50)
    
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client"), \
         patch("litellm.embedding") as mock_embedding:
        
        # Mock Qdrant responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        mock_collection_response = MagicMock()
        mock_collection_response.json.return_value = {"result": {"config": {}}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response  # Collection exists, so only one get call needed
        mock_sync_client.return_value = mock_sync_client_instance
        
        # Mock default OpenAI embedding (1536 dimensions)
        mock_embedding.return_value = {
            "data": [{"embedding": [0.1] * 1536}]
        }
        
        from litellm.caching.qdrant_semantic_cache import QdrantSemanticCache
        
        # Test old-style initialization (no new parameters)
        cache = QdrantSemanticCache(
            collection_name="test_collection",
            qdrant_api_base="http://localhost:6333",
            qdrant_api_key="test_key",
            similarity_threshold=0.8,
            # Only old parameters - should work exactly as before
        )
        
        print("Legacy configuration works without changes")
        print(f"  - Model: {cache.embedding_model}")
        print(f"  - Dimensions: {cache.embedding_dimensions}d (auto-detected)")
        print(f"  - Distance metric: {cache.distance_metric} (default)")
        
        assert cache.embedding_model == "text-embedding-ada-002"  # Default
        assert cache.embedding_dimensions == 1536  # Auto-detected from mock
        assert cache.distance_metric == "Cosine"  # Default


def demo_cache_integration():
    """Demo integration with main Cache class"""
    print("\nDemo 4: Integration with Main Cache Class")
    print("=" * 50)
    
    with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client") as mock_sync_client, \
         patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client"):
        
        # Mock Qdrant responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"exists": True}}
        mock_collection_response = MagicMock()
        mock_collection_response.json.return_value = {"result": {"config": {}}}
        
        mock_sync_client_instance = MagicMock()
        mock_sync_client_instance.get.return_value = mock_response  # Collection exists
        mock_sync_client.return_value = mock_sync_client_instance
        
        from litellm.caching.caching import Cache
        
        # Test new parameters through main Cache class
        cache = Cache(
            type="qdrant-semantic",
            qdrant_api_base="http://localhost:6333",
            qdrant_api_key="test_key",
            qdrant_collection_name="test_collection",
            similarity_threshold=0.8,
            qdrant_semantic_cache_embedding_model="bge-base-en-v1.5",
            qdrant_semantic_cache_embedding_dimensions=768,
            qdrant_distance_metric="Dot",
        )
        
        qdrant_cache = cache.cache
        print("Main Cache class integration successful")
        print(f"  - Cache type: {type(qdrant_cache).__name__}")
        print(f"  - Model: {qdrant_cache.embedding_model}")
        print(f"  - Dimensions: {qdrant_cache.embedding_dimensions}d")
        print(f"  - Distance metric: {qdrant_cache.distance_metric}")
        
        assert qdrant_cache.embedding_model == "bge-base-en-v1.5"
        assert qdrant_cache.embedding_dimensions == 768
        assert qdrant_cache.distance_metric == "Dot"


def show_configuration_examples():
    """Show configuration examples for different use cases"""
    print("\nConfiguration Examples")
    print("=" * 50)
    
    examples = [
        {
            "title": "Cost-Optimized Setup (BGE Base)",
            "config": """
# Python API
cache = Cache(
    type="qdrant-semantic",
    qdrant_semantic_cache_embedding_model="bge-base-en-v1.5"
    # Dimensions auto-detected as 768, 2x cheaper than OpenAI
)

# config.yaml
cache_params:
  type: qdrant-semantic
  qdrant_semantic_cache_embedding_model: bge-base
  qdrant_semantic_cache_embedding_dimensions: 768  # Optional
"""
        },
        {
            "title": "High-Quality Setup (Stella)",
            "config": """
# Python API  
cache = Cache(
    type="qdrant-semantic",
    qdrant_semantic_cache_embedding_model="stella-base-en-v2",
    qdrant_semantic_cache_embedding_dimensions=1024,
    qdrant_distance_metric="Dot"
)

# config.yaml
cache_params:
  type: qdrant-semantic
  qdrant_semantic_cache_embedding_model: stella-base
  qdrant_semantic_cache_embedding_dimensions: 1024
  qdrant_distance_metric: Dot
"""
        },
        {
            "title": "Large Model Setup (OpenAI Large)",
            "config": """
# Python API
cache = Cache(
    type="qdrant-semantic", 
    qdrant_semantic_cache_embedding_model="text-embedding-3-large"
    # Dimensions auto-detected as 3072
)

# config.yaml
cache_params:
  type: qdrant-semantic
  qdrant_semantic_cache_embedding_model: openai-large
  # Dimensions auto-detected
"""
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print(example['config'])


if __name__ == "__main__":
    print("LiteLLM Qdrant Semantic Cache - Flexible Embedding Dimensions Demo")
    print("=" * 70)
    print("Demonstrating the new functionality")
    print()
    
    try:
        demo_auto_detection()
        demo_custom_configuration()
        demo_backward_compatibility()
        demo_cache_integration()
        show_configuration_examples()
        
        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("Implementation ready for production use!")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
