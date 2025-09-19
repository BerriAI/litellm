"""
Example demonstrating flexible embedding dimensions in Qdrant Semantic Cache.

This example shows how to use different embedding models with various dimensions:
- OpenAI models (1536 dimensions)
- Smaller models like bge-base (768 dimensions) 
- Larger models like stella (1024+ dimensions)
- Custom embedding models with any dimension size
"""

import os
import litellm
from litellm.caching import Cache

# Example 1: Auto-detect dimensions for different models
def example_auto_detect_dimensions():
    """
    Example showing auto-detection of embedding dimensions for different models.
    The cache will automatically detect the dimensions by making a test embedding call.
    """
    print("=== Example 1: Auto-detect Embedding Dimensions ===")
    
    # Different embedding models with different dimensions
    models_to_test = [
        "openai/text-embedding-3-small",  # 1536 dimensions
        "openai/text-embedding-3-large",  # 3072 dimensions  
        "cohere/embed-english-v3.0",      # 1024 dimensions
        "voyage/voyage-large-2",          # 1536 dimensions
    ]
    
    for model in models_to_test:
        try:
            # Cache will auto-detect dimensions for each model
            cache = Cache(
                type="qdrant-semantic",
                qdrant_api_base=os.environ.get("QDRANT_API_BASE", "http://localhost:6333"),
                qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
                qdrant_collection_name=f"cache_{model.replace('/', '_').replace('-', '_')}",
                similarity_threshold=0.8,
                qdrant_semantic_cache_embedding_model=model,
                # No dimensions specified - will auto-detect
            )
            
            print(f"✓ {model}: Auto-detected dimensions = {cache.cache.embedding_dimensions}")
            
        except Exception as e:
            print(f"✗ {model}: Failed to initialize - {e}")


# Example 2: Specify custom dimensions and distance metrics
def example_custom_dimensions():
    """
    Example showing how to specify custom dimensions and distance metrics.
    Useful when you know the exact dimensions or want to use specific distance metrics.
    """
    print("\n=== Example 2: Custom Dimensions and Distance Metrics ===")
    
    configs = [
        {
            "model": "bge-base-en-v1.5",
            "dimensions": 768,
            "distance": "Cosine",
            "description": "BGE Base model with cosine similarity"
        },
        {
            "model": "stella-base-en-v2", 
            "dimensions": 1024,
            "distance": "Dot",
            "description": "Stella model with dot product similarity"
        },
        {
            "model": "custom-embedding-model",
            "dimensions": 384,
            "distance": "Euclid",
            "description": "Small custom model with Euclidean distance"
        }
    ]
    
    for config in configs:
        try:
            cache = Cache(
                type="qdrant-semantic",
                qdrant_api_base=os.environ.get("QDRANT_API_BASE", "http://localhost:6333"),
                qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
                qdrant_collection_name=f"cache_custom_{config['model'].replace('-', '_')}",
                similarity_threshold=0.8,
                qdrant_semantic_cache_embedding_model=config["model"],
                qdrant_semantic_cache_embedding_dimensions=config["dimensions"],
                qdrant_distance_metric=config["distance"],
            )
            
            print(f"✓ {config['description']}")
            print(f"  Model: {config['model']}")
            print(f"  Dimensions: {cache.cache.embedding_dimensions}")
            print(f"  Distance: {cache.cache.distance_metric}")
            
        except Exception as e:
            print(f"✗ {config['description']}: Failed - {e}")


# Example 3: Using with LiteLLM Proxy configuration
def example_proxy_config():
    """
    Example showing how to configure the new parameters in proxy config.yaml
    """
    print("\n=== Example 3: Proxy Configuration ===")
    
    config_yaml = """
model_list:
  - model_name: bge-base
    litellm_params:
      model: bge-base-en-v1.5
      
  - model_name: stella-large  
    litellm_params:
      model: stella-large-en-v2
      
  - model_name: voyage-large
    litellm_params:
      model: voyage/voyage-large-2

litellm_settings:
  set_verbose: true
  cache: True
  cache_params:
    type: qdrant-semantic
    
    # Embedding model configuration
    qdrant_semantic_cache_embedding_model: bge-base
    qdrant_semantic_cache_embedding_dimensions: 768  # Optional: auto-detected if not provided
    qdrant_distance_metric: Cosine                   # Optional: defaults to Cosine
    
    # Qdrant configuration  
    qdrant_collection_name: litellm-semantic-cache
    qdrant_quantization_config: binary
    similarity_threshold: 0.8
"""
    
    print("Configuration for config.yaml:")
    print(config_yaml)


# Example 4: Performance comparison between models
def example_model_comparison():
    """
    Example showing how different embedding models might perform for semantic caching.
    """
    print("\n=== Example 4: Model Performance Comparison ===")
    
    models = [
        {
            "name": "text-embedding-3-small", 
            "dimensions": 1536,
            "cost": "High",
            "quality": "Good"
        },
        {
            "name": "bge-base-en-v1.5",
            "dimensions": 768, 
            "cost": "Low",
            "quality": "Good"
        },
        {
            "name": "stella-base-en-v2",
            "dimensions": 1024,
            "cost": "Medium", 
            "quality": "Very Good"
        },
        {
            "name": "voyage-large-2",
            "dimensions": 1536,
            "cost": "Medium",
            "quality": "Excellent"
        }
    ]
    
    print("Model Comparison for Semantic Caching:")
    print("=" * 60)
    print(f"{'Model':<25} {'Dimensions':<12} {'Cost':<8} {'Quality'}")
    print("-" * 60)
    
    for model in models:
        print(f"{model['name']:<25} {model['dimensions']:<12} {model['cost']:<8} {model['quality']}")
    
    print("\nRecommendations:")
    print("- For cost optimization: Use bge-base-en-v1.5 (768d)")
    print("- For best quality: Use stella-base-en-v2 (1024d) or voyage-large-2 (1536d)")
    print("- For compatibility: Use text-embedding-3-small (1536d)")


if __name__ == "__main__":
    print("LiteLLM Qdrant Semantic Cache - Flexible Embedding Dimensions Examples")
    print("=" * 70)
    
    # Run examples
    example_auto_detect_dimensions()
    example_custom_dimensions() 
    example_proxy_config()
    example_model_comparison()
    
    print("\n" + "=" * 70)
    print("Examples completed! Check the documentation for more details.")
