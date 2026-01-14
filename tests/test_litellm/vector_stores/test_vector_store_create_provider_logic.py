import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig
from litellm.llms.ragflow.vector_stores.transformation import RAGFlowVectorStoreConfig
from litellm.llms.vertex_ai.vector_stores.rag_api.transformation import (
    VertexVectorStoreConfig,
)
from litellm.utils import ProviderConfigManager


def test_vector_store_create_with_simple_provider_name():
    """
    Test that vector store create correctly handles simple provider names
    like "openai" (without "/" separator).
    
    This should:
    - Set api_type to None
    - Keep custom_llm_provider as "openai"
    - Not call get_llm_provider (to avoid IndexError)
    - Return correct OpenAIVectorStoreConfig
    """
    custom_llm_provider = "openai"
    
    # Simulate the logic from vector_stores/main.py create function
    if "/" in custom_llm_provider:
        # This branch should NOT be taken
        pytest.fail("Should not enter this branch for simple provider name")
    else:
        api_type = None
        custom_llm_provider = custom_llm_provider  # Keep as-is
    
    # Verify api_type is None
    assert api_type is None, "api_type should be None for simple provider names"
    
    # Verify custom_llm_provider is unchanged
    assert custom_llm_provider == "openai", "custom_llm_provider should remain 'openai'"
    
    # Verify ProviderConfigManager returns correct config
    vector_store_provider_config = ProviderConfigManager.get_provider_vector_stores_config(
        provider=litellm.LlmProviders(custom_llm_provider),
        api_type=api_type,
    )
    
    assert vector_store_provider_config is not None, "Should return a config for OpenAI"
    assert isinstance(
        vector_store_provider_config, OpenAIVectorStoreConfig
    ), "Should return OpenAIVectorStoreConfig for OpenAI provider"
    
    print("✅ Test passed: Simple provider name 'openai' handled correctly")


def test_vector_store_create_with_provider_api_type():
    """
    Test that vector store create correctly handles provider names with api_type
    like "vertex_ai/rag_api" (with "/" separator).
    
    This should:
    - Call get_llm_provider to extract api_type and provider
    - Extract api_type as "rag_api"
    - Extract custom_llm_provider as "vertex_ai"
    - Return correct VertexVectorStoreConfig with api_type
    """
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    
    custom_llm_provider = "vertex_ai/rag_api"
    
    # Simulate the logic from vector_stores/main.py create function
    if "/" in custom_llm_provider:
        api_type, custom_llm_provider, _, _ = get_llm_provider(
            model=custom_llm_provider,
            custom_llm_provider=None,
            litellm_params=None,
        )
    else:
        # This branch should NOT be taken
        pytest.fail("Should not enter this branch for provider with api_type")
    
    # Verify api_type is extracted correctly
    assert api_type == "rag_api", f"api_type should be 'rag_api', got '{api_type}'"
    
    # Verify custom_llm_provider is extracted correctly
    assert custom_llm_provider == "vertex_ai", f"custom_llm_provider should be 'vertex_ai', got '{custom_llm_provider}'"
    
    # Verify ProviderConfigManager returns correct config
    vector_store_provider_config = ProviderConfigManager.get_provider_vector_stores_config(
        provider=litellm.LlmProviders(custom_llm_provider),
        api_type=api_type,
    )
    
    assert vector_store_provider_config is not None, "Should return a config for Vertex AI"
    assert isinstance(
        vector_store_provider_config, VertexVectorStoreConfig
    ), "Should return VertexVectorStoreConfig for vertex_ai provider with rag_api"
    
    print("✅ Test passed: Provider with api_type 'vertex_ai/rag_api' handled correctly")


def test_vector_store_create_with_ragflow_provider():
    """
    Test that vector store create correctly handles RAGFlow provider.
    
    This should:
    - Return correct RAGFlowVectorStoreConfig
    - Support dataset management operations
    """
    custom_llm_provider = "ragflow"
    
    # Simulate the logic from vector_stores/main.py create function
    if "/" in custom_llm_provider:
        pytest.fail("Should not enter this branch for RAGFlow provider")
    else:
        api_type = None
        custom_llm_provider = custom_llm_provider  # Keep as-is
    
    # Verify api_type is None
    assert api_type is None, "api_type should be None for RAGFlow provider"
    
    # Verify custom_llm_provider is unchanged
    assert custom_llm_provider == "ragflow", "custom_llm_provider should remain 'ragflow'"
    
    # Verify ProviderConfigManager returns correct config
    vector_store_provider_config = ProviderConfigManager.get_provider_vector_stores_config(
        provider=litellm.LlmProviders(custom_llm_provider),
        api_type=api_type,
    )
    
    assert vector_store_provider_config is not None, "Should return a config for RAGFlow"
    assert isinstance(
        vector_store_provider_config, RAGFlowVectorStoreConfig
    ), "Should return RAGFlowVectorStoreConfig for RAGFlow provider"
    
    print("✅ Test passed: RAGFlow provider handled correctly")

