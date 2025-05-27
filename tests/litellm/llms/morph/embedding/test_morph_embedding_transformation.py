"""
Unit tests for Morph embedding configuration.

These tests validate the MorphEmbeddingConfig class which extends OpenAILikeEmbeddingHandler.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.morph.embedding.transformation import MorphEmbeddingConfig


class TestMorphEmbeddingConfig:
    """Test class for MorphEmbeddingConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = MorphEmbeddingConfig()
        headers = {}
        api_key = "fake-morph-key"

        result = config.validate_environment(
            headers=headers,
            model="morph/morph-embedding-v2",
            optional_params={},
            api_key=api_key,
            api_base="https://api.morphllm.com/v1",
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_get_openai_compatible_provider_info(self):
        """Test the _get_openai_compatible_provider_info method"""
        config = MorphEmbeddingConfig()
        api_key = "fake-morph-key"

        result = config._get_openai_compatible_provider_info(
            api_base=None,
            api_key=api_key,
        )

        # Verify correct API base is returned
        assert result[0] == "https://api.morphllm.com/v1"
        assert result[1] == api_key

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = MorphEmbeddingConfig()
        
        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="morph/morph-embedding-v2",
                optional_params={},
                api_key=None,
                api_base="https://api.morphllm.com/v1",
            )

        assert "Morph API key is required" in str(excinfo.value)

    def test_inheritance(self):
        """Test proper inheritance from OpenAILikeEmbeddingHandler"""
        config = MorphEmbeddingConfig()

        from litellm.llms.openai_like.embedding.handler import OpenAILikeEmbeddingHandler

        assert isinstance(config, OpenAILikeEmbeddingHandler)
        assert hasattr(config, "_get_openai_compatible_provider_info")

    def test_morph_embedding_mock(self, respx_mock):
        """
        Mock test for Morph embeddings API.
        This test mocks the actual HTTP request to test the integration properly.
        """
        import respx
        from litellm import embedding
        
        # Set up environment variables for the test
        api_key = "fake-morph-key"
        api_base = "https://api.morphllm.com/v1"
        model = "morph/morph-embedding-v2"
        
        # Mock the HTTP request to the Morph API
        respx_mock.post(f"{api_base}/embeddings").respond(
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "index": 0
                    }
                ],
                "model": "morph-embedding-v2",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5
                }
            },
            status_code=200
        )
        
        # Make the actual API call through LiteLLM
        response = embedding(
            model=model,
            input="Hello world",
            api_key=api_key,
            api_base=api_base
        )
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, "data")
        assert len(response.data) > 0
        assert hasattr(response.data[0], "embedding")
        assert isinstance(response.data[0].embedding, list)
        assert len(response.data[0].embedding) == 5 