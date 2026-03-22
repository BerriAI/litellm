"""
E2E test for hosted_vllm embeddings with real API calls.

This test verifies that the hosted_vllm provider works correctly with real API endpoints.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm


class TestHostedVLLMEmbeddingE2E:
    """E2E test suite for hosted_vllm provider embeddings."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sync_mode", [True, False])
    async def test_hosted_vllm_embedding_basic(self, sync_mode):
        """Test basic embedding call with hosted_vllm provider."""
        # Skip if API base is not configured
        api_base = os.getenv("HOSTED_VLLM_API_BASE")
        if not api_base:
            pytest.skip("HOSTED_VLLM_API_BASE environment variable not set")

        model = "hosted_vllm/nomic-ai/nomic-embed-text-v1.5"
        input_text = "Hello, this is a test embedding"

        if sync_mode:
            response = litellm.embedding(
                model=model,
                input=input_text,
                api_base=api_base,
            )
        else:
            response = await litellm.aembedding(
                model=model,
                input=input_text,
                api_base=api_base,
            )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "data")
        assert len(response.data) == 1
        # Adapt for response data as a dict (legacy or OpenAI compat)
        item = response.data[0]
        # If data is a dict, use key lookup; if it's an object, fallback to attribute
        if isinstance(item, dict):
            assert "embedding" in item
            assert len(item["embedding"]) > 0
        else:
            assert hasattr(item, "embedding")
            assert len(item.embedding) > 0
        assert hasattr(response, "usage")
        assert getattr(response.usage, "total_tokens", 0) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sync_mode", [True, False])
    async def test_hosted_vllm_embedding_multiple_inputs(self, sync_mode):
        """Test embedding with multiple inputs."""
        api_base = os.getenv("HOSTED_VLLM_API_BASE")
        if not api_base:
            pytest.skip("HOSTED_VLLM_API_BASE environment variable not set")

        model = "hosted_vllm/nomic-ai/nomic-embed-text-v1.5"
        inputs = [
            "First test sentence",
            "Second test sentence",
            "Third test sentence",
        ]

        if sync_mode:
            response = litellm.embedding(
                model=model,
                input=inputs,
                api_base=api_base,
            )
        else:
            response = await litellm.aembedding(
                model=model,
                input=inputs,
                api_base=api_base,
            )

        # Verify response structure
        assert response is not None
        assert len(response.data) == 3
        for i, emb_data in enumerate(response.data):
            assert emb_data["index"] == i
            assert len(emb_data["embedding"]) > 0

    def test_hosted_vllm_embedding_with_api_key(self):
        """Test embedding with API key authentication."""
        api_base = os.getenv("HOSTED_VLLM_API_BASE")
        api_key = os.getenv("HOSTED_VLLM_API_KEY")
        
        if not api_base:
            pytest.skip("HOSTED_VLLM_API_BASE environment variable not set")
        
        if not api_key:
            pytest.skip("HOSTED_VLLM_API_KEY environment variable not set")

        model = "hosted_vllm/nomic-ai/nomic-embed-text-v1.5"
        input_text = "Test with API key"

        response = litellm.embedding(
            model=model,
            input=input_text,
            api_base=api_base,
            api_key=api_key,
        )

        # Verify response
        assert response is not None
        assert len(response.data) == 1
        assert len(response.data[0]["embedding"]) > 0

    def test_hosted_vllm_embedding_deterministic(self):
        """Test that same input produces same embedding (deterministic)."""
        api_base = os.getenv("HOSTED_VLLM_API_BASE")
        if not api_base:
            pytest.skip("HOSTED_VLLM_API_BASE environment variable not set")

        model = "hosted_vllm/nomic-ai/nomic-embed-text-v1.5"
        input_text = "This should produce the same embedding every time"

        response1 = litellm.embedding(
            model=model,
            input=input_text,
            api_base=api_base,
        )

        response2 = litellm.embedding(
            model=model,
            input=input_text,
            api_base=api_base,
        )

        # Verify embeddings are identical
        emb1 = response1.data[0]["embedding"]
        emb2 = response2.data[0]["embedding"]
        assert emb1 == emb2, "Embeddings should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
