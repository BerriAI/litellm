"""
Test suite for Amazon Nova Multimodal Embeddings integration with LiteLLM.

Tests cover:
- Synchronous text embeddings
- Synchronous image embeddings
- Synchronous video/audio embeddings
- Asynchronous embeddings with segmentation
- Different embedding purposes and dimensions
- Error handling
"""

import pytest

from litellm.llms.bedrock.embed.amazon_nova_transformation import (
    AmazonNovaEmbeddingConfig,
)


class TestNovaTransformationRequest:
    """Test request transformation for Nova embeddings."""

    def test_async_invoke_requires_output_s3_uri(self):
        """Test that async invoke requires output_s3_uri."""
        config = AmazonNovaEmbeddingConfig()

        inference_params = {
            "embedding_purpose": "GENERIC_INDEX",
        }

        with pytest.raises(ValueError, match="output_s3_uri is required"):
            config._transform_request(
                input="Test text",
                inference_params=inference_params,
                async_invoke_route=True,
                model_id="amazon.nova-2-multimodal-embeddings-v1:0",
                output_s3_uri=None,
            )


class TestNovaEmbeddingIntegration:
    """Integration tests for Nova embeddings through LiteLLM."""
