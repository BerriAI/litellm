"""
Test that streaming response headers are propagated to the final assembled response.
Fixes: https://github.com/BerriAI/litellm/issues/19930
"""

import pytest
from unittest.mock import MagicMock

import litellm
from litellm import ModelResponse
from litellm.main import stream_chunk_builder
from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta


class TestStreamingHeadersPropagation:
    """Tests for streaming headers propagation to final response."""

    def test_stream_chunk_builder_propagates_additional_headers(self):
        """
        Test that stream_chunk_builder copies _hidden_params["additional_headers"]
        from chunks to the final assembled response.
        """
        # Create mock streaming chunks with _hidden_params containing headers
        chunk1 = ModelResponseStream(
            id="test-id",
            model="gpt-4",
            choices=[StreamingChoices(index=0, delta=Delta(content="Hello "))],
        )
        chunk1._hidden_params = {
            "additional_headers": {
                "llm_provider-x-ms-region": "eastus",
                "llm_provider-x-ratelimit-remaining-tokens": "10000",
            },
            "model_id": "test-model-id",
            "api_base": "https://api.openai.com",
            "custom_llm_provider": "azure",
        }

        chunk2 = ModelResponseStream(
            id="test-id",
            model="gpt-4",
            choices=[
                StreamingChoices(index=0, delta=Delta(content="World"), finish_reason="stop")
            ],
        )
        chunk2._hidden_params = {}

        chunks = [chunk1, chunk2]

        # Call stream_chunk_builder
        result = stream_chunk_builder(chunks=chunks, messages=[{"role": "user", "content": "Hi"}])

        # Verify the result has _hidden_params with headers
        assert result is not None
        assert hasattr(result, "_hidden_params")
        assert "additional_headers" in result._hidden_params

        # Check that the specific headers are preserved
        additional_headers = result._hidden_params["additional_headers"]
        assert additional_headers.get("llm_provider-x-ms-region") == "eastus"
        assert additional_headers.get("llm_provider-x-ratelimit-remaining-tokens") == "10000"

        # Check other hidden params are also copied
        assert result._hidden_params.get("model_id") == "test-model-id"
        assert result._hidden_params.get("api_base") == "https://api.openai.com"
        assert result._hidden_params.get("custom_llm_provider") == "azure"

    def test_stream_chunk_builder_handles_missing_hidden_params(self):
        """
        Test that stream_chunk_builder works correctly when chunks don't have _hidden_params.
        """
        # Create chunks without _hidden_params
        chunk1 = ModelResponseStream(
            id="test-id",
            model="gpt-4",
            choices=[StreamingChoices(index=0, delta=Delta(content="Hello "))],
        )
        # Don't set _hidden_params

        chunk2 = ModelResponseStream(
            id="test-id",
            model="gpt-4",
            choices=[
                StreamingChoices(index=0, delta=Delta(content="World"), finish_reason="stop")
            ],
        )

        chunks = [chunk1, chunk2]

        # Call stream_chunk_builder - should not raise
        result = stream_chunk_builder(chunks=chunks, messages=[{"role": "user", "content": "Hi"}])

        # Verify result is valid
        assert result is not None

    def test_stream_chunk_builder_handles_empty_additional_headers(self):
        """
        Test that stream_chunk_builder handles empty additional_headers gracefully.
        """
        chunk1 = ModelResponseStream(
            id="test-id",
            model="gpt-4",
            choices=[StreamingChoices(index=0, delta=Delta(content="Hello "))],
        )
        chunk1._hidden_params = {
            "model_id": "test-id",
            # No additional_headers
        }

        chunk2 = ModelResponseStream(
            id="test-id",
            model="gpt-4",
            choices=[
                StreamingChoices(index=0, delta=Delta(content="World"), finish_reason="stop")
            ],
        )

        chunks = [chunk1, chunk2]

        result = stream_chunk_builder(chunks=chunks, messages=[{"role": "user", "content": "Hi"}])

        assert result is not None
        assert hasattr(result, "_hidden_params")
        # additional_headers should not be present if it wasn't in the source
        assert "additional_headers" not in result._hidden_params or result._hidden_params.get("additional_headers") is None

