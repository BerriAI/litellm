"""
Test that Vertex AI text embedding responses preserve billableCharacterCount
in usage.prompt_tokens_details.character_count.

Fixes: https://github.com/BerriAI/litellm/issues/45999
"""

import pytest

from litellm.llms.vertex_ai.vertex_embeddings.transformation import (
    VertexAITextEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse


class TestVertexTextEmbeddingBillableCharacterCount:
    def setup_method(self):
        self.config = VertexAITextEmbeddingConfig()

    def test_billable_character_count_mapped_to_usage_details(self):
        """
        Test that metadata.billableCharacterCount from Vertex is mapped to
        usage.prompt_tokens_details.character_count.
        """
        # Sample Vertex AI text embedding response with billable character count
        vertex_response = {
            "predictions": [
                {
                    "embeddings": {
                        "statistics": {"truncated": False, "token_count": 8},
                        "values": [0.1, 0.2, 0.3],  # Sample embedding values
                    }
                },
                {
                    "embeddings": {
                        "statistics": {"truncated": False, "token_count": 1},
                        "values": [0.4, 0.5, 0.6],
                    }
                },
            ],
            "metadata": {"billableCharacterCount": 21},
        }

        model_response = EmbeddingResponse()
        result = self.config.transform_vertex_response_to_openai(
            response=vertex_response,
            model="text-embedding-005",
            model_response=model_response,
        )

        # Verify usage tokens
        assert result.usage.prompt_tokens == 9  # 8 + 1
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 9

        # Verify billable character count is preserved
        assert result.usage.prompt_tokens_details is not None
        assert result.usage.prompt_tokens_details.character_count == 21

    def test_no_character_count_when_metadata_missing(self):
        """
        Test that usage.prompt_tokens_details is not set when metadata
        is missing from the response.
        """
        # Vertex response without metadata
        vertex_response = {
            "predictions": [
                {
                    "embeddings": {
                        "statistics": {"truncated": False, "token_count": 5},
                        "values": [0.1, 0.2, 0.3],
                    }
                }
            ]
        }

        model_response = EmbeddingResponse()
        result = self.config.transform_vertex_response_to_openai(
            response=vertex_response,
            model="text-embedding-005",
            model_response=model_response,
        )

        # Verify usage tokens
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 5

        # Verify no prompt_tokens_details when metadata is absent
        assert result.usage.prompt_tokens_details is None

    def test_no_character_count_when_billable_count_missing(self):
        """
        Test that usage.prompt_tokens_details is not set when metadata exists
        but billableCharacterCount is missing.
        """
        # Vertex response with metadata but no billableCharacterCount
        vertex_response = {
            "predictions": [
                {
                    "embeddings": {
                        "statistics": {"truncated": False, "token_count": 3},
                        "values": [0.1, 0.2],
                    }
                }
            ],
            "metadata": {"someOtherField": "value"},
        }

        model_response = EmbeddingResponse()
        result = self.config.transform_vertex_response_to_openai(
            response=vertex_response,
            model="gemini-embedding-001",
            model_response=model_response,
        )

        # Verify usage tokens
        assert result.usage.prompt_tokens == 3
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 3

        # Verify no prompt_tokens_details when billableCharacterCount is absent
        assert result.usage.prompt_tokens_details is None
