"""
Vertex text embedding response transformation.
"""

from litellm.llms.vertex_ai.vertex_embeddings.transformation import VertexAITextEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


def _response(metadata: dict | None) -> dict:
    response: dict = {
        "predictions": [
            {"embeddings": {"statistics": {"truncated": False, "token_count": 2}, "values": [0.1, 0.2]}},
            {"embeddings": {"statistics": {"truncated": False, "token_count": 2}, "values": [0.3, 0.4]}},
        ]
    }
    if metadata is not None:
        response["metadata"] = metadata
    return response


class TestBillableCharacterCount:
    def test_billable_character_count_is_passed_through(self):
        model_response = VertexAITextEmbeddingConfig().transform_vertex_response_to_openai(
            response=_response({"billableCharacterCount": 17}),
            model="text-embedding-005",
            model_response=EmbeddingResponse(),
        )

        usage = model_response.usage
        assert usage.prompt_tokens == 4
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.character_count == 17

    def test_missing_metadata_leaves_details_unset(self):
        model_response = VertexAITextEmbeddingConfig().transform_vertex_response_to_openai(
            response=_response(None),
            model="text-embedding-005",
            model_response=EmbeddingResponse(),
        )

        usage = model_response.usage
        assert usage.prompt_tokens == 4
        assert usage.prompt_tokens_details is None

    def test_non_integer_character_count_is_ignored(self):
        model_response = VertexAITextEmbeddingConfig().transform_vertex_response_to_openai(
            response=_response({"billableCharacterCount": "17"}),
            model="text-embedding-005",
            model_response=EmbeddingResponse(),
        )

        assert model_response.usage.prompt_tokens_details is None
