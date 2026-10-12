"""Tests for litellm/llms/vertex_ai/vertex_embeddings/transformation.py."""

from litellm.llms.vertex_ai.vertex_embeddings.transformation import (
    VertexAITextEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse


def _vertex_response(token_counts: list[int], billable_character_count: int | None = None) -> dict:
    response: dict = {
        "predictions": [
            {"embeddings": {"statistics": {"truncated": False, "token_count": n}, "values": [0.1]}}
            for n in token_counts
        ]
    }
    if billable_character_count is not None:
        response["metadata"] = {"billableCharacterCount": billable_character_count}
    return response


def test_billable_character_count_is_mapped_to_prompt_tokens_details():
    """Vertex reports the billable unit for per-character-priced models (e.g.
    text-embedding-005) as top-level metadata.billableCharacterCount. It must land
    on usage.prompt_tokens_details.character_count instead of being discarded."""
    out = VertexAITextEmbeddingConfig().transform_vertex_response_to_openai(
        _vertex_response([8, 1], billable_character_count=21),
        "text-embedding-005",
        EmbeddingResponse(),
    )

    assert out.usage.prompt_tokens == 9
    assert out.usage.prompt_tokens_details is not None
    assert out.usage.prompt_tokens_details.character_count == 21


def test_prompt_tokens_details_stay_none_without_metadata():
    """Responses without metadata keep the old shape: no prompt_tokens_details."""
    out = VertexAITextEmbeddingConfig().transform_vertex_response_to_openai(
        _vertex_response([8]),
        "gemini-embedding-001",
        EmbeddingResponse(),
    )

    assert out.usage.prompt_tokens == 8
    assert out.usage.prompt_tokens_details is None
