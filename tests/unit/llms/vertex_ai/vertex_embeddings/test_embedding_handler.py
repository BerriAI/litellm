from typing import Final

import pytest

from litellm.llms.vertex_ai.vertex_embeddings.transformation import VertexAITextEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


def _gecko_prediction(values: list[float], token_count: int) -> dict[str, object]:
    return {"embeddings": {"values": values, "statistics": {"token_count": token_count, "truncated": False}}}


@pytest.mark.parametrize(
    ("metadata", "character_count"),
    [({"metadata": {"billableCharacterCount": 10}}, 10), ({}, None)],
    ids=["billable_character_count_reported", "no_metadata"],
)
def test_vertex_embedding_usage_carries_the_billable_character_count_google_prices_on(
    metadata: dict[str, object], character_count: int | None
) -> None:
    payload: Final = {"predictions": [_gecko_prediction([0.1, 0.2, 0.3], 3), _gecko_prediction([1, 2.0, 3], 5)], **metadata}

    response: Final = VertexAITextEmbeddingConfig().transform_vertex_response_to_openai(
        response=payload, model="text-embedding-005", model_response=EmbeddingResponse()
    )

    assert (response.usage.prompt_tokens, getattr(response.usage.prompt_tokens_details, "character_count", None)) == (
        8,
        character_count,
    )
