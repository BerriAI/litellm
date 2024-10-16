from typing import List, Optional

from ..types.utils import (
    Categories,
    CategoryAppliedInputTypes,
    CategoryScores,
    Embedding,
    EmbeddingResponse,
    ImageObject,
    ImageResponse,
    Moderation,
    ModerationCreateResponse,
)


def mock_embedding(model: str, mock_response: Optional[List[float]]):
    if mock_response is None:
        mock_response = [0.0] * 1536
    return EmbeddingResponse(
        model=model,
        data=[Embedding(embedding=mock_response, index=0, object="embedding")],
    )


def mock_image_generation(model: str, mock_response: str):
    return ImageResponse(
        data=[ImageObject(url=mock_response)],
    )
