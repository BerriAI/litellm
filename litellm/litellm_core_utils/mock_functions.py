from typing import List, Optional

from ..types.utils import (
    Embedding,
    EmbeddingResponse,
    ImageObject,
    ImageResponse,
    Usage,
)


def mock_embedding(model: str, mock_response: Optional[List[float]]):
    if mock_response is None:
        mock_response = [0.0] * 1536
    return EmbeddingResponse(
        model=model,
        data=[Embedding(embedding=mock_response, index=0, object="embedding")],
        usage=Usage(prompt_tokens=10, completion_tokens=0),
    )


def mock_image_generation(model: str, mock_response: str):
    return ImageResponse(
        data=[ImageObject(url=mock_response)],
    )
