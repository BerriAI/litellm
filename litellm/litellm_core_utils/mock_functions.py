from collections.abc import Sequence

from ..types.utils import (
    Embedding,
    EmbeddingResponse,
    ImageObject,
    ImageResponse,
    Usage,
)


def _embedding_input_count(embedding_input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]]) -> int:
    match embedding_input:
        case str() | [int(), *_]:
            return 1
        case _:
            return len(embedding_input)


def mock_embedding(
    model: str,
    embedding_input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
    mock_response: list[float] | None,
):
    if mock_response is None:
        mock_response = [0.0] * 1536
    elif mock_response == "error":
        raise Exception("Mock error")
    return EmbeddingResponse(
        model=model,
        data=[
            Embedding(embedding=mock_response, index=index, object="embedding")
            for index in range(_embedding_input_count(embedding_input))
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=0),
    )


def mock_image_generation(model: str, mock_response: str):
    return ImageResponse(
        data=[ImageObject(url=mock_response)],
    )
