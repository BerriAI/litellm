from collections.abc import Sequence
from typing import Final

from ..exceptions import BadRequestError
from ..types.utils import (
    Embedding,
    EmbeddingResponse,
    ImageObject,
    ImageResponse,
    Usage,
)

# OpenAI caps an embedding input array at 2048 items, as of 2026-10-05:
# https://platform.openai.com/docs/api-reference/embeddings/create
_MAX_EMBEDDING_INPUTS: Final = 2048


def _embedding_input_count(embedding_input: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]]) -> int:
    if isinstance(embedding_input, str) or (len(embedding_input) > 0 and isinstance(embedding_input[0], int)):
        return 1
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
    input_count: Final = _embedding_input_count(embedding_input)
    if not 1 <= input_count <= _MAX_EMBEDDING_INPUTS:
        raise BadRequestError(
            message=f"input must contain between 1 and {_MAX_EMBEDDING_INPUTS} items, got {input_count}",
            model=model,
            llm_provider=None,
            body={"param": "input"},
        )
    return EmbeddingResponse(
        model=model,
        data=[Embedding(embedding=mock_response, index=index, object="embedding") for index in range(input_count)],
        usage=Usage(prompt_tokens=10, completion_tokens=0),
    )


def mock_image_generation(model: str, mock_response: str):
    return ImageResponse(
        data=[ImageObject(url=mock_response)],
    )
