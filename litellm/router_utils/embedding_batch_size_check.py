"""Validate the number of inputs in a single embeddings request."""

from collections.abc import Mapping, Sequence
from typing import Final

import litellm


def count_embedding_inputs(input: str | Sequence[object]) -> int:
    """Return the number of embeddings requested without tokenizing the input."""
    if isinstance(input, str):
        return 1
    if input and isinstance(input[0], int):
        return 1
    return len(input)


def raise_if_embedding_batch_too_large(
    input: str | Sequence[object],
    model_info: Mapping[str, object],
    model: str,
    llm_provider: str,
) -> None:
    """Reject an embeddings request that exceeds its deployment's input limit."""
    limit: Final = model_info.get("max_embedding_batch_size")
    if not isinstance(limit, int):
        return

    input_count: Final = count_embedding_inputs(input)
    if input_count <= limit:
        return

    raise litellm.BadRequestError(
        message=(
            f"Embedding request has {input_count} inputs, which exceeds "
            f"max_embedding_batch_size={limit} for model={model}. "
            "Split the input into smaller batches."
        ),
        model=model,
        llm_provider=llm_provider,
    )
