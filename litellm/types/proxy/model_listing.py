"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Literal

from typing_extensions import NotRequired, TypedDict


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ModelInfoResponse(TypedDict):
    """OpenAI-compatible model object. `mode`, `max_input_tokens`,
    `max_output_tokens`, and `max_model_len` are attached when configured or
    known to the cost map; `metadata` is present only when the endpoint is
    called with include_metadata=true.
    """

    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    mode: NotRequired[str]
    max_input_tokens: NotRequired[int]
    max_output_tokens: NotRequired[int]
    max_model_len: NotRequired[int]
    metadata: NotRequired[ModelInfoMetadata]
