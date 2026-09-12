"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ModelInfoResponse(TypedDict):
    """OpenAI-compatible model object. `mode`, `max_input_tokens`, `max_output_tokens`,
    and the per-token costs are attached when the cost map or deployment config knows
    them; `metadata` is present only with include_metadata=true.
    """

    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    mode: NotRequired[str]
    max_input_tokens: NotRequired[int]
    max_output_tokens: NotRequired[int]
    input_cost_per_token: NotRequired[ReadOnly[float]]
    output_cost_per_token: NotRequired[ReadOnly[float]]
    metadata: NotRequired[ModelInfoMetadata]
