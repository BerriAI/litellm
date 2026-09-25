"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ModelPricing(TypedDict):
    """Effective token prices for a listed model, in USD per token.

    ``None`` is not the same as free: an unmapped model reports null so a caller doing
    cost accounting cannot read a missing price as a zero-cost model.
    """

    input_cost_per_token: ReadOnly[float | None]
    output_cost_per_token: ReadOnly[float | None]


class ModelInfoResponse(TypedDict):
    """OpenAI-compatible model object. `mode`, `max_input_tokens`, and
    `max_output_tokens` are attached when the cost map or deployment config
    knows them; `metadata` is present only with include_metadata=true.
    """

    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    mode: NotRequired[str]
    max_input_tokens: NotRequired[int]
    max_output_tokens: NotRequired[int]
    metadata: NotRequired[ModelInfoMetadata]
    pricing: NotRequired[ModelPricing]
