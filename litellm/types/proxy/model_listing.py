"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Final, Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ResolvedCosts(TypedDict, total=False):
    """The per-token prices a listing resolved, keyed as `ModelInfoResponse` spells them.

    A price the cost map could not supply is absent rather than zero, so that a
    caller costing out its own usage never reads an unpriced model as free.
    """

    input_cost_per_token: ReadOnly[float]
    output_cost_per_token: ReadOnly[float]


EMPTY_RESOLVED_COSTS: Final[ResolvedCosts] = {}


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
