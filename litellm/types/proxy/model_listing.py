"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ModelInfoResponse(TypedDict):
    """OpenAI-compatible model object. `mode`, `max_input_tokens`, and
    `max_output_tokens` are attached when the cost map or deployment config
    knows them; `metadata` is present only with include_metadata=true.

    `catalog_only` marks a row the proxy advertises but does not route, so a
    client choosing a model from this listing can skip the ones a request would
    be rejected for. It is absent on every routable model.
    """

    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    mode: NotRequired[str]
    max_input_tokens: NotRequired[int]
    max_output_tokens: NotRequired[int]
    metadata: NotRequired[ModelInfoMetadata]
    catalog_only: NotRequired[ReadOnly[Literal[True]]]
