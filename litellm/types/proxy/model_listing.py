"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Literal

from typing_extensions import NotRequired, TypedDict


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ModelInfoResponse(TypedDict):
    """OpenAI-compatible model object. `metadata` is present only when the
    endpoint is called with include_metadata=true.
    """

    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    metadata: NotRequired[ModelInfoMetadata]
