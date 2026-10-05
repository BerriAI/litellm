from collections.abc import Mapping, Sequence

"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict


class ModelRequestDefaultsInfo(TypedDict, total=False):
    output_token_budget: ReadOnly[int]
    output_token_budget_by_reasoning_effort: ReadOnly[Mapping[str, int]]


class ModelDiscoveryInfo(TypedDict, total=False):
    context_window: ReadOnly[int]
    supports_function_calling: ReadOnly[bool]
    supports_parallel_function_calling: ReadOnly[bool]
    supports_reasoning: ReadOnly[bool]
    reasoning_effort_levels: ReadOnly[Sequence[str]]
    default_reasoning_effort: ReadOnly[str]
    supported_endpoints: ReadOnly[Sequence[str]]
    supported_modalities: ReadOnly[Sequence[str]]
    supported_output_modalities: ReadOnly[Sequence[str]]
    request_defaults: ReadOnly[ModelRequestDefaultsInfo]


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ModelInfoResponse(ModelDiscoveryInfo):
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
