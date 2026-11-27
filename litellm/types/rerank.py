"""
LiteLLM Follows the cohere API format for the re rank API
https://docs.cohere.com/reference/rerank

"""

from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeVar, overload

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
)
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.types.llms.base import LiteLLMBaseModel


class RerankRequest(LiteLLMBaseModel):
    model: str
    query: str
    top_n: int | None = None
    documents: list[str | dict]
    rank_fields: list[str] | None = None
    return_documents: bool | None = None
    max_chunks_per_doc: int | None = None
    max_tokens_per_doc: int | None = None
    # Optional task/query instruction passed through to providers that support it
    # (e.g. hosted vLLM / Qwen3-Reranker, DeepInfra). Omitted from the outgoing
    # request when None, so this is fully backward-compatible.
    instruction: str | None = None
    truncate_prompt_tokens: int | None = None
    truncation_side: Literal["left", "right"] | None = None
    max_tokens_per_query: int | None = None


class HostedVLLMRerankTruncationParams(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    truncate_prompt_tokens: int | None = None
    truncation_side: Literal["left", "right"] | None = None
    max_tokens_per_query: int | None = None
    max_tokens_per_doc: int | None = None


class OptionalRerankParams(TypedDict, total=False):
    query: str
    top_n: int | None
    documents: list[str | dict]
    rank_fields: list[str] | None
    return_documents: bool | None
    max_chunks_per_doc: int | None
    max_tokens_per_doc: int | None
    instruction: str | None
    truncate_prompt_tokens: ReadOnly[int | None]
    truncation_side: ReadOnly[Literal["left", "right"] | None]
    max_tokens_per_query: ReadOnly[int | None]


class RerankBilledUnits(TypedDict, total=False):
    search_units: int | None
    total_tokens: int | None
    input_tokens: ReadOnly[int | None]
    output_tokens: ReadOnly[int | None]


class RerankTokens(TypedDict, total=False):
    input_tokens: int | None
    output_tokens: int | None


class RerankResponseMeta(TypedDict, total=False):
    api_version: dict[str, object] | None
    billed_units: RerankBilledUnits | None
    tokens: RerankTokens | None


class RerankUsage(TypedDict):
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    total_tokens: ReadOnly[int]


_RERANK_META: Final = TypeAdapter(RerankResponseMeta)
_FIELD_VALUE: Final = TypeAdapter(object)
_Default = TypeVar("_Default")


def get_rerank_usage(response: Mapping[str, object]) -> RerankUsage | None:
    if "results" not in response or response.get("meta") is None:
        return None
    try:
        meta: Final = _RERANK_META.validate_python(response.get("meta"))
    except ValidationError:
        return None
    tokens: Final = meta.get("tokens") or RerankTokens()
    billed: Final = meta.get("billed_units") or RerankBilledUnits()
    input_tokens: Final = next(
        (count for count in (tokens.get("input_tokens"), billed.get("input_tokens")) if count is not None), None
    )
    output_tokens: Final = next(
        (count for count in (tokens.get("output_tokens"), billed.get("output_tokens")) if count is not None), None
    )
    total_tokens: Final = billed.get("total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    completion_tokens: Final = output_tokens or 0
    prompt_tokens: Final = input_tokens if input_tokens is not None else max((total_tokens or 0) - completion_tokens, 0)
    return RerankUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
    )


class RerankResponseDocument(TypedDict):
    text: str


class RerankResponseResult(TypedDict, total=False):
    index: Required[int]
    relevance_score: Required[float]
    document: RerankResponseDocument


_PROVIDER_USAGE: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)
_DUMPED_FIELDS: Final = TypeAdapter(dict[str, object])
_PROVIDER_USAGE_KEYS: Final = frozenset({"usage", "provider_usage"})


def provider_usage_or_none(value: object) -> Mapping[str, object] | None:
    try:
        return _PROVIDER_USAGE.validate_python(value)
    except ValidationError:
        return None


class RerankResponse(LiteLLMBaseModel):  # frozen-ok: existing conversion updates supplied response objects in place
    model_config = ConfigDict(validate_by_name=True)

    id: str | None = None
    results: list[RerankResponseResult] | None = None  # Contains index and relevance_score
    meta: RerankResponseMeta | None = None  # Contains api_version and billed_units
    # The provider's raw block is "usage" only in by-alias (client JSON) dumps. The attribute and plain dumps use
    # provider_usage, since logging, spend tracking and routing read a `usage` key or attribute as litellm's token Usage
    provider_usage: Mapping[str, object] | None = Field(default=None, alias="usage")

    # Define private attributes using PrivateAttr
    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    @field_validator("provider_usage", mode="before")
    @classmethod
    def _ignore_malformed_provider_usage(cls, value: object) -> Mapping[str, object] | None:
        return provider_usage_or_none(value)

    @model_serializer(mode="wrap")
    def _omit_absent_provider_usage(self, handler: SerializerFunctionWrapHandler) -> Mapping[str, object]:
        fields: Final = _DUMPED_FIELDS.validate_python(handler(self))
        if self.provider_usage is not None:
            return fields
        return {key: value for key, value in fields.items() if key not in _PROVIDER_USAGE_KEYS}

    @property
    def hidden_params(self) -> dict[str, object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params

    @overload
    def __getitem__(self, key: Literal["id"]) -> str | None: ...

    @overload
    def __getitem__(self, key: Literal["results"]) -> list[RerankResponseResult] | None: ...

    @overload
    def __getitem__(self, key: Literal["meta"]) -> RerankResponseMeta | None: ...

    @overload
    def __getitem__(self, key: str) -> str | Sequence[RerankResponseResult] | RerankResponseMeta | None: ...

    def __getitem__(self, key: str) -> object:
        return _FIELD_VALUE.validate_python(self.__dict__[key])

    @overload
    def get(self, key: Literal["id"], default: None = None) -> str | None: ...

    @overload
    def get(self, key: Literal["results"], default: None = None) -> list[RerankResponseResult] | None: ...

    @overload
    def get(self, key: Literal["results"], default: _Default) -> Sequence[RerankResponseResult] | _Default | None: ...

    @overload
    def get(self, key: Literal["meta"], default: None = None) -> RerankResponseMeta | None: ...

    @overload
    def get(self, key: str, default: object = None) -> object: ...

    def get(self, key: str, default: object = None) -> object:
        return _FIELD_VALUE.validate_python(self.__dict__.get(key, default))

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__
