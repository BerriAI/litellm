"""
LiteLLM Follows the cohere API format for the re rank API
https://docs.cohere.com/reference/rerank

"""

from pydantic import BaseModel, PrivateAttr
from typing_extensions import Required, TypedDict


class RerankRequest(BaseModel):
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


class OptionalRerankParams(TypedDict, total=False):
    query: str
    top_n: int | None
    documents: list[str | dict]
    rank_fields: list[str] | None
    return_documents: bool | None
    max_chunks_per_doc: int | None
    max_tokens_per_doc: int | None
    instruction: str | None


class RerankBilledUnits(TypedDict, total=False):
    search_units: int | None
    total_tokens: int | None


class RerankTokens(TypedDict, total=False):
    input_tokens: int | None
    output_tokens: int | None


class RerankResponseMeta(TypedDict, total=False):
    api_version: dict | None
    billed_units: RerankBilledUnits | None
    tokens: RerankTokens | None


class RerankResponseDocument(TypedDict):
    text: str


class RerankResponseResult(TypedDict, total=False):
    index: Required[int]
    relevance_score: Required[float]
    document: RerankResponseDocument


class RerankResponse(BaseModel):
    id: str | None = None
    results: list[RerankResponseResult] | None = None  # Contains index and relevance_score
    meta: RerankResponseMeta | None = None  # Contains api_version and billed_units

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key) -> bool:
        return key in self.__dict__
