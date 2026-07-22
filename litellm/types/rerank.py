"""
LiteLLM Follows the cohere API format for the re rank API
https://docs.cohere.com/reference/rerank

"""

from typing import List, Optional, Union

from pydantic import BaseModel, PrivateAttr, computed_field
from typing_extensions import Required, TypedDict


class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: Optional[int] = None
    documents: List[Union[str, dict]]
    rank_fields: Optional[List[str]] = None
    return_documents: Optional[bool] = None
    max_chunks_per_doc: Optional[int] = None
    max_tokens_per_doc: Optional[int] = None
    # Optional task/query instruction passed through to providers that support it
    # (e.g. hosted vLLM / Qwen3-Reranker, DeepInfra). Omitted from the outgoing
    # request when None, so this is fully backward-compatible.
    instruction: Optional[str] = None


class OptionalRerankParams(TypedDict, total=False):
    query: str
    top_n: Optional[int]
    documents: List[Union[str, dict]]
    rank_fields: Optional[List[str]]
    return_documents: Optional[bool]
    max_chunks_per_doc: Optional[int]
    max_tokens_per_doc: Optional[int]
    instruction: Optional[str]


class RerankBilledUnits(TypedDict, total=False):
    search_units: Optional[int]
    total_tokens: Optional[int]


class RerankTokens(TypedDict, total=False):
    input_tokens: Optional[int]
    output_tokens: Optional[int]


class RerankResponseMeta(TypedDict, total=False):
    api_version: Optional[dict]
    billed_units: Optional[RerankBilledUnits]
    tokens: Optional[RerankTokens]


class RerankResponseDocument(TypedDict):
    text: str


class RerankResponseResult(TypedDict, total=False):
    index: Required[int]
    relevance_score: Required[float]
    document: RerankResponseDocument


class RerankResponse(BaseModel):
    id: Optional[str] = None
    results: Optional[List[RerankResponseResult]] = None  # Contains index and relevance_score
    meta: Optional[RerankResponseMeta] = None  # Contains api_version and billed_units

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)

    @computed_field
    @property
    def usage(self) -> Optional[dict]:  # pyright: ignore[reportReturnType]  # TypedDict .get() returns Unknown, acceptable here
        if self.meta is None:
            return None
        tokens = self.meta.get("tokens") or {}
        billed = self.meta.get("billed_units") or {}
        input_tokens = tokens.get("input_tokens") or billed.get("total_tokens")
        if input_tokens is None:
            return None
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": 0,
            "total_tokens": input_tokens,
        }

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        return key in self.__dict__
