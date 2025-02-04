"""
LiteLLM Follows the cohere API format for the re rank API
https://docs.cohere.com/reference/rerank

"""

from typing import List, Optional, Union

from pydantic import BaseModel, PrivateAttr
from typing_extensions import TypedDict


class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: Optional[int] = None
    documents: List[Union[str, dict]]
    rank_fields: Optional[List[str]] = None
    return_documents: Optional[bool] = None
    max_chunks_per_doc: Optional[int] = None


class OptionalRerankParams(TypedDict, total=False):
    query: str
    top_n: Optional[int]
    documents: List[Union[str, dict]]
    rank_fields: Optional[List[str]]
    return_documents: Optional[bool]
    max_chunks_per_doc: Optional[int]


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


class RerankResponseResult(TypedDict):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    id: Optional[str] = None
    results: Optional[List[RerankResponseResult]] = (
        None  # Contains index and relevance_score
    )
    meta: Optional[RerankResponseMeta] = None  # Contains api_version and billed_units

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        return key in self.__dict__
