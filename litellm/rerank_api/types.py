"""
LiteLLM Follows the cohere API format for the re rank API
https://docs.cohere.com/reference/rerank

"""

from typing import List, Optional, Union

from pydantic import BaseModel


class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: Optional[int] = None
    documents: List[Union[str, dict]]
    rank_fields: Optional[List[str]] = None
    return_documents: Optional[bool] = None
    max_chunks_per_doc: Optional[int] = None


class RerankResponse(BaseModel):
    id: str
    results: List[dict]  # Contains index and relevance_score
    meta: dict  # Contains api_version and billed_units
    _hidden_params: dict = {}

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        return key in self.__dict__
