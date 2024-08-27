"""
LiteLLM Follows the cohere API format for the re rank API
https://docs.cohere.com/reference/rerank

"""

from pydantic import BaseModel


class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: int
    documents: list[str]


class RerankResponse(BaseModel):
    id: str
    results: list[dict]  # Contains index and relevance_score
    meta: dict  # Contains api_version and billed_units
