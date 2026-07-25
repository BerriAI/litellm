from typing import List, Literal

from pydantic import BaseModel


class OptimizedRequestLog(BaseModel):
    request_id: str
    timestamp: str
    model: str
    total_tokens: int
    optimization_type: Literal["compression", "caching", "both"]
    spend: float
    savings: float
    original_cost: float
    compression_savings_spend: float
    prompt_caching_savings_spend: float
    tokens_saved: int
    cache_read_tokens: int


class OptimizedRequestLogsResponse(BaseModel):
    logs: List[OptimizedRequestLog]
    total: int
    page: int
    page_size: int
    total_pages: int
