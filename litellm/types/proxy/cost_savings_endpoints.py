from typing import Literal

from pydantic import BaseModel, Field

OptimizationType = Literal["caching", "compression"]


class CostSavingsMetrics(BaseModel):
    cache_savings: float = 0.0
    compression_savings: float = 0.0
    total_savings: float = 0.0
    spend: float = 0.0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    compression_saved_tokens: int = 0


class DailyCostSavings(BaseModel):
    date: str
    metrics: CostSavingsMetrics


class CostSavingsActivityResponse(BaseModel):
    results: list[DailyCostSavings]
    totals: CostSavingsMetrics = Field(default_factory=CostSavingsMetrics)
    unpriced_models: list[str] = Field(
        default_factory=list,
        description="Models with optimized tokens in the window but no usable prices in the model cost map; "
        "their savings are reported as 0",
    )


class OptimizedRequestSummary(BaseModel):
    request_id: str
    start_time: str
    model: str
    total_tokens: int
    optimizations: list[OptimizationType]
    original_cost: float
    optimized_cost: float
    savings: float


class RecentOptimizedRequestsResponse(BaseModel):
    requests: list[OptimizedRequestSummary]
    scanned_requests: int = Field(
        description="Number of most-recent requests in the window scanned for optimizations; "
        "optimized requests older than the scan window are not listed"
    )
