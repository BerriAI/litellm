from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class GuardrailMetrics(BaseModel):
    """Aggregated metrics for a guardrail."""

    total_requests: int = 0
    success_count: int = 0
    intervened_count: int = 0
    failed_count: int = 0
    not_run_count: int = 0
    fail_rate: float = 0.0  # percentage
    avg_latency_ms: float = 0.0


class GuardrailSummary(BaseModel):
    """Summary view for guardrails table."""

    guardrail_name: str
    provider: str
    total_requests: int
    fail_rate: float  # percentage
    avg_latency_ms: float


class GuardrailMetricsResponse(BaseModel):
    """Response for /guardrail/metrics endpoint."""

    results: List[GuardrailSummary]
    metadata: Dict[str, Any]


class GuardrailDailyMetrics(BaseModel):
    """Daily time-series data for a guardrail."""

    date: str
    total_requests: int
    intervened_count: int
    success_count: int
    fail_rate: float
    avg_latency_ms: float


class GuardrailDetailMetrics(BaseModel):
    """Detailed metrics for guardrail overview page."""

    requests_evaluated: int
    fail_rate: float
    avg_latency_ms: float
    blocked_count: int  # intervened in selected period
    daily_metrics: List[GuardrailDailyMetrics]


class GuardrailLogEntry(BaseModel):
    """Individual request log entry."""

    request_id: str
    timestamp: str
    model: str
    status: str  # "blocked" or "passed"
    guardrail_response: Optional[dict] = None
    request_content: Optional[str] = None
    latency_ms: float = 0.0


class GuardrailLogsResponse(BaseModel):
    """Response for logs tab."""

    logs: List[GuardrailLogEntry]
    total_count: int
    page: int
    page_size: int
