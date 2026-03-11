from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GroupByDimension(str, Enum):
    DATE = "date"
    MODEL = "model"
    API_KEY = "api_key"
    TEAM = "team"
    ORGANIZATION = "organization"
    MODEL_GROUP = "model_group"
    PROVIDER = "custom_llm_provider"


class SpendMetrics(BaseModel):
    spend: float = Field(default=0.0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    cache_read_input_tokens: int = Field(default=0)
    cache_creation_input_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    api_requests: int = Field(default=0)


class MetricBase(BaseModel):
    metrics: SpendMetrics


class KeyMetadata(BaseModel):
    """Metadata for a key"""

    key_alias: Optional[str] = None
    team_id: Optional[str] = None


class KeyMetricWithMetadata(MetricBase):
    """Base class for metrics with additional metadata"""

    metadata: KeyMetadata = Field(default_factory=KeyMetadata)


class MetricWithMetadata(MetricBase):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # API key breakdown for this metric (e.g., which API keys are using this MCP server)
    api_key_breakdown: Dict[str, KeyMetricWithMetadata] = Field(
        default_factory=dict
    )  # api_key -> {metrics, metadata}


class BreakdownMetrics(BaseModel):
    """Breakdown of spend by different dimensions"""

    mcp_servers: Dict[str, MetricWithMetadata] = Field(
        default_factory=dict
    )  # mcp_server -> {metrics, metadata}
    models: Dict[str, MetricWithMetadata] = Field(
        default_factory=dict
    )  # model -> {metrics, metadata}
    model_groups: Dict[str, MetricWithMetadata] = Field(
        default_factory=dict
    )  # model_group -> {metrics, metadata}
    providers: Dict[str, MetricWithMetadata] = Field(
        default_factory=dict
    )  # provider -> {metrics, metadata}
    endpoints: Dict[str, MetricWithMetadata] = Field(
        default_factory=dict
    )  # endpoint -> {metrics, metadata}
    api_keys: Dict[str, KeyMetricWithMetadata] = Field(
        default_factory=dict
    )  # api_key -> {metrics, metadata}
    entities: Dict[str, MetricWithMetadata] = Field(
        default_factory=dict
    )  # entity -> {metrics, metadata}


class DailySpendData(BaseModel):
    date: date
    metrics: SpendMetrics
    breakdown: BreakdownMetrics = Field(default_factory=BreakdownMetrics)


class DailySpendMetadata(BaseModel):
    total_spend: float = Field(default=0.0)
    total_prompt_tokens: int = Field(default=0)
    total_completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_api_requests: int = Field(default=0)
    total_successful_requests: int = Field(default=0)
    total_failed_requests: int = Field(default=0)
    total_cache_read_input_tokens: int = Field(default=0)
    total_cache_creation_input_tokens: int = Field(default=0)
    page: int = Field(default=1)
    total_pages: int = Field(default=1)
    has_more: bool = Field(default=False)


class SpendAnalyticsPaginatedResponse(BaseModel):
    results: List[DailySpendData]
    metadata: DailySpendMetadata = Field(default_factory=DailySpendMetadata)


class LiteLLM_DailyUserSpend(BaseModel):
    id: str
    user_id: str
    date: str
    api_key: str
    mcp_server_id: Optional[str] = None
    model: Optional[str] = None
    model_group: Optional[str] = None
    custom_llm_provider: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    spend: float = 0.0
    api_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class GroupedData(TypedDict):
    metrics: SpendMetrics
    breakdown: BreakdownMetrics
