from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt

TokenCount: TypeAlias = Annotated[StrictInt, Field(ge=0)]


class CacheTokenBuckets(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uncached_input_tokens: TokenCount = 0
    cache_read_input_tokens: TokenCount = 0
    cache_creation_5m_input_tokens: TokenCount = 0
    cache_creation_1h_input_tokens: TokenCount = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.uncached_input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_5m_input_tokens
            + self.cache_creation_1h_input_tokens
        )


class CacheEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    observed_at: float
    expires_at: float
    source: Literal["provider_usage"] = "provider_usage"
    confidence: Literal["observed"] = "observed"


class CacheCostScenario(BaseModel):
    tokens: CacheTokenBuckets
    input_cost: float


class CachePredictionArm(BaseModel):
    deployment_id: str
    model: str | None = None
    cache_state: Literal["warm", "partial", "stale", "unknown", "disabled"] = "unknown"
    reason: str | None = None
    estimate: CacheCostScenario | None = None
    cold: CacheCostScenario | None = None
    warm: CacheCostScenario | None = None
    evidence: CacheEvidence | None = None
    token_count_source: Literal["anthropic_count_tokens"] | None = None


class CachePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_deployment_id: str = Field(min_length=1, max_length=256)
    candidate_deployment_id: str = Field(min_length=1, max_length=256)
    request: Mapping[str, JsonValue]


class CachePredictionResponse(BaseModel):
    stay: CachePredictionArm
    switch: CachePredictionArm
    switch_delta: float | None
    cache_rebuild_penalty: float | None
    pricing_basis: Literal["input_before_discounts_and_margins"] = "input_before_discounts_and_margins"
    cache_guarantee: Literal[False] = False
