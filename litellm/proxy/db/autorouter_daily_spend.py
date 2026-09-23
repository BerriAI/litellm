from math import isclose
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

DAILY_COSTS_COMPLETE_SQL: Final = """
    autorouter_accounted_requests = api_requests
    AND autorouter_accounted_requests >= 0
    AND autorouter_requests BETWEEN 0 AND successful_requests
    AND autorouter_classifier_cost_recorded_requests BETWEEN 0 AND autorouter_requests
    AND autorouter_estimated_requests BETWEEN 0 AND autorouter_requests
    AND (autorouter_savings_spend = 0 OR autorouter_estimated_requests > 0)
"""

AUTOROUTER_DAILY_COSTS_SQL: Final = f"""
SELECT
    COALESCE(SUM(autorouter_requests), 0)::bigint AS requests,
    COALESCE(SUM(autorouter_llm_spend), 0)::float8 AS llm_spend,
    COALESCE(SUM(autorouter_classifier_cost), 0)::float8 AS classifier_cost,
    COALESCE(SUM(autorouter_classifier_cost_recorded_requests), 0)::bigint AS classifier_requests,
    COALESCE(SUM(autorouter_estimated_requests), 0)::bigint AS estimated_requests,
    COALESCE(SUM(autorouter_estimated_actual_spend), 0)::float8 AS estimated_actual_spend,
    COALESCE(SUM(autorouter_savings_spend), 0)::float8 AS saved_spend,
    COALESCE(BOOL_AND({DAILY_COSTS_COMPLETE_SQL}), TRUE) AS complete
FROM "LiteLLM_DailyUserSpend"
WHERE date >= $1::text AND date <= $2::text
  AND ($3::text IS NULL OR api_key = $3::text)
  AND ($4::text IS NULL OR user_id = $4::text)
"""


class AutoRouterDailyCosts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: int = 0
    llm_spend: float = 0.0
    classifier_cost: float = 0.0
    classifier_requests: int = 0
    estimated_requests: int = 0
    estimated_actual_spend: float = 0.0
    saved_spend: float = 0.0
    complete: bool = True
    comparison_complete: bool = True

    def matches_savings(self, saved_spend: float) -> bool:
        return self.complete and isclose(self.saved_spend, saved_spend, rel_tol=1e-9, abs_tol=1e-9)

    @property
    def classifier_complete(self) -> bool:
        return self.classifier_requests == self.requests

    @property
    def coverage(self) -> Literal["complete", "partial", "unavailable"]:
        if self.complete and self.classifier_complete:
            return "complete"
        return "partial" if self.requests > 0 else "unavailable"

    @property
    def recorded_llm_spend(self) -> float | None:
        return self.llm_spend if self.complete or self.requests > 0 else None

    @property
    def recorded_classifier_cost(self) -> float | None:
        return self.classifier_cost if self.classifier_complete and self.recorded_llm_spend is not None else None

    @property
    def recorded_spend(self) -> float | None:
        return self.llm_spend + self.classifier_cost if self.recorded_llm_spend is not None else None

    def baseline_spend(self, saved_spend: float) -> float | None:
        if not self.matches_savings(saved_spend) or not self.classifier_complete or not self.comparison_complete:
            return None
        if self.requests > 0 and self.estimated_requests == 0:
            return None
        return self.estimated_actual_spend + saved_spend
