"""
Auto-router per-session rollup model.

Canonical definition for ``litellm_autoroutersession``, the row the spend flush
maintains per (api_key, session_id, router_name).
"""

from collections.abc import Mapping
from datetime import datetime

from pydantic import Field

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_AutoRouterSession(LiteLLMPydanticObjectBase):
    api_key: str
    session_id: str
    router_name: str
    router_type: str
    first_turn_at: datetime
    last_turn_at: datetime
    last_model: str
    turns: int
    spend: float
    saved_spend: float
    savings_estimated_turns: int = 0
    savings_estimated_actual_spend: float = 0.0
    savings_estimated_saved_spend: float = 0.0
    savings_estimated_baseline_models: Mapping[str, int] = Field(default_factory=dict)
    classifier_cost: float
    tier_turns: Mapping[str, int]
    baseline_models: Mapping[str, int]

    @property
    def baseline_model(self) -> str | None:
        """A recorded baseline label when excluded turns cannot change the selected model."""
        if not self.baseline_models:
            return None
        if self.savings_estimated_turns < self.turns and len(self.baseline_models) > 1:
            return None
        return max(
            self.baseline_models,
            key=lambda model: (self.baseline_models[model], model),
        )
