"""
Auto-router per-session rollup model.

Canonical definition for ``litellm_autoroutersession``, the row the spend flush
maintains per (api_key, session_id, router_name).
"""

from collections.abc import Mapping
from datetime import datetime

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
    classifier_cost: float
    tier_turns: Mapping[str, int]
    baseline_models: Mapping[str, int]

    @property
    def baseline_model(self) -> str | None:
        """The baseline most of this session's turns were priced against, or None when no turn recorded one.

        A router reconfigured mid-session leaves turns priced against two baselines; the row keeps both
        counts, and the label is the one that priced the most money-carrying turns rather than whatever the
        router is configured with now.
        """
        if not self.baseline_models:
            return None
        return max(self.baseline_models, key=lambda model: (self.baseline_models[model], model))
