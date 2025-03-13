"""This module """

from typing import Literal, Optional, TypedDict

class ThinkingConfig(TypedDict):
    type: Literal["enabled"]
    budget_tokens: int

class ParameterMapping:
    @staticmethod
    def map_reasoning_to_thinking(
        reasoning_effort: Optional[Literal["low", "medium", "high"]]
    ) -> Optional[ThinkingConfig]:
        if not reasoning_effort:
            return None
            
        mapping = {
            "low": {"type": "enabled", "budget_tokens": 8000},
            "medium": {"type": "enabled", "budget_tokens": 16000},
            "high": {"type": "enabled", "budget_tokens": 24000}
        }
        return mapping.get(reasoning_effort)