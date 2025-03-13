"""This module holds mappings for OpenAI parameters, which may have different mappings in other models.

It provides the `ParameterMapping` class, which contains methods for converting OpenAI-specific 
parameters into a structured format. These mappings help standardize configurations across 
different models that may use different parameter schemes.


Methods:
    - ParameterMapping.map_reasoning_to_thinking(reasoning_effort): 
      Maps OpenAI's reasoning effort levels to a `ThinkingConfig`, specifying a token budget.

"""

from typing import Literal, Optional, TypedDict, cast

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

        mapping: dict[str, ThinkingConfig] = {
            "low": {"type": "enabled", "budget_tokens": 8000},
            "medium": {"type": "enabled", "budget_tokens": 16000},
            "high": {"type": "enabled", "budget_tokens": 24000}
        }
        return cast(Optional[ThinkingConfig], mapping.get(reasoning_effort))