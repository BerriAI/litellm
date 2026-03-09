"""
Competitor intent: entity + intent disambiguation with safe (non-competitor) defaults.

Base logic in base.py; industry-specific checkers in submodules (e.g. airline.py).
"""

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.competitor_intent.airline import \
    AirlineCompetitorIntentChecker
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.competitor_intent.base import (
    BaseCompetitorIntentChecker, normalize, text_for_entity_matching)

__all__ = [
    "BaseCompetitorIntentChecker",
    "AirlineCompetitorIntentChecker",
    "normalize",
    "text_for_entity_matching",
]
