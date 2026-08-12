"""
Complexity-based Auto Router

A rule-based routing strategy that uses weighted scoring across multiple dimensions
to classify requests by complexity and route them to appropriate models.

No external API calls - all scoring is local and <1ms.
"""

from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
    classification_system_prompt,
)
from litellm.router_strategy.complexity_router.config import (
    DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
    DEFAULT_COMPLEXITY_CONFIG,
    ComplexityRouterConfig,
    ComplexityTier,
    ReminderMarkerPair,
)

__all__ = [
    "DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE",
    "DEFAULT_COMPLEXITY_CONFIG",
    "ComplexityRouter",
    "ComplexityRouterConfig",
    "ComplexityTier",
    "ReminderMarkerPair",
    "classification_system_prompt",
]
