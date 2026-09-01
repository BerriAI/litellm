"""
Complexity-based Auto Router

A rule-based routing strategy that uses weighted scoring across multiple dimensions
to classify requests by complexity and route them to appropriate models.

No external API calls - all scoring is local and <1ms.
"""

from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
    classification_system_prompt,
    custom_tier_classification_prompt,
)
from litellm.router_strategy.complexity_router.config import (
    DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
    DEFAULT_COMPLEXITY_CONFIG,
    ClassificationRubric,
    ComplexityRouterConfig,
    ComplexityTier,
    ReminderMarkerPair,
    TierDefinition,
    normalize_classification_prompt,
)

__all__ = [
    "DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE",
    "DEFAULT_COMPLEXITY_CONFIG",
    "ClassificationRubric",
    "ComplexityRouter",
    "ComplexityRouterConfig",
    "ComplexityTier",
    "ReminderMarkerPair",
    "TierDefinition",
    "classification_system_prompt",
    "custom_tier_classification_prompt",
    "normalize_classification_prompt",
]
