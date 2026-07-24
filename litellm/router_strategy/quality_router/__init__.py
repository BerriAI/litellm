"""
Quality-tier auto-router.

Re-uses the ComplexityRouter's classification to decide a request's complexity,
then maps that complexity to an admin-configured quality tier and resolves the
target model from each candidate's `model_info.litellm_routing_preferences`.
"""

from .config import (
    DEFAULT_COMPLEXITY_TO_QUALITY,
    QualityRouterConfig,
    RoutingPreferences,
)
from .quality_router import QualityRouter

__all__ = [
    "QualityRouter",
    "QualityRouterConfig",
    "RoutingPreferences",
    "DEFAULT_COMPLEXITY_TO_QUALITY",
]
