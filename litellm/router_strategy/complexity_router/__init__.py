"""
Complexity-based Auto Router

A rule-based routing strategy that uses weighted scoring across multiple dimensions
to classify requests by complexity and route them to appropriate models.

No external API calls - all scoring is local and <1ms.
"""

from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import (
    ComplexityTier,
    DEFAULT_COMPLEXITY_CONFIG,
    ComplexityRouterConfig,
)

__all__ = [
    "ComplexityRouter",
    "ComplexityTier",
    "DEFAULT_COMPLEXITY_CONFIG",
    "ComplexityRouterConfig",
]
