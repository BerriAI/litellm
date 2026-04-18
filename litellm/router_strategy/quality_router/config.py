"""
Configuration models for the QualityRouter.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Default mapping from ComplexityTier name (string) to quality tier (int).
# Higher tier = higher capability requirement.
DEFAULT_COMPLEXITY_TO_QUALITY: Dict[str, int] = {
    "SIMPLE": 1,
    "MEDIUM": 2,
    "COMPLEX": 3,
    "REASONING": 4,
}


class QualityRouterConfig(BaseModel):
    """Configuration for the QualityRouter."""

    available_models: List[str] = Field(
        default_factory=list,
        description=(
            "List of candidate model names this router may route to. Each model "
            "must declare its quality_tier in model_info.litellm_routing_preferences."
        ),
    )

    default_model: Optional[str] = Field(
        default=None,
        description="Fallback model when no quality tier resolves.",
    )

    complexity_to_quality: Dict[str, int] = Field(
        default_factory=lambda: DEFAULT_COMPLEXITY_TO_QUALITY.copy(),
        description="Mapping from ComplexityTier name to quality tier (int).",
    )

    model_config = ConfigDict(extra="allow")


class RoutingPreferences(BaseModel):
    """Per-deployment routing preferences declared on model_info."""

    quality_tier: int = Field(
        ...,
        description="The quality tier this deployment satisfies.",
    )

    keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Substring keywords (case-insensitive) that, when present in the "
            "user message, route the request to this deployment. See `order` "
            "for explicit collision handling, otherwise ties fall through to "
            "(highest quality_tier, then cheapest model_info.input_cost_per_token)."
        ),
    )

    order: Optional[int] = Field(
        default=None,
        description=(
            "Explicit priority used to break ties between deployments at the "
            "same quality tier. Lower values win. Applies both to keyword "
            "collisions and to picking between multiple deployments at the "
            "same quality_tier. Tiebreak order is "
            "(quality_tier DESC, order ASC, input_cost_per_token ASC, "
            "model_name ASC) — quality always wins first, then explicit "
            "order, then price."
        ),
    )

    model_config = ConfigDict(extra="allow")
