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

    capabilities: List[str] = Field(
        default_factory=list,
        description=(
            "Capability tags this deployment supports (e.g. 'vision', "
            "'function_calling', 'json_mode'). The QualityRouter will only "
            "route to deployments whose capabilities are a superset of any "
            "capabilities required by the request."
        ),
    )

    model_config = ConfigDict(extra="allow")
