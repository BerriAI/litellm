"""Model tier definitions and routing table for auto-routing.

Defines the mapping from task categories to model tiers, and from
tiers to specific model IDs. Supports loading overrides from a
YAML config file.

Ported from ClawRouter's tiers with Python 3.9 compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_router_logger

from .classifier import TaskCategory


class ModelTier(str, Enum):
    """Model cost tiers."""

    LOW = "low"
    MID = "mid"
    TOP = "top"


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single model tier."""

    model: str
    max_cost_per_m_tokens: float
    fallback_models: Tuple[str, ...] = ()


# Hardcoded defaults (used when no config file is found)
DEFAULT_TIER_MODELS: Dict[ModelTier, TierConfig] = {
    ModelTier.LOW: TierConfig(
        model="deepseek/deepseek-chat-v3-0324",
        max_cost_per_m_tokens=1.0,
    ),
    ModelTier.MID: TierConfig(
        model="zai/glm-4.7",
        max_cost_per_m_tokens=5.0,
        fallback_models=("google/gemini-2.5-flash",),
    ),
    ModelTier.TOP: TierConfig(
        model="anthropic/claude-sonnet-4-5",
        max_cost_per_m_tokens=30.0,
    ),
}

# Default category -> tier mapping
DEFAULT_ROUTING: Dict[TaskCategory, ModelTier] = {
    TaskCategory.HEARTBEAT: ModelTier.LOW,
    TaskCategory.SIMPLE_CHAT: ModelTier.LOW,
    TaskCategory.LOOKUP: ModelTier.LOW,
    TaskCategory.TRANSLATION: ModelTier.MID,
    TaskCategory.SUMMARIZATION: ModelTier.MID,
    TaskCategory.CODING: ModelTier.MID,
    TaskCategory.CREATIVE: ModelTier.MID,
    TaskCategory.REASONING: ModelTier.TOP,
    TaskCategory.ANALYSIS: ModelTier.TOP,
}


@dataclass
class RoutingConfig:
    """Full routing configuration loaded from defaults + YAML overrides."""

    tier_models: Dict[ModelTier, TierConfig] = field(
        default_factory=lambda: dict(DEFAULT_TIER_MODELS)
    )
    category_routing: Dict[TaskCategory, ModelTier] = field(
        default_factory=lambda: dict(DEFAULT_ROUTING)
    )
    custom_patterns: List[Dict[str, str]] = field(default_factory=list)

    def get_model_for_category(self, category: TaskCategory) -> str:
        """Get the model ID for a given task category."""
        tier = self.category_routing.get(category, ModelTier.MID)
        tier_config = self.tier_models.get(tier, DEFAULT_TIER_MODELS[ModelTier.MID])
        return tier_config.model

    def get_tier_for_category(self, category: TaskCategory) -> ModelTier:
        """Get the tier for a given task category."""
        return self.category_routing.get(category, ModelTier.MID)

    def get_tier_config(self, tier: ModelTier) -> TierConfig:
        """Get the configuration for a specific tier."""
        return self.tier_models.get(tier, DEFAULT_TIER_MODELS[tier])


def load_routing_config(config_path: Optional[str] = None) -> RoutingConfig:
    """Load routing configuration from YAML file, falling back to defaults.

    Args:
        config_path: Path to routing_rules.yaml. If None, uses defaults.

    Returns:
        RoutingConfig with merged defaults and overrides.
    """
    config = RoutingConfig()

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        verbose_router_logger.debug(
            "Routing config file not found, using defaults: %s", str(path)
        )
        return config

    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            verbose_router_logger.warning(
                "Invalid routing config format, using defaults"
            )
            return config

        # Load tier overrides
        if "tiers" in raw and isinstance(raw["tiers"], dict):
            for tier_name, tier_data in raw["tiers"].items():
                try:
                    tier = ModelTier(tier_name)
                    if isinstance(tier_data, dict):
                        fallback_raw = tier_data.get("fallback_models")
                        fallback = (
                            tuple(fallback_raw)
                            if isinstance(fallback_raw, list)
                            else DEFAULT_TIER_MODELS[tier].fallback_models
                        )
                        config.tier_models[tier] = TierConfig(
                            model=tier_data.get(
                                "model",
                                DEFAULT_TIER_MODELS[tier].model,
                            ),
                            max_cost_per_m_tokens=float(
                                tier_data.get(
                                    "max_cost_per_m_tokens",
                                    DEFAULT_TIER_MODELS[tier].max_cost_per_m_tokens,
                                )
                            ),
                            fallback_models=fallback,
                        )
                except (ValueError, KeyError) as e:
                    verbose_router_logger.warning(
                        "Skipping invalid tier config for %s: %s",
                        tier_name,
                        str(e),
                    )

        # Load routing overrides
        if "routing" in raw and isinstance(raw["routing"], dict):
            for category_name, tier_name in raw["routing"].items():
                try:
                    category = TaskCategory(category_name)
                    tier = ModelTier(tier_name)
                    config.category_routing[category] = tier
                except ValueError as e:
                    verbose_router_logger.warning(
                        "Skipping invalid routing rule %s -> %s: %s",
                        category_name,
                        tier_name,
                        str(e),
                    )

        # Load custom patterns
        if "custom_patterns" in raw and isinstance(raw["custom_patterns"], list):
            config.custom_patterns = raw["custom_patterns"]

        verbose_router_logger.info(
            "Loaded routing config from %s (tiers=%d, routes=%d, custom_patterns=%d)",
            str(path),
            len(config.tier_models),
            len(config.category_routing),
            len(config.custom_patterns),
        )

    except Exception as e:
        verbose_router_logger.error(
            "Failed to load routing config from %s, using defaults: %s",
            str(path),
            str(e),
        )

    return config
