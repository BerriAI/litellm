"""
Auto-Routing Strategy using regex-based classification and tier-based routing.

Replaces the previous semantic_router-based approach with a zero-dependency
regex + tier classification system (ported from ClawRouter).
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger

from .smart_router import SmartRouter
from .tiers import RoutingConfig, load_routing_config

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


class AutoRouter(CustomLogger):
    def __init__(
        self,
        model_name: str,
        default_model: str,
        litellm_router_instance: "Router",
        auto_router_config_path: Optional[str] = None,
        auto_router_config: Optional[str] = None,
    ):
        """
        Auto-Router class that uses regex classification + tier routing to
        route requests to the appropriate model.

        Args:
            model_name: The name of the model to use for the auto-router.
                e.g. if model = "auto-router1" then use this router.
            default_model: The default model to use if no route is found.
            litellm_router_instance: The instance of the LiteLLM Router.
            auto_router_config_path: Path to a YAML config file with routing overrides.
            auto_router_config: Inline JSON or YAML string with routing config.
        """
        self.model_name = model_name
        self.default_model = default_model
        self.litellm_router_instance: "Router" = litellm_router_instance

        # Build the routing config from file or inline config
        routing_config = self._build_routing_config(
            auto_router_config_path, auto_router_config
        )
        self.smart_router = SmartRouter(routing_config=routing_config)

    def _build_routing_config(
        self,
        config_path: Optional[str],
        config_str: Optional[str],
    ) -> RoutingConfig:
        """Build RoutingConfig from a file path or inline config string."""
        if config_path:
            return load_routing_config(config_path)
        elif config_str:
            return self._parse_inline_config(config_str)
        else:
            # Use defaults — no config required for regex-based routing
            return RoutingConfig()

    @staticmethod
    def _parse_inline_config(config_str: str) -> RoutingConfig:
        """Parse an inline JSON or YAML string into a RoutingConfig.

        Supports both JSON and YAML formats. The config should have the same
        structure as the YAML config file (tiers, routing, custom_patterns).
        """
        raw: Optional[dict] = None

        # Try JSON first
        try:
            raw = json.loads(config_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try YAML if JSON failed
        if raw is None:
            try:
                import yaml

                raw = yaml.safe_load(config_str)
            except Exception:
                pass

        if not isinstance(raw, dict):
            verbose_router_logger.warning(
                "Could not parse inline auto_router_config, using defaults"
            )
            return RoutingConfig()

        # Build config from parsed dict (same logic as load_routing_config)
        from .tiers import (
            DEFAULT_TIER_MODELS,
            ModelTier,
            TierConfig,
        )
        from .classifier import TaskCategory

        config = RoutingConfig()

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
                                "model", DEFAULT_TIER_MODELS[tier].model
                            ),
                            max_cost_per_m_tokens=float(
                                tier_data.get(
                                    "max_cost_per_m_tokens",
                                    DEFAULT_TIER_MODELS[tier].max_cost_per_m_tokens,
                                )
                            ),
                            fallback_models=fallback,
                        )
                except (ValueError, KeyError):
                    pass

        # Load routing overrides
        if "routing" in raw and isinstance(raw["routing"], dict):
            for category_name, tier_name in raw["routing"].items():
                try:
                    category = TaskCategory(category_name)
                    tier = ModelTier(tier_name)
                    config.category_routing[category] = tier
                except ValueError:
                    pass

        # Load custom patterns
        if "custom_patterns" in raw and isinstance(raw["custom_patterns"], list):
            config.custom_patterns = raw["custom_patterns"]

        return config

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        This hook is called before the routing decision is made.

        Uses regex-based classification to determine which model tier should
        handle the request, then returns the appropriate model.
        """
        from litellm.types.router import PreRoutingHookResponse

        if messages is None:
            return None

        decision = self.smart_router.resolve_route(messages)

        verbose_router_logger.debug(
            "Auto-router decision: model=%s, tier=%s, category=%s, confidence=%.2f",
            decision.model,
            decision.tier,
            decision.category,
            decision.confidence,
        )

        return PreRoutingHookResponse(
            model=decision.model,
            messages=messages,
        )
