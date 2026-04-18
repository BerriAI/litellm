"""
Quality-tier Auto Router.

Routes a request to a model at a target quality tier. The quality tier is
inferred by re-using the existing ComplexityRouter's classification, then
mapped through an admin-configured `complexity_to_quality` table. Each
candidate model declares its own `quality_tier` in
`model_info.litellm_routing_preferences`.
"""

from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
)

from .config import QualityRouterConfig

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


class QualityRouter(CustomLogger):
    """
    Routes requests to a model at a target quality tier.

    Pipeline:
        1. Classify the user message via ComplexityRouter to get a ComplexityTier.
        2. Map that tier name to a target quality tier (int) via
           `config.complexity_to_quality`.
        3. Resolve the target quality tier to a concrete model using the
           per-deployment `quality_tier` declared in
           `model_info.litellm_routing_preferences`.
    """

    def __init__(
        self,
        model_name: str,
        litellm_router_instance: "Router",
        default_model: Optional[str] = None,
        quality_router_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance

        if quality_router_config:
            self.config = QualityRouterConfig(**quality_router_config)
        else:
            self.config = QualityRouterConfig()

        # Explicit default_model arg overrides anything in the config dict.
        if default_model:
            self.config.default_model = default_model

        # Internal scorer — re-use the existing rule-based classifier.
        self._scorer = ComplexityRouter(
            model_name=f"{model_name}::scorer",
            litellm_router_instance=litellm_router_instance,
        )

        # Pre-built tier → models index for O(1) resolution. Capabilities are
        # tracked separately so resolution can filter by required capabilities
        # without complicating the tier-walk loop.
        self._model_capabilities: Dict[str, FrozenSet[str]] = {}
        self._tier_to_models: Dict[int, List[str]] = self._build_tier_index()

        verbose_router_logger.debug(
            f"QualityRouter initialized for {model_name} with "
            f"available_models={self.config.available_models}, "
            f"default_model={self.config.default_model}, "
            f"tier_index={self._tier_to_models}"
        )

    def _get_routing_preferences(self, deployment: Any) -> Optional[Dict[str, Any]]:
        """
        Extract litellm_routing_preferences from a deployment, handling both
        dict-shaped and Pydantic-object-shaped deployments.
        """
        # Dict-shaped deployment.
        if isinstance(deployment, dict):
            model_info = deployment.get("model_info") or {}
            if isinstance(model_info, dict):
                return model_info.get("litellm_routing_preferences")
            # Pydantic ModelInfo nested in a dict.
            return getattr(model_info, "litellm_routing_preferences", None)

        # Pydantic-object deployment.
        model_info = getattr(deployment, "model_info", None)
        if model_info is None:
            return None
        if isinstance(model_info, dict):
            return model_info.get("litellm_routing_preferences")
        return getattr(model_info, "litellm_routing_preferences", None)

    def _get_deployment_model_name(self, deployment: Any) -> Optional[str]:
        """Extract `model_name` from a dict- or object-shaped deployment."""
        if isinstance(deployment, dict):
            return deployment.get("model_name")
        return getattr(deployment, "model_name", None)

    def _build_tier_index(self) -> Dict[int, List[str]]:
        """
        Build {quality_tier: [model_name, ...]} for every model in
        `available_models`. Raises if any listed model is missing
        `litellm_routing_preferences`.
        """
        model_list = getattr(self.litellm_router_instance, "model_list", None) or []
        available = set(self.config.available_models)

        # Track which available models we've matched so we can error on missing.
        seen: Dict[str, bool] = {name: False for name in available}
        tier_to_models: Dict[int, List[str]] = {}

        for deployment in model_list:
            name = self._get_deployment_model_name(deployment)
            if name is None or name not in available:
                continue

            prefs = self._get_routing_preferences(deployment)
            if prefs is None:
                raise ValueError(
                    f"QualityRouter: model '{name}' is listed in available_models "
                    f"but has no model_info.litellm_routing_preferences"
                )

            # Accept dict or Pydantic-shaped prefs.
            if isinstance(prefs, dict):
                tier = prefs.get("quality_tier")
                capabilities = prefs.get("capabilities") or []
            else:
                tier = getattr(prefs, "quality_tier", None)
                capabilities = getattr(prefs, "capabilities", None) or []

            if tier is None:
                raise ValueError(
                    f"QualityRouter: model '{name}' has litellm_routing_preferences "
                    f"but no quality_tier field"
                )

            tier_int = int(tier)
            tier_to_models.setdefault(tier_int, []).append(name)
            self._model_capabilities[name] = frozenset(capabilities)
            seen[name] = True

        missing = [name for name, found in seen.items() if not found]
        if missing:
            raise ValueError(
                f"QualityRouter: the following available_models are not present in "
                f"the router's model_list (or are missing routing preferences): {missing}"
            )

        return tier_to_models

    def _model_supports_capabilities(
        self, model_name: str, required: FrozenSet[str]
    ) -> bool:
        """True if the model's declared capabilities are a superset of required."""
        if not required:
            return True
        return required.issubset(self._model_capabilities.get(model_name, frozenset()))

    def _first_capable_model_at_tier(
        self, tier: int, required: FrozenSet[str]
    ) -> Optional[str]:
        """First model at `tier` that supports all `required` capabilities, or None."""
        for name in self._tier_to_models.get(tier, []):
            if self._model_supports_capabilities(name, required):
                return name
        return None

    def _resolve_model_for_quality_tier(
        self,
        tier: int,
        required_capabilities: Optional[Set[str]] = None,
    ) -> str:
        """
        Resolve a quality tier to a concrete model name.

        Strategy:
            1. Exact tier match → first capability-matching model at that tier.
            2. Otherwise round up to the next higher tier that has a
               capability-matching model.
            3. Otherwise fall back to `config.default_model` — but only if it
               also satisfies required capabilities. Routing to a model that
               lacks a required capability would silently produce wrong results.
        """
        required: FrozenSet[str] = (
            frozenset(required_capabilities) if required_capabilities else frozenset()
        )

        match = self._first_capable_model_at_tier(tier, required)
        if match is not None:
            return match

        higher_tiers = sorted(t for t in self._tier_to_models if t > tier)
        for t in higher_tiers:
            match = self._first_capable_model_at_tier(t, required)
            if match is not None:
                return match

        if self.config.default_model and self._model_supports_capabilities(
            self.config.default_model, required
        ):
            return self.config.default_model

        if required:
            raise ValueError(
                f"QualityRouter: no model satisfies quality tier {tier} with "
                f"required capabilities {sorted(required)}"
            )

        if self.config.default_model:
            return self.config.default_model

        raise ValueError(
            f"QualityRouter: no model available for quality tier {tier} and "
            f"no default_model configured"
        )

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """Classify the request, map to a quality tier, resolve the model."""
        from litellm.types.router import PreRoutingHookResponse

        if messages is None or len(messages) == 0:
            verbose_router_logger.debug(
                "QualityRouter: No messages provided, skipping routing"
            )
            return None

        # Extract last user message and last system prompt — same rules as
        # ComplexityRouter.async_pre_routing_hook.
        user_message: Optional[str] = None
        system_prompt: Optional[str] = None

        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = " ".join(text_parts).strip()
            if isinstance(content, str) and content:
                if role == "user" and user_message is None:
                    user_message = content
                elif role == "system" and system_prompt is None:
                    system_prompt = content

        # Required capabilities are an optional client-side override.
        # Accept either an iterable of strings or None. Anything else is ignored
        # rather than raising — matches the lenient style of other router params.
        raw_caps = (request_kwargs or {}).get("litellm_capabilities")
        required_capabilities: Optional[Set[str]] = (
            {str(c) for c in raw_caps}
            if isinstance(raw_caps, (list, tuple, set, frozenset)) and raw_caps
            else None
        )

        if user_message is None:
            verbose_router_logger.debug(
                "QualityRouter: No user message found, routing to default model"
            )
            if not self.config.default_model:
                raise ValueError(
                    "QualityRouter: no user message and no default_model configured"
                )
            if required_capabilities and not self._model_supports_capabilities(
                self.config.default_model, frozenset(required_capabilities)
            ):
                raise ValueError(
                    f"QualityRouter: no user message and default_model "
                    f"'{self.config.default_model}' does not satisfy required "
                    f"capabilities {sorted(required_capabilities)}"
                )
            return PreRoutingHookResponse(
                model=self.config.default_model,
                messages=messages,
            )

        complexity_tier, score, signals = self._scorer.classify(
            user_message, system_prompt
        )
        complexity_name = (
            complexity_tier.value
            if hasattr(complexity_tier, "value")
            else str(complexity_tier)
        )

        quality_tier = self.config.complexity_to_quality.get(complexity_name)
        if quality_tier is None:
            raise ValueError(
                f"QualityRouter: complexity tier '{complexity_name}' not present "
                f"in complexity_to_quality mapping {self.config.complexity_to_quality}"
            )

        routed_model = self._resolve_model_for_quality_tier(
            int(quality_tier), required_capabilities=required_capabilities
        )

        verbose_router_logger.info(
            f"QualityRouter: complexity={complexity_name}, score={score:.3f}, "
            f"signals={signals}, quality_tier={quality_tier}, "
            f"required_capabilities={sorted(required_capabilities) if required_capabilities else []}, "
            f"routed_model={routed_model}"
        )

        return PreRoutingHookResponse(
            model=routed_model,
            messages=messages,
        )
