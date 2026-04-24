"""
Quality-tier Auto Router.

Routes a request to a model at a target quality tier. The quality tier is
inferred by re-using the existing ComplexityRouter's classification, then
mapped through an admin-configured `complexity_to_quality` table. Each
candidate model declares its own `quality_tier` in
`model_info.litellm_routing_preferences`.

Optional keyword override: deployments may also declare `keywords` in
`litellm_routing_preferences`. If any declared keyword appears in the user
message (case-insensitive substring match), the router short-circuits the
complexity-classification flow and routes to the matching deployment. When
multiple deployments match, ties are broken by (highest quality_tier first,
then cheapest `model_info.input_cost_per_token`).
"""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
)

from .config import QualityRouterConfig, RoutingPreferences

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


class QualityRouter(CustomLogger):
    """
    Routes requests to a model at a target quality tier, with an optional
    keyword override.
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

        # Per-model indices populated alongside the tier index. `_model_keywords`
        # stores keywords lowercased so we can substring-match against the
        # lowercased user message in O(total-keyword-count). `_model_quality`,
        # `_model_cost`, and `_model_order` drive tiebreaking — `_model_order`
        # is the explicit priority (lower wins, unset = +inf).
        self._model_keywords: Dict[str, List[str]] = {}
        self._model_quality: Dict[str, int] = {}
        self._model_cost: Dict[str, Optional[float]] = {}
        self._model_order: Dict[str, Optional[int]] = {}

        # Tier → models index. Built lazily on first access so the QualityRouter
        # deployment does NOT need to appear after all its referenced models in
        # the config — when `_build_tier_index` runs eagerly in `__init__`, the
        # router instance's `model_list` is still being assembled incrementally
        # by `_create_deployment`, and any `available_models` defined AFTER the
        # router entry in config.yaml would silently be reported as missing.
        self._tier_to_models_cache: Optional[Dict[int, List[str]]] = None

        verbose_router_logger.debug(
            f"QualityRouter initialized for {model_name} with "
            f"available_models={self.config.available_models}, "
            f"default_model={self.config.default_model}"
        )

    @property
    def _tier_to_models(self) -> Dict[int, List[str]]:
        """Lazy tier→models index; built on first access."""
        if self._tier_to_models_cache is None:
            self._tier_to_models_cache = self._build_tier_index()
        return self._tier_to_models_cache

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

    def _get_deployment_input_cost(self, deployment: Any) -> Optional[float]:
        """
        Extract `input_cost_per_token` from a deployment's model_info.

        Returns None when not declared — None is treated as "infinite cost"
        for the cheapest-tiebreak ordering, so unpriced models lose ties to
        priced ones. (Admins who want a model to win on price must declare it.)
        """
        if isinstance(deployment, dict):
            model_info = deployment.get("model_info") or {}
        else:
            model_info = getattr(deployment, "model_info", None) or {}

        if isinstance(model_info, dict):
            cost = model_info.get("input_cost_per_token")
        else:
            cost = getattr(model_info, "input_cost_per_token", None)

        if cost is None:
            return None
        try:
            return float(cost)
        except (TypeError, ValueError):
            return None

    def _get_deployment_model_name(self, deployment: Any) -> Optional[str]:
        """Extract `model_name` from a dict- or object-shaped deployment."""
        if isinstance(deployment, dict):
            return deployment.get("model_name")
        return getattr(deployment, "model_name", None)

    def _build_tier_index(self) -> Dict[int, List[str]]:
        """
        Build {quality_tier: [model_name, ...]} for every model in
        `available_models`, plus side indices `_model_keywords`,
        `_model_quality`, and `_model_cost`. Raises if any listed model is
        missing `litellm_routing_preferences`.
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

            raw_prefs = self._get_routing_preferences(deployment)
            if raw_prefs is None:
                raise ValueError(
                    f"QualityRouter: model '{name}' is listed in available_models "
                    f"but has no model_info.litellm_routing_preferences"
                )

            # Validate via the Pydantic model so we get a clear error for
            # missing quality_tier, wrong types, etc. This also means
            # `RoutingPreferences` is the single source of truth for the
            # accepted shape — readers relied on raw dicts before.
            try:
                if isinstance(raw_prefs, RoutingPreferences):
                    prefs = raw_prefs
                elif isinstance(raw_prefs, dict):
                    prefs = RoutingPreferences(**raw_prefs)
                else:
                    # A Pydantic object of some other shape — coerce via its dict.
                    prefs = RoutingPreferences(
                        **(
                            raw_prefs.model_dump()
                            if hasattr(raw_prefs, "model_dump")
                            else dict(raw_prefs)
                        )
                    )
            except Exception as e:
                raise ValueError(
                    f"QualityRouter: model '{name}' has invalid "
                    f"litellm_routing_preferences: {e}"
                ) from e

            tier_int = int(prefs.quality_tier)
            tier_to_models.setdefault(tier_int, []).append(name)
            self._model_keywords[name] = [str(k).lower() for k in prefs.keywords if k]
            self._model_quality[name] = tier_int
            self._model_cost[name] = self._get_deployment_input_cost(deployment)
            self._model_order[name] = prefs.order
            seen[name] = True

        missing = [name for name, found in seen.items() if not found]
        if missing:
            raise ValueError(
                f"QualityRouter: the following available_models are not present in "
                f"the router's model_list (or are missing routing preferences): {missing}"
            )

        # Sort each tier's model list so `_resolve_model_for_quality_tier`
        # (which picks index [0]) honors (order ASC, cost ASC, name ASC).
        # Quality is moot within a single tier; keep parity with the keyword
        # tiebreak by ordering on (order, cost, name) here.
        for models in tier_to_models.values():
            models.sort(key=lambda n: (self._order_key(n), self._cost_key(n), n))

        return tier_to_models

    def _order_key(self, model_name: str) -> float:
        """`order` lookup as a float — unset becomes +inf so explicit wins."""
        order = self._model_order.get(model_name)
        return float(order) if order is not None else math.inf

    def _cost_key(self, model_name: str) -> float:
        """`input_cost_per_token` as a float — unset becomes +inf."""
        cost = self._model_cost.get(model_name)
        return float(cost) if cost is not None else math.inf

    def _keyword_override(self, user_message: str) -> Optional[Tuple[str, str]]:
        """
        Find a deployment whose declared keywords appear in `user_message`.

        Returns (model_name, matched_keyword) or None when no keyword matches.
        When multiple deployments match, sorts by:
            1. quality_tier DESC (best quality always wins first)
            2. `order` ASC (explicit priority — unset = +inf so explicit wins
               within the same tier)
            3. input_cost_per_token ASC (unpriced = +inf so priced wins)
            4. model_name ASC (deterministic stability)
        """
        # Touch the lazy index so `_model_keywords` / `_model_quality` /
        # `_model_cost` / `_model_order` are populated.
        _ = self._tier_to_models

        text = user_message.lower()

        matches: List[Tuple[str, str]] = []  # (model_name, matched_keyword)
        for model_name, keywords in self._model_keywords.items():
            for kw in keywords:
                if kw and kw in text:
                    matches.append((model_name, kw))
                    break  # one match per model is enough

        if not matches:
            return None

        def sort_key(match: Tuple[str, str]) -> Tuple[int, float, float, str]:
            name = match[0]
            quality = self._model_quality.get(name, 0)
            order_val = self._order_key(name)
            cost = self._model_cost.get(name)
            cost_val = cost if cost is not None else math.inf
            # Negate quality so higher tier sorts first under ASC sort.
            return (-quality, order_val, cost_val, name)

        matches.sort(key=sort_key)
        return matches[0]

    def _resolve_model_for_quality_tier(self, tier: int) -> str:
        """
        Resolve a quality tier to a concrete model name.

        Strategy:
            1. Exact tier match → first model registered at that tier.
            2. Round UP to the next higher tier that has a model (closer to a
               request we might lack capacity for).
            3. Round DOWN to the closest lower tier that has a model (degrade
               gracefully instead of jumping straight to `default_model`,
               which may be off-tier).
            4. Fall back to `config.default_model`.
            5. Otherwise raise.
        """
        tier_index = self._tier_to_models
        if tier in tier_index and tier_index[tier]:
            return tier_index[tier][0]

        # Round up.
        higher_tiers = sorted(t for t in tier_index if t > tier)
        for t in higher_tiers:
            if tier_index[t]:
                return tier_index[t][0]

        # Round down — closest lower tier first.
        lower_tiers = sorted((t for t in tier_index if t < tier), reverse=True)
        for t in lower_tiers:
            if tier_index[t]:
                return tier_index[t][0]

        if self.config.default_model:
            return self.config.default_model

        raise ValueError(
            f"QualityRouter: no model available for quality tier {tier} and "
            f"no default_model configured"
        )

    def _stash_decision(
        self,
        request_kwargs: Optional[Dict[str, Any]],
        decision: Dict[str, Any],
    ) -> None:
        """
        Stash the routing decision in request_kwargs.metadata so the Router can
        lift it into response headers (`x-litellm-quality-router-*`). The same
        dict object flows from here through to `make_call.set_response_headers`.
        """
        if request_kwargs is None:
            return
        metadata = request_kwargs.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["quality_router_decision"] = decision

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """Try keyword override first; fall back to complexity-tier routing."""
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

        if user_message is None:
            verbose_router_logger.debug(
                "QualityRouter: No user message found, routing to default model"
            )
            if not self.config.default_model:
                raise ValueError(
                    "QualityRouter: no user message and no default_model configured"
                )
            return PreRoutingHookResponse(
                model=self.config.default_model,
                messages=messages,
            )

        # Try keyword override first — it short-circuits complexity classification.
        keyword_match = self._keyword_override(user_message)
        if keyword_match is not None:
            routed_model, matched_keyword = keyword_match
            verbose_router_logger.info(
                f"QualityRouter: keyword override matched='{matched_keyword}' "
                f"routed_model={routed_model} "
                f"(quality_tier={self._model_quality.get(routed_model)}, "
                f"input_cost_per_token={self._model_cost.get(routed_model)})"
            )
            self._stash_decision(
                request_kwargs,
                {
                    "router_model_name": self.model_name,
                    "routed_model": routed_model,
                    "routed_via": "keyword",
                    "matched_keyword": matched_keyword,
                    "quality_tier": self._model_quality.get(routed_model),
                    "complexity_tier": None,
                },
            )
            return PreRoutingHookResponse(
                model=routed_model,
                messages=messages,
            )

        # No keyword match → complexity classification flow.
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

        routed_model = self._resolve_model_for_quality_tier(int(quality_tier))

        verbose_router_logger.info(
            f"QualityRouter: complexity={complexity_name}, score={score:.3f}, "
            f"signals={signals}, quality_tier={quality_tier}, "
            f"routed_model={routed_model}"
        )

        self._stash_decision(
            request_kwargs,
            {
                "router_model_name": self.model_name,
                "routed_model": routed_model,
                "routed_via": "quality_tier",
                "matched_keyword": None,
                "quality_tier": int(quality_tier),
                "complexity_tier": complexity_name,
            },
        )

        return PreRoutingHookResponse(
            model=routed_model,
            messages=messages,
        )
