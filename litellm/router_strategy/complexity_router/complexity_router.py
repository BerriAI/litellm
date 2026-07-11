"""
Complexity-based Auto Router

A rule-based routing strategy that uses weighted scoring across multiple dimensions
to classify requests by complexity and route them to appropriate models.

By default, scoring is local (regex/keyword-based) with no external API calls and <1ms
latency. Optionally, classifier_type="llm" routes classification through a configured
model instead, trading that latency/cost guarantee for potentially better accuracy.

Inspired by ClawRouter: https://github.com/BlockRunAI/ClawRouter
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse

from .config import (
    DEFAULT_CODE_KEYWORDS,
    DEFAULT_REASONING_KEYWORDS,
    DEFAULT_SIMPLE_KEYWORDS,
    DEFAULT_TECHNICAL_KEYWORDS,
    TIER_SEVERITY_ORDER,
    ComplexityRouterConfig,
    ComplexityTier,
)

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


class TierClassification(BaseModel):
    """Structured response schema for the LLM-based complexity classifier."""

    tier: Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]


_CLASSIFICATION_PROMPT_TEMPLATE = """Classify the complexity of the following user request into exactly one tier.

Tiers:
- SIMPLE: factual lookups, greetings, short direct questions with no reasoning or code involved.
- MEDIUM: everyday requests needing some explanation or minor code/technical content.
- COMPLEX: requests involving non-trivial code, architecture, or multi-step technical work.
- REASONING: requests explicitly requiring step-by-step reasoning, analysis, or weighing tradeoffs.

{system_context}Request:
{prompt}"""


def _append_custom_keywords(base_keywords: list[str], custom_keywords: Optional[list[str]]) -> list[str]:
    if not custom_keywords:
        return base_keywords
    base_lowered = frozenset(keyword.lower() for keyword in base_keywords)
    deduped_custom = {keyword.lower(): keyword for keyword in custom_keywords if keyword.lower() not in base_lowered}
    return [*base_keywords, *deduped_custom.values()]


# Metadata keys that carry the parent request's budget reservation. These must not
# reach the classifier's internal acompletion call: the reservation belongs to the
# routed completion that the classifier is deciding on, not to the classifier call
# itself, and forwarding it would let the classifier's cost-tracking reconcile
# against a reservation it isn't responsible for.
_BUDGET_RESERVATION_METADATA_KEYS = frozenset({"user_api_key_budget_reservation", "user_api_key_auth"})


def _classifier_call_metadata(metadata: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not metadata:
        return metadata
    return {k: v for k, v in metadata.items() if k not in _BUDGET_RESERVATION_METADATA_KEYS}


class DimensionScore:
    """Represents a score for a single dimension with optional signal."""

    __slots__ = ("name", "score", "signal")

    def __init__(self, name: str, score: float, signal: Optional[str] = None):
        self.name = name
        self.score = score
        self.signal = signal


class ComplexityRouter(CustomLogger):
    """
    Complexity router that classifies requests and routes to appropriate models.

    By default, handles requests in <1ms with zero external API calls, using weighted
    scoring across multiple dimensions:
    - Token count (short=simple, long=complex)
    - Code presence (code keywords → complex)
    - Reasoning markers ("step by step", "think through" → reasoning tier)
    - Technical terms (domain complexity)
    - Simple indicators ("what is", "define" → simple, negative weight)
    - Multi-step patterns ("first...then", numbered steps)
    - Question complexity (multiple questions)
    """

    def __init__(
        self,
        model_name: str,
        litellm_router_instance: "Router",
        complexity_router_config: Optional[Dict[str, Any]] = None,
        default_model: Optional[str] = None,
    ):
        """
        Initialize ComplexityRouter.

        Args:
            model_name: The name of the model/deployment using this router.
            litellm_router_instance: The LiteLLM Router instance.
            complexity_router_config: Optional configuration dict from proxy config.
            default_model: Optional default model to use if tier cannot be determined.
        """
        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance

        # Parse config - always create a new instance to avoid singleton mutation
        if complexity_router_config:
            self.config = ComplexityRouterConfig(**complexity_router_config)
        else:
            self.config = ComplexityRouterConfig()

        # Override default_model if provided
        if default_model:
            self.config.default_model = default_model

        # Build effective keyword lists (use config overrides or defaults)
        self.code_keywords = self.config.code_keywords or DEFAULT_CODE_KEYWORDS
        self.reasoning_keywords = self.config.reasoning_keywords or DEFAULT_REASONING_KEYWORDS
        self.technical_keywords = _append_custom_keywords(
            self.config.technical_keywords or DEFAULT_TECHNICAL_KEYWORDS,
            self.config.custom_technical_keywords,
        )
        self.simple_keywords = self.config.simple_keywords or DEFAULT_SIMPLE_KEYWORDS

        # Pre-compile regex patterns for efficiency
        # Use non-greedy .*? to prevent ReDoS on pathological inputs
        self._multi_step_patterns = [
            re.compile(r"first.*?then", re.IGNORECASE),
            re.compile(r"step\s*\d", re.IGNORECASE),
            re.compile(r"\d+\.\s"),
            re.compile(r"[a-z]\)\s", re.IGNORECASE),
        ]

        # Optional adaptive soft-floor selector. Built lazily on first use so the
        # parent Router can finish registering underlying deployments first.
        self.adaptive_router: Optional[Any] = None
        self._model_home_tier: Dict[str, ComplexityTier] = {}
        self._adaptive_init_attempted = False

        verbose_router_logger.debug(f"ComplexityRouter initialized for {model_name} with tiers: {self.config.tiers}")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        Uses a simple heuristic: ~4 characters per token on average.
        """
        return len(text) // 4

    def _score_token_count(self, estimated_tokens: int) -> DimensionScore:
        """Score based on token count."""
        thresholds = self.config.token_thresholds
        simple_threshold = thresholds.get("simple", 15)
        complex_threshold = thresholds.get("complex", 400)

        if estimated_tokens < simple_threshold:
            return DimensionScore("tokenCount", -1.0, f"short ({estimated_tokens} tokens)")
        if estimated_tokens > complex_threshold:
            return DimensionScore("tokenCount", 1.0, f"long ({estimated_tokens} tokens)")
        return DimensionScore("tokenCount", 0, None)

    def _keyword_matches(self, text: str, keyword: str) -> bool:
        """
        Check if a keyword matches in text using word boundary matching.

        For single-word keywords, uses regex word boundaries to avoid
        false positives (e.g., "error" matching "terrorism", "class" matching "classical").
        For multi-word phrases, uses substring matching.
        """
        kw_lower = keyword.lower()

        # For single-word keywords, use word boundary matching to avoid false positives
        # e.g., "api" should not match "capital", "error" should not match "terrorism"
        if " " not in kw_lower:
            pattern = r"\b" + re.escape(kw_lower) + r"\b"
            return bool(re.search(pattern, text))

        # For multi-word phrases, substring matching is fine
        return kw_lower in text

    def _score_keyword_match(
        self,
        text: str,
        keywords: List[str],
        name: str,
        signal_label: str,
        thresholds: Tuple[int, int],  # (low, high)
        scores: Tuple[float, float, float],  # (none, low, high)
    ) -> Tuple[DimensionScore, int]:
        """Score based on keyword matches using word boundary matching.

        Returns:
            Tuple of (DimensionScore, match_count) so callers can reuse the count.
        """
        low_threshold, high_threshold = thresholds
        score_none, score_low, score_high = scores

        matches = [kw for kw in keywords if self._keyword_matches(text, kw)]
        match_count = len(matches)

        if match_count >= high_threshold:
            return (
                DimensionScore(name, score_high, f"{signal_label} ({', '.join(matches[:3])})"),
                match_count,
            )
        if match_count >= low_threshold:
            return (
                DimensionScore(name, score_low, f"{signal_label} ({', '.join(matches[:3])})"),
                match_count,
            )
        return DimensionScore(name, score_none, None), match_count

    def _score_multi_step(self, text: str) -> DimensionScore:
        """Score based on multi-step patterns."""
        hits = sum(1 for p in self._multi_step_patterns if p.search(text))
        if hits > 0:
            return DimensionScore("multiStepPatterns", 0.5, "multi-step")
        return DimensionScore("multiStepPatterns", 0, None)

    def _score_question_complexity(self, text: str) -> DimensionScore:
        """Score based on number of question marks."""
        count = text.count("?")
        if count > 3:
            return DimensionScore("questionComplexity", 0.5, f"{count} questions")
        return DimensionScore("questionComplexity", 0, None)

    def classify(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[ComplexityTier, float, List[str]]:
        """
        Classify a prompt by complexity.

        Args:
            prompt: The user's prompt/message.
            system_prompt: Optional system prompt for context.

        Returns:
            Tuple of (tier, score, signals) where:
            - tier: The ComplexityTier (SIMPLE, MEDIUM, COMPLEX, REASONING)
            - score: The raw weighted score
            - signals: List of triggered signals for debugging
        """
        # Combine text for analysis.
        # System prompt is intentionally included in code/technical/simple scoring
        # because it provides deployment-level context (e.g., "You are a Python assistant"
        # signals that code-capable models are appropriate). Reasoning markers use
        # user_text only to prevent system prompts from forcing REASONING tier.
        full_text = f"{system_prompt or ''} {prompt}".lower()
        user_text = prompt.lower()

        # Estimate tokens
        estimated_tokens = self._estimate_tokens(prompt)

        # Score all dimensions, capturing match counts where needed
        code_score, _ = self._score_keyword_match(
            full_text,
            self.code_keywords,
            "codePresence",
            "code",
            (1, 2),
            (0, 0.5, 1.0),
        )
        reasoning_score, reasoning_match_count = self._score_keyword_match(
            user_text,
            self.reasoning_keywords,
            "reasoningMarkers",
            "reasoning",
            (1, 2),
            (0, 0.7, 1.0),
        )
        technical_score, _ = self._score_keyword_match(
            full_text,
            self.technical_keywords,
            "technicalTerms",
            "technical",
            (2, 4),
            (0, 0.5, 1.0),
        )
        simple_score, _ = self._score_keyword_match(
            full_text,
            self.simple_keywords,
            "simpleIndicators",
            "simple",
            (1, 2),
            (0, -1.0, -1.0),
        )

        dimensions: List[DimensionScore] = [
            self._score_token_count(estimated_tokens),
            code_score,
            reasoning_score,
            technical_score,
            simple_score,
            self._score_multi_step(full_text),
            self._score_question_complexity(prompt),
        ]

        # Collect signals
        signals = [d.signal for d in dimensions if d.signal is not None]

        # Compute weighted score
        weights = self.config.dimension_weights
        weighted_score = sum(d.score * weights.get(d.name, 0) for d in dimensions)

        # Check for reasoning override (2+ reasoning markers)
        # Reuse match count from _score_keyword_match to avoid scanning twice
        if reasoning_match_count >= 2:
            return ComplexityTier.REASONING, weighted_score, signals

        # Map score to tier
        boundaries = self.config.tier_boundaries
        simple_medium = boundaries.get("simple_medium", 0.15)
        medium_complex = boundaries.get("medium_complex", 0.35)
        complex_reasoning = boundaries.get("complex_reasoning", 0.60)

        if weighted_score < simple_medium:
            tier = ComplexityTier.SIMPLE
        elif weighted_score < medium_complex:
            tier = ComplexityTier.MEDIUM
        elif weighted_score < complex_reasoning:
            tier = ComplexityTier.COMPLEX
        else:
            tier = ComplexityTier.REASONING

        return tier, weighted_score, signals

    async def aclassify(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[ComplexityTier, float, list[str]]:
        """
        Classify a prompt by complexity, using the LLM classifier when configured.

        Falls back to the local heuristic scorer if classifier_type is "heuristic",
        or if the LLM call fails, times out, or returns an unparseable response.
        """
        if self.config.classifier_type != "llm" or self.config.classifier_llm_config is None:
            return self.classify(prompt, system_prompt)

        try:
            tier = await self._classify_with_llm(prompt, system_prompt, request_kwargs)
            return tier, 1.0, [f"llm-classifier:{tier.value}"]
        except Exception as e:  # noqa: BLE001 -- external LLM call can fail in many distinct ways (timeout, provider error, validation, parse error); any failure must fall back to the heuristic scorer
            verbose_router_logger.warning(
                f"ComplexityRouter: LLM classifier failed ({e}), falling back to heuristic scoring"
            )
            return self.classify(prompt, system_prompt)

    async def _classify_with_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_kwargs: Optional[dict[str, Any]] = None,
    ) -> ComplexityTier:
        """Call the configured classifier model and parse its structured tier response."""
        llm_config = self.config.classifier_llm_config
        if llm_config is None:
            raise ValueError("classifier_llm_config is not set")

        system_context = f"Context: {system_prompt}\n\n" if system_prompt else ""
        classification_prompt = _CLASSIFICATION_PROMPT_TEMPLATE.format(system_context=system_context, prompt=prompt)

        # Forward the original request's metadata so the classifier call's spend is
        # attributed to the calling key/team instead of being dropped. Excludes the
        # parent request's budget reservation, which the routed completion (not this
        # internal classifier call) is responsible for reconciling.
        metadata = _classifier_call_metadata((request_kwargs or {}).get("litellm_metadata"))

        response: ModelResponse = await self.litellm_router_instance.acompletion(
            model=llm_config.model,
            messages=[{"role": "user", "content": classification_prompt}],
            response_format=TierClassification,
            timeout=llm_config.timeout_ms / 1000,
            metadata=metadata,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM classifier returned empty content")
        result = TierClassification.model_validate_json(content)
        return ComplexityTier[result.tier]

    def get_model_for_tier(self, tier: ComplexityTier) -> str:
        """
        Get the model name for a given complexity tier.

        Args:
            tier: The complexity tier.

        Returns:
            The model name configured for that tier.
        """
        tier_key = tier.value if isinstance(tier, ComplexityTier) else tier

        # Check config tiers mapping
        model = self.config.tiers.get(tier_key)
        if model:
            if isinstance(model, list):
                if not model:
                    raise ValueError(f"Empty model pool for tier {tier_key}")
                return model[0]
            return model

        # Fallback to default model if configured
        if self.config.default_model:
            return self.config.default_model

        # Last resort: return MEDIUM tier model or error
        medium_model = self.config.tiers.get(ComplexityTier.MEDIUM.value)
        if medium_model:
            if isinstance(medium_model, list):
                if not medium_model:
                    raise ValueError("Empty model pool for MEDIUM tier")
                return medium_model[0]
            return medium_model

        raise ValueError(f"No model configured for tier {tier_key} and no default_model set")

    def _tier_pools(self) -> Dict[str, List[str]]:
        return {tier: (models if isinstance(models, list) else [models]) for tier, models in self.config.tiers.items()}

    def _ensure_adaptive_router(self) -> Optional[Any]:
        """Lazily construct the embedded AdaptiveRouter used for soft-floor picks."""
        if not self.config.adaptive:
            return None
        if self.adaptive_router is not None:
            return self.adaptive_router
        if self._adaptive_init_attempted:
            return self.adaptive_router
        self._adaptive_init_attempted = True

        from litellm.router_strategy.adaptive_router.adaptive_router import (
            AdaptiveRouter,
        )
        from litellm.router_strategy.adaptive_router.config import (
            ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY,
        )
        from litellm.types.router import (
            AdaptiveRouterConfig,
            AdaptiveRouterPreferences,
        )

        pools = self._tier_pools()
        available_models = list(dict.fromkeys(model for models in pools.values() for model in models))
        home_tier: Dict[str, ComplexityTier] = {}
        for tier_name, models in pools.items():
            tier = ComplexityTier(tier_name)
            for model in models:
                if model not in home_tier:
                    home_tier[model] = tier
        self._model_home_tier = home_tier

        model_to_prefs: Dict[str, AdaptiveRouterPreferences] = {}
        model_to_cost: Dict[str, float] = {}
        model_list = getattr(self.litellm_router_instance, "model_list", None) or []
        name_to_indices = getattr(self.litellm_router_instance, "model_name_to_deployment_indices", {}) or {}
        for name in available_models:
            indices = name_to_indices.get(name, [])
            if not indices:
                model_to_prefs[name] = AdaptiveRouterPreferences(quality_tier=2, strengths=[])
                model_to_cost[name] = 0.0
                continue
            deployment = model_list[indices[0]]
            mi = deployment.get("model_info") if isinstance(deployment, dict) else deployment.model_info
            mi_dict: Dict[str, Any] = mi if isinstance(mi, dict) else (mi.model_dump() if mi else {})
            prefs_raw = mi_dict.get("adaptive_router_preferences")
            if prefs_raw is not None:
                model_to_prefs[name] = AdaptiveRouterPreferences(**prefs_raw)
            else:
                model_to_prefs[name] = AdaptiveRouterPreferences(quality_tier=2, strengths=[])

            lp = deployment.get("litellm_params") if isinstance(deployment, dict) else deployment.litellm_params
            lp_dict: Dict[str, Any] = lp if isinstance(lp, dict) else (lp.model_dump() if lp else {})
            cost = lp_dict.get("input_cost_per_token")
            model_to_cost[name] = float(cost) if cost is not None else 0.0

        self.adaptive_router = AdaptiveRouter(
            router_name=self.model_name,
            config=AdaptiveRouterConfig(
                available_models=available_models,
                weights=self.config.adaptive_weights,
            ),
            model_to_prefs=model_to_prefs,
            model_to_cost=model_to_cost,
        )
        # Stash key helper for pre-routing metadata writes.
        self._adaptive_chosen_model_key = ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY
        return self.adaptive_router

    def _soft_floor_pick(self, classified_tier: ComplexityTier, user_message: str) -> str:
        """Thompson-sample across all tier pools with a tier-distance penalty."""
        from litellm.router_strategy.adaptive_router.bandit import (
            normalized_cost,
            thompson_sample,
        )
        from litellm.router_strategy.adaptive_router.classifier import classify_prompt

        adaptive = self._ensure_adaptive_router()
        if adaptive is None:
            return self.get_model_for_tier(classified_tier)

        request_type = classify_prompt(user_message)
        classified_idx = TIER_SEVERITY_ORDER.index(classified_tier)
        all_costs = [adaptive.model_to_cost.get(m, 0.0) for m in adaptive.config.available_models]
        quality_weight = self.config.adaptive_weights.quality
        cost_weight = self.config.adaptive_weights.cost
        penalty_weight = self.config.tier_distance_penalty

        best_model: Optional[str] = None
        best_score = float("-inf")
        for model in adaptive.config.available_models:
            cell = adaptive._cells[(request_type, model)]
            quality_sample = thompson_sample(cell)
            cost_score = normalized_cost(adaptive.model_to_cost.get(model, 0.0), all_costs)
            home = self._model_home_tier.get(model, classified_tier)
            distance = abs(TIER_SEVERITY_ORDER.index(home) - classified_idx)
            score = quality_weight * quality_sample + cost_weight * cost_score - penalty_weight * distance
            if score > best_score:
                best_score = score
                best_model = model
        if best_model is None:
            return self.get_model_for_tier(classified_tier)
        return best_model

    def _resolve_messages(
        self,
        messages: Optional[List[Dict[str, Any]]],
        request_kwargs: Dict,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Resolve messages from the request, converting from other formats if needed.

        Uses the guardrail translation handler dispatch to convert Responses API
        ``input`` (or other non-chat-completions formats) into OpenAI-spec messages.
        """
        if messages:
            return messages

        from litellm.litellm_core_utils.api_route_to_call_types import (
            get_call_types_for_route,
        )
        from litellm.llms import load_guardrail_translation_mappings
        from litellm.types.utils import CallTypes

        mappings = load_guardrail_translation_mappings()
        call_type: Optional[CallTypes] = None

        # 1. Try route-based inference from proxy metadata
        route = request_kwargs.get("litellm_metadata", {}).get("user_api_key_request_route")
        if route:
            call_types_list = get_call_types_for_route(route)
            if call_types_list:
                for ct in call_types_list:
                    if ct in mappings:
                        call_type = ct
                        break

        # 2. Fallback: try each mapped handler until one produces messages
        handlers_to_try: List[Any] = []
        if call_type is not None and call_type in mappings:
            handlers_to_try.append(mappings[call_type]())
        else:
            handlers_to_try.extend(handler_cls() for handler_cls in mappings.values())

        for handler in handlers_to_try:
            structured = handler.get_structured_messages(request_kwargs)
            if structured:
                return [
                    msg if isinstance(msg, dict) else msg.model_dump()  # type: ignore
                    for msg in structured
                ]
        return None

    @staticmethod
    def _extract_user_message_and_system_prompt(
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract the last user message text and last system prompt from messages."""
        user_message: Optional[str] = None
        system_prompt: Optional[str] = None

        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = " ".join(text_parts).strip()
            if isinstance(content, str) and content:
                if role == "user" and user_message is None:
                    user_message = content
                elif role == "system" and system_prompt is None:
                    system_prompt = content
            if user_message is not None and system_prompt is not None:
                break

        return user_message, system_prompt

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        Pre-routing hook called before the routing decision.

        Classifies the request by complexity and returns the appropriate model.
        Supports chat completions (messages), Responses API (input), and other
        formats via the guardrail translation handler dispatch.

        Args:
            model: The original model name requested.
            request_kwargs: The request kwargs.
            messages: The messages in the request.
            input: Optional input for Responses API or embeddings.
            specific_deployment: Whether a specific deployment was requested.

        Returns:
            PreRoutingHookResponse with the routed model, or None if no routing needed.
        """
        from litellm.types.router import PreRoutingHookResponse

        resolved_messages = self._resolve_messages(messages, request_kwargs)

        if not resolved_messages:
            verbose_router_logger.debug("ComplexityRouter: No messages could be resolved, skipping routing")
            return None

        # Determine whether the original request used messages directly
        has_original_messages = messages is not None and len(messages) > 0

        user_message, system_prompt = self._extract_user_message_and_system_prompt(resolved_messages)

        if user_message is None:
            verbose_router_logger.debug("ComplexityRouter: No user message found, routing to default model")
            return PreRoutingHookResponse(
                model=self.config.default_model or self.get_model_for_tier(ComplexityTier.MEDIUM),
                messages=messages if has_original_messages else None,
            )

        tier, score, signals = await self.aclassify(user_message, system_prompt, request_kwargs)
        if self.config.adaptive:
            routed_model = self._soft_floor_pick(tier, user_message)
            adaptive = self._ensure_adaptive_router()
            if adaptive is not None:
                kwargs_metadata = request_kwargs.setdefault("metadata", {})
                if isinstance(kwargs_metadata, dict):
                    chosen_key = getattr(self, "_adaptive_chosen_model_key", "adaptive_router_chosen_model")
                    kwargs_metadata[chosen_key] = routed_model
            verbose_router_logger.info(
                f"ComplexityRouter[adaptive]: tier={tier.value}, score={score:.3f}, "
                f"signals={signals}, routed_model={routed_model}"
            )
        else:
            routed_model = self.get_model_for_tier(tier)
            verbose_router_logger.info(
                f"ComplexityRouter: tier={tier.value}, score={score:.3f}, signals={signals}, routed_model={routed_model}"
            )

        return PreRoutingHookResponse(
            model=routed_model,
            messages=messages if has_original_messages else None,
        )
