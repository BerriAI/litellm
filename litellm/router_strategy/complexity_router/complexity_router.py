"""
Complexity-based Auto Router

A rule-based routing strategy that uses weighted scoring across multiple dimensions
to classify requests by complexity and route them to appropriate models.

By default, scoring is local (regex/keyword-based) with no external API calls and <1ms
latency. Optionally, classifier_type="llm" routes classification through a configured
model instead, trading that latency/cost guarantee for potentially better accuracy.
keyword_tier_rules (lexical or, with semantic_keyword_matching, embedding-based) are
evaluated before either classification strategy and force a tier outright when matched.

Inspired by ClawRouter: https://github.com/BlockRunAI/ClawRouter
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Union, cast

from pydantic import BaseModel

from litellm._logging import verbose_router_logger
from litellm.constants import RETURN_RAW_MODEL_NAME_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import get_request_metadata_field
from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.types.utils import (
    ModelResponse,
    RoutingDecisionCause,
    StandardLoggingRoutingDecision,
    StandardLoggingRoutingDecisionTierBoundaries,
)

from .config import (
    DEFAULT_CODE_KEYWORDS,
    DEFAULT_ESCALATION_KEYWORDS,
    DEFAULT_REASONING_KEYWORDS,
    DEFAULT_SIMPLE_KEYWORDS,
    DEFAULT_TECHNICAL_KEYWORDS,
    TIER_SEVERITY_ORDER,
    ComplexityRouterConfig,
    ComplexityTier,
)

if TYPE_CHECKING:
    from semantic_router.routers import SemanticRouter

    from litellm.router import Router
    from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
    from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any
    SemanticRouter = Any


class TierClassification(BaseModel):
    """Structured response schema for the LLM-based complexity classifier."""

    tier: Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]


_CLASSIFICATION_PROMPT_TEMPLATE = """Classify the complexity of the following user request into exactly one tier.

Judge the intellectual difficulty of answering correctly, not how short the request is.

Tiers:
- SIMPLE: greetings, chitchat, or factual lookups with a short known answer. Do not use SIMPLE for unsolved problems, proofs, deep theory, multi-step analysis, or non-trivial code, even if the request is only one sentence.
- MEDIUM: everyday requests that need some explanation, light reasoning, or minor code/technical content.
- COMPLEX: non-trivial code, architecture, multi-step technical work, or specialized domain depth.
- REASONING: open-ended analysis, proofs, famous hard problems, step-by-step reasoning, tradeoffs, or anything where a correct answer requires careful thought rather than a quick lookup.

{system_context}Request:
{prompt}"""


def _append_custom_keywords(base_keywords: list[str], custom_keywords: list[str] | None) -> list[str]:
    if not custom_keywords:
        return base_keywords
    base_lowered = frozenset(keyword.lower() for keyword in base_keywords)
    deduped_custom = {keyword.lower(): keyword for keyword in custom_keywords if keyword.lower() not in base_lowered}
    return [*base_keywords, *deduped_custom.values()]


# Metadata keys that carry only the parent request's budget reservation state. These
# must not reach internal sub-calls (classifier, embedding): the reservation belongs to
# the routed completion being decided on, not to the sub-call itself, and forwarding it
# would let the sub-call's cost callback finalize the reservation, causing the routed
# completion's callback to skip incrementing key/team budget counters.
#
# Note: user_api_key_auth itself is intentionally kept; it is required by
# _filter_deployments_by_model_access_groups to scope embedding/classifier model
# selection to the caller's authorized access groups. It is forwarded as a sanitized
# copy with its budget_reservation sub-field removed, because the proxy cost callback
# (_get_budget_reservation_from_metadata) falls back to reading the reservation from
# inside the auth object when the top-level key is absent; forwarding it unsanitized
# would re-create the exact double-finalization this stripping exists to prevent.
_BUDGET_RESERVATION_METADATA_KEYS = frozenset({"user_api_key_budget_reservation"})


def _sanitize_user_api_key_auth(auth: Any) -> Any:
    if isinstance(auth, dict):
        return {k: v for k, v in auth.items() if k != "budget_reservation"}
    if getattr(auth, "budget_reservation", None) is not None and hasattr(auth, "model_copy"):
        return auth.model_copy(update={"budget_reservation": None})
    return auth


def _classifier_call_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        k: _sanitize_user_api_key_auth(v) if k == "user_api_key_auth" else v
        for k, v in metadata.items()
        if k not in _BUDGET_RESERVATION_METADATA_KEYS
    }


def _effective_turn_off_message_logging(request_kwargs: Mapping[str, Any] | None) -> bool | None:
    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        initialize_standard_callback_dynamic_params,
    )

    return initialize_standard_callback_dynamic_params(dict(request_kwargs) if request_kwargs else {}).get(
        "turn_off_message_logging"
    )


class DimensionScore:
    """Represents a score for a single dimension with optional signal."""

    __slots__ = ("name", "score", "signal")

    def __init__(self, name: str, score: float, signal: str | None = None):
        self.name = name
        self.score = score
        self.signal = signal


class KeywordOverride(NamedTuple):
    """A keyword_tier_rules match: the winning tier and, on the lexical path, the keyword that fired."""

    tier: ComplexityTier
    matched_keyword: str | None


class ClassificationOutcome(NamedTuple):
    """What the classifier decided and which mechanism actually produced it.

    `cause` reflects the path that ran, not the configured classifier_type: an LLM
    classifier that fails falls back to the heuristic scorer and reports it.
    `score` is None on the LLM path, which produces a tier label and no score.
    """

    tier: ComplexityTier
    score: float | None
    signals: tuple[str, ...]
    cause: Literal["heuristic_scorer", "reasoning_override", "llm_classifier"]


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
        litellm_router_instance: Router,
        complexity_router_config: dict[str, Any] | None = None,
        default_model: str | None = None,
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
        self.escalation_keywords = (
            self.config.escalation_keywords
            if self.config.escalation_keywords is not None
            else DEFAULT_ESCALATION_KEYWORDS
        )

        # Lazily built on first semantic request and cached for reuse (route
        # embeddings are static, only the prompt is embedded per request). The lock
        # serializes the one-time build so concurrent cold-start requests don't each
        # construct the index and fire duplicate embedding calls.
        self._semantic_routelayer: SemanticRouter | None = None
        self._semantic_routelayer_lock = asyncio.Lock()

        # Pre-compile regex patterns for efficiency
        # Use non-greedy .*? to prevent ReDoS on pathological inputs
        self._multi_step_patterns = [
            re.compile(r"first.*?then", re.IGNORECASE),
            re.compile(r"step\s*\d", re.IGNORECASE),
            re.compile(r"\d+\.\s"),
            re.compile(r"[a-z]\)\s", re.IGNORECASE),
        ]

        self.adaptive_router: AdaptiveRouter | None = None
        self._model_tiers: dict[str, tuple[ComplexityTier, ...]] = {}
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
        disclosable_text: str,
        keywords: list[str],
        name: str,
        signal_label: str,
        thresholds: tuple[int, int],  # (low, high)
        scores: tuple[float, float, float],  # (none, low, high)
    ) -> tuple[DimensionScore, int]:
        """Score based on keyword matches using word boundary matching.

        Scoring reads `text`, which for most dimensions includes the system prompt.
        The signal names only the terms that also appear in `disclosable_text`, the
        caller's own message: signals are persisted to the request's spend log, which
        the caller can read, so naming a term matched solely in the system prompt would
        let a caller recover configured terms from a prompt it cannot see. Terms it did
        not supply are reported as a count instead, which explains the score without
        disclosing anything. `disclosable_text` is required rather than defaulted so a
        future dimension has to state which text it is willing to quote.

        Returns:
            Tuple of (DimensionScore, match_count) so callers can reuse the count.
        """
        low_threshold, high_threshold = thresholds
        score_none, score_low, score_high = scores

        matches = [kw for kw in keywords if self._keyword_matches(text, kw)]
        match_count = len(matches)
        if match_count < low_threshold:
            return DimensionScore(name, score_none, None), match_count

        disclosable = [kw for kw in matches if self._keyword_matches(disclosable_text, kw)]
        detail = ", ".join(disclosable[:3]) if disclosable else f"{match_count} matches"
        score = score_high if match_count >= high_threshold else score_low
        return DimensionScore(name, score, f"{signal_label} ({detail})"), match_count

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

    def classify(self, prompt: str, system_prompt: str | None = None) -> tuple[ComplexityTier, float, list[str]]:
        """Classify a prompt by complexity, discarding which rule decided the tier.

        Kept for callers that only need the tier and score; `_score_and_classify` is the
        single computation behind both, so the two can never disagree.
        """
        tier, score, signals, _cause = self._score_and_classify(prompt, system_prompt)
        return tier, score, list(signals)

    def _score_and_classify(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[ComplexityTier, float, tuple[str, ...], Literal["heuristic_scorer", "reasoning_override"]]:
        """
        Classify a prompt by complexity, reporting whether the score chose the tier.

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
            user_text,
            self.code_keywords,
            "codePresence",
            "code",
            (1, 2),
            (0, 0.5, 1.0),
        )
        reasoning_score, reasoning_match_count = self._score_keyword_match(
            user_text,
            user_text,
            self.reasoning_keywords,
            "reasoningMarkers",
            "reasoning",
            (1, 2),
            (0, 0.7, 1.0),
        )
        technical_score, _ = self._score_keyword_match(
            full_text,
            user_text,
            self.technical_keywords,
            "technicalTerms",
            "technical",
            (2, 4),
            (0, 0.5, 1.0),
        )
        simple_score, _ = self._score_keyword_match(
            full_text,
            user_text,
            self.simple_keywords,
            "simpleIndicators",
            "simple",
            (1, 2),
            (0, -1.0, -1.0),
        )

        dimensions: list[DimensionScore] = [
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
            return ComplexityTier.REASONING, weighted_score, tuple(signals), "reasoning_override"

        # Map score to tier
        boundaries = self._effective_tier_boundaries()
        if weighted_score < boundaries["simple_medium"]:
            tier = ComplexityTier.SIMPLE
        elif weighted_score < boundaries["medium_complex"]:
            tier = ComplexityTier.MEDIUM
        elif weighted_score < boundaries["complex_reasoning"]:
            tier = ComplexityTier.COMPLEX
        else:
            tier = ComplexityTier.REASONING

        return tier, weighted_score, tuple(signals), "heuristic_scorer"

    def _effective_tier_boundaries(self) -> StandardLoggingRoutingDecisionTierBoundaries:
        """The tier boundaries in effect, with the documented defaults filled in.

        Shared by score-to-tier mapping and the per-request routing decision snapshot,
        so a logged decision always reflects the boundaries that actually applied.
        """
        boundaries = self.config.tier_boundaries
        return StandardLoggingRoutingDecisionTierBoundaries(
            simple_medium=boundaries.get("simple_medium", 0.15),
            medium_complex=boundaries.get("medium_complex", 0.35),
            complex_reasoning=boundaries.get("complex_reasoning", 0.60),
        )

    def _build_routing_decision(
        self,
        *,
        routed_model: str,
        cause: RoutingDecisionCause,
        tier: ComplexityTier | None = None,
        score: float | None = None,
        signals: tuple[str, ...] | None = None,
        matched_keyword: str | None = None,
        escalation_keyword: str | None = None,
        escalated: bool = False,
        classifier_model: str | None = None,
    ) -> StandardLoggingRoutingDecision:
        """Assemble the per-request provenance record for this router's decision.

        Optional facts are omitted rather than set to None, so a spend log row only
        carries the keys that applied to its path. `tier_boundaries` rides with
        `score` because the score is only interpretable against the boundaries that
        mapped it to a tier.
        """
        decision = StandardLoggingRoutingDecision(
            router_model_name=self.model_name,
            router_type="complexity",
            routed_model=routed_model,
            cause=cause,
        )
        if tier is not None:
            decision["tier"] = tier.value
        if score is not None:
            decision["score"] = score
            decision["tier_boundaries"] = self._effective_tier_boundaries()
        if signals:
            # Stored as a list because this record is serialized to JSON for the spend
            # log and read back as an array by the dashboard; a sequence type that only
            # happens to survive the serializer would make the wire shape depend on it.
            decision["signals"] = list(signals)
        if matched_keyword is not None:
            decision["matched_keyword"] = matched_keyword
        if escalation_keyword is not None:
            # Two separate facts: the caller asked to escalate, and whether the tier
            # actually moved. A request that escalates from an already-highest tier has
            # nowhere to go, so it records the keyword with escalated=False rather than
            # dropping the ask (which reads as an ordinary route) or claiming a bump
            # that never happened. Every path reports both the same way.
            decision["escalation_keyword"] = escalation_keyword
            decision["escalated"] = escalated
        if classifier_model is not None:
            decision["classifier_model"] = classifier_model
        return decision

    async def aclassify(
        self,
        prompt: str,
        system_prompt: str | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> ClassificationOutcome:
        """
        Classify a prompt by complexity, using the LLM classifier when configured.

        Falls back to the local heuristic scorer if classifier_type is "heuristic",
        or if the LLM call fails, times out, or returns an unparseable response.
        The outcome's `cause` reports which path actually classified the request.
        """
        if self.config.classifier_type != "llm" or self.config.classifier_llm_config is None:
            tier, score, signals, cause = self._score_and_classify(prompt, system_prompt)
            return ClassificationOutcome(tier=tier, score=score, signals=signals, cause=cause)

        try:
            tier = await self._classify_with_llm(prompt, system_prompt, request_kwargs)
            return ClassificationOutcome(
                tier=tier, score=None, signals=(f"llm-classifier:{tier.value}",), cause="llm_classifier"
            )
        except Exception as e:  # noqa: BLE001 -- external LLM call can fail in many distinct ways (timeout, provider error, validation, parse error); any failure must fall back to the heuristic scorer
            verbose_router_logger.warning(
                f"ComplexityRouter: LLM classifier failed ({e}), falling back to heuristic scoring"
            )
            tier, score, signals, cause = self._score_and_classify(prompt, system_prompt)
            return ClassificationOutcome(tier=tier, score=score, signals=signals, cause=cause)

    async def _classify_with_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        request_kwargs: dict[str, Any] | None = None,
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
        request_metadata = (request_kwargs or {}).get("litellm_metadata") or (request_kwargs or {}).get("metadata")
        metadata = _classifier_call_metadata(request_metadata)
        turn_off_message_logging = _effective_turn_off_message_logging(request_kwargs)

        proxy_server_request = {
            "body": {
                "model": llm_config.model,
                "messages": [{"role": "user", "content": classification_prompt}],
                "response_format": type_to_response_format_param(TierClassification),
            }
        }

        response: ModelResponse = await self.litellm_router_instance.acompletion(
            model=llm_config.model,
            messages=[{"role": "user", "content": classification_prompt}],
            response_format=TierClassification,
            timeout=llm_config.timeout_ms / 1000,
            metadata=metadata,
            proxy_server_request=proxy_server_request,
            turn_off_message_logging=turn_off_message_logging,
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

        if tier_key in self.config.tiers:
            return self._pick_from_tier_value(self.config.tiers[tier_key], tier_key)

        if self.config.default_model:
            return self.config.default_model

        medium_key = ComplexityTier.MEDIUM.value
        if medium_key in self.config.tiers:
            return self._pick_from_tier_value(self.config.tiers[medium_key], medium_key)

        raise ValueError(f"No model configured for tier {tier_key} and no default_model set")

    @staticmethod
    def _pick_from_tier_value(model: str | list[str], tier_key: str) -> str:
        if isinstance(model, str):
            return model
        if not model:
            raise ValueError(f"Empty model pool for tier {tier_key}")
        return random.choice(model)

    def _tier_pools(self) -> dict[str, list[str]]:
        return {tier: (models if isinstance(models, list) else [models]) for tier, models in self.config.tiers.items()}

    async def _pick_model_for_tier(
        self,
        tier: ComplexityTier,
        raw_messages: list[dict[str, Any]] | None,
        resolved_messages: list[dict[str, Any]] | None,
        request_kwargs: dict,
    ) -> str:
        if not self.config.plugins:
            warm_pick = await self._warm_aware_pick(self._tier_pools().get(tier.value, []), request_kwargs)
            if warm_pick is not None:
                return warm_pick
            return self.get_model_for_tier(tier)

        from litellm.types.router import RoutingContext

        tier_key = tier.value
        metadata_key = "litellm_metadata" if "litellm_metadata" in request_kwargs else "metadata"
        context = RoutingContext(
            raw_messages=raw_messages or [],
            structured_messages=resolved_messages or [],
            candidate_models=list(self._tier_pools().get(tier_key, [])),
            metadata=request_kwargs.get(metadata_key) or {},
        )
        for plugin in self.config.plugins:
            context = await plugin.run(context)

        if not context.candidate_models:
            # A plugin narrowing a tier to zero candidates is a policy decision (e.g. no
            # model this tenant's budget allows) -- falling back to default_model here
            # (which was never checked against the plugins) would let that policy be
            # silently bypassed. Raise instead, matching the Router-level plugin
            # pipeline's own fail-closed behavior for the same situation.
            raise ValueError(f"No candidate models left for tier {tier_key} after routing-plugin filtering")
        warm_pick = await self._warm_aware_pick(context.candidate_models, request_kwargs)
        if warm_pick is not None:
            return warm_pick
        return self._pick_from_tier_value(context.candidate_models, tier_key)

    async def _warm_aware_pick(self, pool: Sequence[str], request_kwargs: Mapping[str, object]) -> str | None:
        config = self.config.cache_warming
        if not config.enabled or len(pool) <= 1:
            return None
        session_id = get_request_metadata_field(request_kwargs, "session_id")
        if session_id is None:
            return None
        store = self.get_cache_warming_store()
        if store is None or store.redis_cache is None:
            return None
        from litellm.router_strategy.complexity_router.cache_warming.types import is_cache_fresh

        caller_scope = get_request_metadata_field(request_kwargs, "user_api_key_hash") or "unscoped"
        record_key = store.record_key(self.model_name, caller_scope, session_id)
        record = await store.get_record(record_key)
        if record is None:
            return None
        warmth = await store.get_warmth(record_key, tuple(pool))
        now = time.time()
        warmed = frozenset(model for model, warmed_at in warmth.items() if is_cache_fresh(warmed_at, now))
        served = frozenset((record.served_model,)) if is_cache_fresh(record.last_activity, now) else frozenset[str]()
        candidates = tuple(model for model in pool if model in warmed | served)
        if not candidates:
            return None
        return random.choice(candidates)

    def _ensure_adaptive_router(self) -> Any | None:
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
        self._model_tiers = {
            model: tuple(ComplexityTier(tier_name) for tier_name, models in pools.items() if model in models)
            for model in available_models
        }

        model_to_prefs: dict[str, AdaptiveRouterPreferences] = {}
        model_to_cost: dict[str, float] = {}
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
            mi_dict: dict[str, Any] = mi if isinstance(mi, dict) else (mi.model_dump() if mi else {})
            prefs_raw = mi_dict.get("adaptive_router_preferences")
            if prefs_raw is not None:
                model_to_prefs[name] = AdaptiveRouterPreferences(**prefs_raw)
            else:
                model_to_prefs[name] = AdaptiveRouterPreferences(quality_tier=2, strengths=[])

            lp = deployment.get("litellm_params") if isinstance(deployment, dict) else deployment.litellm_params
            lp_dict: dict[str, Any] = lp if isinstance(lp, dict) else (lp.model_dump() if lp else {})
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
        self._adaptive_chosen_model_key = ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY
        return self.adaptive_router

    def _soft_floor_pick(
        self,
        classified_tier: ComplexityTier,
        user_message: str,
        request_kwargs: dict[str, Any] | None = None,
    ) -> str:
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
        pools = self._tier_pools()
        classified_candidates = tuple(pools.get(classified_tier.value, ()))
        cold_start_candidates = tuple(
            model for model in classified_candidates if adaptive._cells[(request_type, model)].total_samples == 0
        )
        if cold_start_candidates:
            chosen_model = random.choice(cold_start_candidates)
            if request_kwargs is not None:
                metadata = request_kwargs.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["adaptive_router_decision"] = {
                        "phase": "cold_start",
                        "classified_tier": classified_tier.value,
                        "request_type": request_type.value,
                        "eligible_mode": "classified_tier",
                        "quality_weight": self.config.adaptive_weights.quality,
                        "cost_weight": self.config.adaptive_weights.cost,
                        "tier_distance_penalty": self.config.tier_distance_penalty,
                        "chosen_model": chosen_model,
                        "candidates": [
                            {
                                "model": model,
                                "total_samples": adaptive._cells[(request_type, model)].total_samples,
                            }
                            for model in cold_start_candidates
                        ],
                    }
            return chosen_model
        if self.config.adaptive_eligible == "classified_tier":
            candidates = list(classified_candidates)
            if not candidates:
                return self.get_model_for_tier(classified_tier)
        else:
            candidates = list(adaptive.config.available_models)

        all_costs = [adaptive.model_to_cost.get(m, 0.0) for m in candidates]
        quality_weight = self.config.adaptive_weights.quality
        cost_weight = self.config.adaptive_weights.cost
        penalty_weight = self.config.tier_distance_penalty

        best_model: str | None = None
        best_score = float("-inf")
        candidate_scores: list[dict[str, Any]] = []
        for model in candidates:
            cell = adaptive._cells[(request_type, model)]
            quality_sample = thompson_sample(cell)
            cost_score = normalized_cost(adaptive.model_to_cost.get(model, 0.0), all_costs)
            if self.config.adaptive_eligible == "classified_tier":
                distance = 0
            else:
                model_tiers = self._model_tiers.get(model, (classified_tier,))
                distance = min(
                    abs(TIER_SEVERITY_ORDER.index(model_tier) - classified_idx) for model_tier in model_tiers
                )
            score = quality_weight * quality_sample + cost_weight * cost_score - penalty_weight * distance
            candidate_scores.append(
                {
                    "model": model,
                    "quality_sample": quality_sample,
                    "cost_score": cost_score,
                    "tier_distance": distance,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_model = model
        if best_model is None:
            return self.get_model_for_tier(classified_tier)
        if request_kwargs is not None:
            metadata = request_kwargs.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["adaptive_router_decision"] = {
                    "phase": "adaptive",
                    "classified_tier": classified_tier.value,
                    "request_type": request_type.value,
                    "eligible_mode": self.config.adaptive_eligible,
                    "quality_weight": quality_weight,
                    "cost_weight": cost_weight,
                    "tier_distance_penalty": penalty_weight,
                    "chosen_model": best_model,
                    "candidates": candidate_scores,
                }
        return best_model

    def _matched_escalation_keyword(self, user_message: str) -> str | None:
        """The escalation keyword the prompt contains, or None when escalation is off.

        Matching is a case-sensitive substring test so the default "LITELLM ESCALATE"
        only fires on the deliberate, shouted form and not on incidental lowercase
        mentions of the word (e.g. "how do I escalate this ticket").
        """
        if not self.escalation_keywords:
            return None
        return next((keyword for keyword in self.escalation_keywords if keyword in user_message), None)

    def _tier_for_model(self, model: str) -> ComplexityTier | None:
        """Return the most-severe configured tier whose pool contains this model."""
        pools = self._tier_pools()
        matched = tuple(ComplexityTier(tier_name) for tier_name, models in pools.items() if model in models)
        if not matched:
            return None
        return max(matched, key=TIER_SEVERITY_ORDER.index)

    def _escalate_tier(self, tier: ComplexityTier) -> ComplexityTier:
        """Bump a tier one step up to the next-higher configured tier.

        Returns the input tier unchanged when it is already the highest configured
        tier, so escalation can never route below the model the user would otherwise
        have received.
        """
        configured = frozenset(self.config.tiers)
        current_index = TIER_SEVERITY_ORDER.index(tier)
        higher_tiers = tuple(
            candidate for candidate in TIER_SEVERITY_ORDER[current_index + 1 :] if candidate.value in configured
        )
        return higher_tiers[0] if higher_tiers else tier

    def _escalated_pin(self, pinned_model: str) -> str | None:
        """Bump a session's pinned model to the next-higher configured tier.

        Returns None when the pin no longer maps to any configured tier, signalling
        a full reclassification instead.
        """
        pinned_tier = self._tier_for_model(pinned_model)
        if pinned_tier is None:
            return None
        escalated_tier = self._escalate_tier(pinned_tier)
        if escalated_tier == pinned_tier:
            return pinned_model
        return self.get_model_for_tier(escalated_tier)

    def _lexical_tier_override(self, user_message: str) -> KeywordOverride | None:
        """When keyword_tier_rules match literally, the most-severe matched tier wins.

        Escalating to the highest tier (rather than the first rule in the list) keeps
        routing independent of the order rules were authored in: a prompt hitting both a
        SIMPLE and a REASONING keyword routes to REASONING.
        """
        rules = self.config.keyword_tier_rules
        if not rules:
            return None
        text = user_message.lower()
        matches = [
            KeywordOverride(tier=rule.tier, matched_keyword=matched_keyword)
            for rule in rules
            if (matched_keyword := next((kw for kw in rule.keywords if self._keyword_matches(text, kw)), None))
            is not None
        ]
        if not matches:
            return None
        return max(matches, key=lambda match: TIER_SEVERITY_ORDER.index(match.tier))

    def _get_or_create_semantic_routelayer(self) -> SemanticRouter:
        """Build (once) a SemanticRouter with one route per tier, utterances = that tier's keywords."""
        if self._semantic_routelayer is not None:
            return self._semantic_routelayer

        from semantic_router.routers import SemanticRouter
        from semantic_router.routers.base import Route

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )

        embedding_model = self.config.embedding_model
        if embedding_model is None:
            raise ValueError("embedding_model is required for semantic keyword matching")

        rules = self.config.keyword_tier_rules or []
        ordered_tiers = tuple(dict.fromkeys(rule.tier.value for rule in rules))
        routes = [
            Route(
                name=tier,
                utterances=[keyword for rule in rules if rule.tier.value == tier for keyword in rule.keywords],
                score_threshold=self.config.match_threshold,
            )
            for tier in ordered_tiers
        ]
        routelayer = SemanticRouter(
            routes=routes,
            encoder=LiteLLMRouterEncoder(
                litellm_router_instance=self.litellm_router_instance,
                model_name=embedding_model,
                score_threshold=self.config.match_threshold,
            ),
            auto_sync="local",
            aggregation="max",
        )
        self._semantic_routelayer = routelayer
        return routelayer

    async def _ensure_semantic_routelayer(self) -> SemanticRouter:
        """Return the cached route layer, building it once under a lock if needed.

        The build embeds the static route utterances via the encoder's synchronous path,
        so it runs in a worker thread to avoid blocking the event loop. A double-checked
        asyncio lock ensures concurrent cold-start requests build it exactly once rather
        than each firing duplicate embedding calls.
        """
        if self._semantic_routelayer is not None:
            return self._semantic_routelayer
        async with self._semantic_routelayer_lock:
            routelayer = self._semantic_routelayer
            if routelayer is None:
                routelayer = await asyncio.to_thread(self._get_or_create_semantic_routelayer)
            return routelayer

    async def _semantic_tier_override(self, user_message: str, request_kwargs: dict) -> ComplexityTier | None:
        """Match the prompt against keyword_tier_rules by embedding similarity.

        Embeds the query ourselves (instead of letting SemanticRouter.acall embed it
        internally) so the caller's metadata/litellm_metadata flows into aembedding()
        and this spend is attributed and budget-checked against the originating key/team,
        the same as any other litellm call. SemanticRouter.acall() has no parameter to
        pass such kwargs through to the encoder, so it's bypassed for the query embedding;
        the route index itself (static utterances, embedded once at build time with no
        caller context) is unaffected and still reused via the precomputed `vector=` path.
        """
        from semantic_router.schema import RouteChoice

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )

        routelayer = await self._ensure_semantic_routelayer()
        encoder = cast(LiteLLMRouterEncoder, routelayer.encoder)  # cast-ok: always the encoder we built above
        # Strip the parent request's budget reservation before forwarding: the reservation
        # belongs to the routed completion this embedding is helping select, not to the
        # embedding call. Forwarding it would let the embedding's cost callback finalize the
        # reservation, so the routed completion's own callback then skips incrementing the
        # key/team budget. Key/team attribution fields are preserved for spend logging.
        metadata = _classifier_call_metadata(request_kwargs.get("metadata"))
        litellm_metadata = _classifier_call_metadata(request_kwargs.get("litellm_metadata"))
        turn_off_message_logging = _effective_turn_off_message_logging(request_kwargs)
        proxy_server_request = {"body": {"model": self.config.embedding_model, "input": [user_message]}}
        query_vector = (
            await encoder.aencode_queries(
                [user_message],
                metadata=metadata,
                litellm_metadata=litellm_metadata,
                proxy_server_request=proxy_server_request,
                turn_off_message_logging=turn_off_message_logging,
            )
        )[0]
        route_choice = await routelayer.acall(vector=query_vector)

        if isinstance(route_choice, list):
            route_choice = route_choice[0] if route_choice else None
        if not isinstance(route_choice, RouteChoice) or not route_choice.name:
            return None
        try:
            return ComplexityTier(route_choice.name)
        except ValueError:
            return None

    async def _resolve_keyword_tier_override(self, user_message: str, request_kwargs: dict) -> KeywordOverride | None:
        """Resolve a keyword_tier_rule override, semantically or lexically per config.

        Returns None (no override -> fall through to the scorer) not only when no rule
        matches, but also when the semantic path fails: the embedding call can error or
        time out, and a routing helper must never turn that into a failed user request.
        """
        if not self.config.keyword_tier_rules:
            return None
        if not self.config.semantic_keyword_matching:
            return self._lexical_tier_override(user_message)
        try:
            semantic_tier = await self._semantic_tier_override(user_message, request_kwargs)
        except Exception as e:  # noqa: BLE001 -- embedding call can fail many ways (timeout, provider/network/parse error); any failure must fall back to scoring, never fail the request
            verbose_router_logger.warning(
                f"ComplexityRouter: semantic keyword matching failed ({e}), falling back to complexity scoring"
            )
            return None
        if semantic_tier is None:
            return None
        # A semantic match is a similarity hit against the rule's utterances, not a
        # literal keyword, so there is no single matched keyword to report.
        return KeywordOverride(tier=semantic_tier, matched_keyword=None)

    def _resolve_messages(
        self,
        messages: list[dict[str, Any]] | None,
        request_kwargs: dict,
    ) -> list[dict[str, Any]] | None:
        """
        Resolve messages from the request, converting from other formats if needed.

        Uses the guardrail translation handler dispatch to convert Responses API
        ``input`` (or other non-chat-completions formats) into OpenAI-spec messages.
        """
        from litellm.litellm_core_utils.prompt_templates.factory import (
            resolve_structured_messages,
        )

        return resolve_structured_messages(messages=messages, request_kwargs=request_kwargs)

    @staticmethod
    def _extract_user_message_and_system_prompt(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        """Extract the last user message text and last system prompt from messages."""
        user_message: str | None = None
        system_prompt: str | None = None

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

    async def _capture_session(
        self, request_kwargs: Mapping[str, object], messages: Sequence[Mapping[str, object]] | None, routed_model: str
    ) -> None:
        if not self.config.cache_warming.enabled:
            return
        from litellm.router_strategy.complexity_router.cache_warming.capture import capture_session

        try:
            await capture_session(self, request_kwargs, messages, routed_model)
        except Exception:  # noqa: BLE001  # a capture failure must never fail the user's request
            verbose_router_logger.exception("cache_warming capture failed; the request continues unaffected")

    def get_cache_warming_store(self) -> CacheWarmingStore | None:
        if not self.config.cache_warming.enabled:
            return None
        from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore

        router = self.litellm_router_instance
        redis_cache = router.cache.redis_cache if router is not None and router.cache is not None else None
        return CacheWarmingStore(redis_cache=redis_cache, auto_router_model_name=self.model_name)

    def _get_session_affinity_cache_key(self, session_id: str, request_kwargs: dict) -> str:
        # Namespace by the caller's API key hash so two different callers reusing the
        # same client-supplied session_id can't poison each other's routing pin. Falls
        # back to "unscoped" only when there's no authenticated caller to scope by
        # (e.g. direct Router usage without the proxy layer).
        caller_scope = get_request_metadata_field(request_kwargs, "user_api_key_hash") or "unscoped"
        return f"complexity_router_session_affinity:v1:{self.model_name}:{caller_scope}:{session_id}"

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict,
        messages: list[dict[str, Any]] | None = None,
        input: Union[str, list] | None = None,
        specific_deployment: bool | None = False,
    ) -> PreRoutingHookResponse | None:
        """
        Pre-routing hook called before the routing decision.

        When `session_affinity` is enabled and a session_id is resolvable on the request,
        pins the model chosen on the session's first turn and reuses it for every later
        turn, skipping classification entirely. Otherwise delegates to `_classify_and_route`.

        Skipped entirely when `plugins` are configured: reusing a stale pin would bypass
        the plugin pipeline on every turn after the first, since a pinned model was never
        re-checked against a policy plugin whose decision can change between turns (e.g. a
        budget plugin, once the session's spend crosses its cap).
        """
        from litellm.types.router import PreRoutingHookResponse

        if self.config.return_raw_model_name:
            metadata_key = "litellm_metadata" if "litellm_metadata" in request_kwargs else "metadata"
            metadata = request_kwargs.setdefault(metadata_key, {})
            if isinstance(metadata, dict):
                metadata[RETURN_RAW_MODEL_NAME_METADATA_KEY] = True

        use_session_affinity = self.config.session_affinity and not self.config.plugins
        session_id = get_request_metadata_field(request_kwargs, "session_id") if use_session_affinity else None
        cache_key = self._get_session_affinity_cache_key(session_id, request_kwargs) if session_id is not None else None

        if cache_key is not None:
            pinned_model = await self.litellm_router_instance.cache.async_get_cache(key=cache_key)
            if isinstance(pinned_model, str):
                routed_model: str | None = pinned_model
                pin_escalation_keyword: str | None = None
                if self.escalation_keywords:
                    resolved_messages = self._resolve_messages(messages, request_kwargs)
                    user_message = (
                        self._extract_user_message_and_system_prompt(resolved_messages)[0]
                        if resolved_messages
                        else None
                    )
                    if user_message is not None:
                        pin_escalation_keyword = self._matched_escalation_keyword(user_message)
                    if pin_escalation_keyword is not None:
                        routed_model = self._escalated_pin(pinned_model)
                if routed_model is not None:
                    # Refresh the TTL on every hit so an active session doesn't lose its
                    # pin mid-conversation just because it outlives the original write.
                    await self.litellm_router_instance.cache.async_set_cache(
                        key=cache_key,
                        value=routed_model,
                        ttl=self.config.session_affinity_ttl_seconds,
                    )
                    if self.config.adaptive:
                        from litellm.router_strategy.adaptive_router.config import (
                            ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY,
                        )

                        kwargs_metadata = request_kwargs.setdefault("metadata", {})
                        if isinstance(kwargs_metadata, dict):
                            kwargs_metadata[ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY] = routed_model
                    await self._capture_session(request_kwargs, messages, routed_model)
                    escalated = routed_model != pinned_model
                    cause: RoutingDecisionCause = "session_affinity_escalation" if escalated else "session_affinity_pin"
                    verbose_router_logger.info(
                        f"ComplexityRouter: routing decision cause={cause}, routed_model={routed_model}"
                    )
                    has_original_messages = messages is not None and len(messages) > 0
                    return PreRoutingHookResponse(
                        model=routed_model,
                        messages=messages if has_original_messages else None,
                        routing_decision=self._build_routing_decision(
                            routed_model=routed_model,
                            cause=cause,
                            escalation_keyword=pin_escalation_keyword,
                            escalated=escalated,
                        ),
                    )

        response = await self._classify_and_route(
            model=model,
            request_kwargs=request_kwargs,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )
        if cache_key is not None and response is not None:
            await self.litellm_router_instance.cache.async_set_cache(
                key=cache_key,
                value=response.model,
                ttl=self.config.session_affinity_ttl_seconds,
            )
        if response is not None:
            await self._capture_session(request_kwargs, messages, response.model)
        return response

    async def _classify_and_route(
        self,
        model: str,
        request_kwargs: dict,
        messages: list[dict[str, Any]] | None = None,
        input: Union[str, list] | None = None,
        specific_deployment: bool | None = False,
    ) -> PreRoutingHookResponse | None:
        """
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
            if not self.config.plugins and self.config.default_model:
                # No plugins configured: preserve the pre-existing default_model-first
                # priority exactly (changing it would be a silent behavior change for
                # every non-plugin user, not just a security fix).
                routed_model = self.config.default_model
            else:
                # Plugins configured: default_model must never bypass them, so it's not
                # checked here at all -- _pick_model_for_tier -> get_model_for_tier still
                # falls back to it (after the MEDIUM tier) once the plugin pipeline runs.
                routed_model = await self._pick_model_for_tier(
                    ComplexityTier.MEDIUM, messages, resolved_messages, request_kwargs
                )
            return PreRoutingHookResponse(
                model=routed_model,
                messages=messages if has_original_messages else None,
                routing_decision=self._build_routing_decision(routed_model=routed_model, cause="default_fallback"),
            )

        escalation_keyword = self._matched_escalation_keyword(user_message)

        override = await self._resolve_keyword_tier_override(user_message, request_kwargs)
        if override is not None:
            routed_tier = self._escalate_tier(override.tier) if escalation_keyword is not None else override.tier
            keyword_escalated = routed_tier != override.tier
            routed_model = await self._pick_model_for_tier(routed_tier, messages, resolved_messages, request_kwargs)
            keyword_cause: RoutingDecisionCause = (
                "semantic_keyword_match" if self.config.semantic_keyword_matching else "literal_keyword_match"
            )
            verbose_router_logger.info(
                f"ComplexityRouter: routing decision cause={keyword_cause}, escalated={keyword_escalated}, "
                f"tier={routed_tier.value}, routed_model={routed_model}"
            )
            return PreRoutingHookResponse(
                model=routed_model,
                messages=messages if has_original_messages else None,
                routing_decision=self._build_routing_decision(
                    routed_model=routed_model,
                    cause=keyword_cause,
                    tier=routed_tier,
                    matched_keyword=override.matched_keyword,
                    escalation_keyword=escalation_keyword,
                    escalated=keyword_escalated,
                ),
            )

        outcome = await self.aclassify(user_message, system_prompt, request_kwargs)
        tier, score, signals = outcome.tier, outcome.score, outcome.signals
        classified_tier = tier
        if escalation_keyword is not None:
            tier = self._escalate_tier(tier)
        escalated = tier != classified_tier
        if escalated:
            signals = (*signals, "escalation")
        score_repr = f"{score:.3f}" if score is not None else "n/a"
        if self.config.adaptive:
            routed_model = self._soft_floor_pick(tier, user_message, request_kwargs)
            adaptive = self._ensure_adaptive_router()
            if adaptive is not None:
                kwargs_metadata = request_kwargs.setdefault("metadata", {})
                if isinstance(kwargs_metadata, dict):
                    chosen_key = getattr(self, "_adaptive_chosen_model_key", "adaptive_router_chosen_model")
                    kwargs_metadata[chosen_key] = routed_model
            verbose_router_logger.info(
                f"ComplexityRouter[adaptive]: routing decision cause={outcome.cause}, "
                f"tier={tier.value}, score={score_repr}, "
                f"signals={signals}, routed_model={routed_model}"
            )
        else:
            routed_model = await self._pick_model_for_tier(tier, messages, resolved_messages, request_kwargs)
            verbose_router_logger.info(
                f"ComplexityRouter: routing decision cause={outcome.cause}, tier={tier.value}, "
                f"score={score_repr}, signals={signals}, routed_model={routed_model}"
            )

        classifier_model = (
            self.config.classifier_llm_config.model
            if outcome.cause == "llm_classifier" and self.config.classifier_llm_config is not None
            else None
        )
        return PreRoutingHookResponse(
            model=routed_model,
            messages=messages if has_original_messages else None,
            routing_decision=self._build_routing_decision(
                routed_model=routed_model,
                cause=outcome.cause,
                tier=tier,
                score=score,
                signals=signals,
                escalation_keyword=escalation_keyword,
                escalated=escalated,
                classifier_model=classifier_model,
            ),
        )
