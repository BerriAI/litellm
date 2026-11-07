"""
Hybrid Router: two-layer routing combining complexity gating with online efficiency.

Layer 1 (Complexity): classifies the request using the same heuristic scorer as the
complexity router and produces a set of eligible models from the tier's candidate pool.

Layer 2 (Efficiency): picks the best model from that set using a per-tier MAB bandit
that optimizes for latency, throughput, and optionally quality in real time.

The per-tier bandit design means each tier tracks arm performance independently.
A model might be fast for SIMPLE queries but slow for COMPLEX ones (longer generation),
and the bandits learn this separately.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.router_strategy.complexity_router.config import (
    DEFAULT_CODE_KEYWORDS,
    DEFAULT_DIMENSION_WEIGHTS,
    DEFAULT_REASONING_KEYWORDS,
    DEFAULT_SIMPLE_KEYWORDS,
    DEFAULT_TECHNICAL_KEYWORDS,
    DEFAULT_TIER_BOUNDARIES,
    DEFAULT_TOKEN_THRESHOLDS,
    ComplexityTier,
)
from litellm.router_strategy.hybrid_router.bandit import MABAlgorithm, make_bandit


@dataclass(frozen=True, slots=True)
class HybridRouterConfig:
    tier_candidates: Mapping[str, tuple[str, ...]]
    bandit: str = "thompson"
    delta: float = 0.1
    epsilon: float = 0.1
    target_tpt: float = 1.0
    tier_priors: Mapping[str, Mapping[str, tuple[float, float]]] | None = None
    dimension_weights: Mapping[str, float] | None = None
    tier_boundaries: Mapping[str, float] | None = None
    token_thresholds: Mapping[str, int] | None = None
    code_keywords: tuple[str, ...] | None = None
    reasoning_keywords: tuple[str, ...] | None = None
    technical_keywords: tuple[str, ...] | None = None
    simple_keywords: tuple[str, ...] | None = None


class TierBandit:
    """A bandit instance scoped to one complexity tier."""

    def __init__(
        self,
        tier: str,
        arms: tuple[str, ...],
        bandit_type: str,
        arm_priors: Mapping[str, tuple[float, float]] | None = None,
        **bandit_kwargs: object,
    ) -> None:
        self._tier: Final = tier
        self._arms: Final = arms
        self._bandit: Final[MABAlgorithm] = make_bandit(
            bandit_type, arms, arm_priors=arm_priors, **bandit_kwargs
        )

    @property
    def tier(self) -> str:
        return self._tier

    def pick(self) -> str:
        return self._bandit.select_arm()

    def pick_from(self, eligible: tuple[str, ...]) -> str:
        if len(eligible) == 1:
            return eligible[0]
        return self._bandit.select_arm_from(eligible)

    def update(self, arm: str, reward: float) -> None:
        self._bandit.update(arm, reward)

    def state(self) -> dict[str, object]:
        return {"tier": self._tier, **self._bandit.state()}


_MULTI_STEP_PATTERNS: Final = (
    re.compile(r"first.*?then", re.IGNORECASE),
    re.compile(r"step\s*\d", re.IGNORECASE),
    re.compile(r"\d+\.\s"),
    re.compile(r"[a-z]\)\s", re.IGNORECASE),
)


class HybridRouter:
    """
    Two-layer router: complexity classification -> per-tier MAB selection.

    Usage:
        config = HybridRouterConfig(
            tier_candidates={
                "SIMPLE": ("model-small", "model-medium"),
                "MEDIUM": ("model-medium", "model-large"),
                "COMPLEX": ("model-large",),
                "REASONING": ("model-large",),
            },
            bandit="thompson",
        )
        router = HybridRouter(config)

        # On each request:
        tier, model = router.route(user_message, system_prompt)

        # After response:
        router.record_success(tier, model, latency, completion_tokens)
    """

    def __init__(self, config: HybridRouterConfig) -> None:
        self._config: Final = config
        self._tier_candidates: Final = config.tier_candidates

        all_models: Final = tuple(
            sorted(frozenset(model for models in self._tier_candidates.values() for model in models))
        )
        self._all_models: Final = all_models

        bandit_kwargs: Final[dict[str, object]] = {"delta": config.delta, "epsilon": config.epsilon}
        tier_bandits: Final[dict[str, TierBandit]] = {}
        for tier_name, candidates in self._tier_candidates.items():
            if len(candidates) >= 2:
                arm_priors = (config.tier_priors or {}).get(tier_name)
                tier_bandits[tier_name] = TierBandit(
                    tier=tier_name,
                    arms=candidates,
                    bandit_type=config.bandit,
                    arm_priors=arm_priors,
                    **bandit_kwargs,
                )
        self._tier_bandits: Final = tier_bandits

        self._code_keywords: Final = config.code_keywords or tuple(DEFAULT_CODE_KEYWORDS)
        self._reasoning_keywords: Final = config.reasoning_keywords or tuple(DEFAULT_REASONING_KEYWORDS)
        self._technical_keywords: Final = config.technical_keywords or tuple(DEFAULT_TECHNICAL_KEYWORDS)
        self._simple_keywords: Final = config.simple_keywords or tuple(DEFAULT_SIMPLE_KEYWORDS)
        self._dimension_weights: Final = config.dimension_weights or MappingProxyType(DEFAULT_DIMENSION_WEIGHTS)
        self._tier_boundaries: Final = config.tier_boundaries or MappingProxyType(DEFAULT_TIER_BOUNDARIES)
        self._token_thresholds: Final = config.token_thresholds or MappingProxyType(DEFAULT_TOKEN_THRESHOLDS)

        self._lock: Final = threading.Lock()

    @property
    def config(self) -> HybridRouterConfig:
        return self._config

    def classify(
        self, user_message: str, system_prompt: str | None = None
    ) -> tuple[ComplexityTier, float, tuple[str, ...]]:
        user_text: Final = user_message.lower()
        estimated_tokens: Final = len(user_message) // 4

        simple_threshold: Final = self._token_thresholds.get("simple", 15)
        complex_threshold: Final = self._token_thresholds.get("complex", 400)

        scores: dict[str, float] = {}
        signals: list[str] = []

        if estimated_tokens < simple_threshold:
            scores["tokenCount"] = -1.0
        elif estimated_tokens > complex_threshold:
            scores["tokenCount"] = 1.0
        else:
            scores["tokenCount"] = 0.0

        def keyword_score(
            text: str,
            keywords: tuple[str, ...],
            name: str,
            label: str,
            thresholds: tuple[int, int],
            score_vals: tuple[float, float, float],
        ) -> int:
            matches: Final = tuple(
                kw for kw in keywords if _keyword_matches(text, kw)
            )
            count: Final = len(matches)
            low_t, high_t = thresholds
            if count >= high_t:
                scores[name] = score_vals[2]
                signals.append(f"{label} ({', '.join(matches[:3])})")
            elif count >= low_t:
                scores[name] = score_vals[1]
                signals.append(f"{label} ({', '.join(matches[:3])})")
            else:
                scores[name] = score_vals[0]
            return count

        keyword_score(user_text, self._code_keywords, "codePresence", "code", (1, 2), (0, 0.5, 1.0))
        reasoning_count: Final = keyword_score(
            user_text, self._reasoning_keywords, "reasoningMarkers", "reasoning", (1, 2), (0, 0.7, 1.0)
        )
        keyword_score(user_text, self._technical_keywords, "technicalTerms", "technical", (2, 4), (0, 0.5, 1.0))
        keyword_score(user_text, self._simple_keywords, "simpleIndicators", "simple", (1, 2), (0, -1.0, -1.0))

        multi_step_hits: Final = sum(1 for p in _MULTI_STEP_PATTERNS if p.search(user_text))
        scores["multiStepPatterns"] = 0.5 if multi_step_hits > 0 else 0.0

        q_count: Final = user_message.count("?")
        scores["questionComplexity"] = 0.5 if q_count > 3 else 0.0

        weighted_score: Final = sum(scores.get(name, 0) * w for name, w in self._dimension_weights.items())

        if reasoning_count >= 2:
            return ComplexityTier.REASONING, weighted_score, tuple(signals)

        simple_medium: Final = self._tier_boundaries.get("simple_medium", 0.15)
        medium_complex: Final = self._tier_boundaries.get("medium_complex", 0.35)
        complex_reasoning: Final = self._tier_boundaries.get("complex_reasoning", 0.60)

        if weighted_score < simple_medium:
            tier = ComplexityTier.SIMPLE
        elif weighted_score < medium_complex:
            tier = ComplexityTier.MEDIUM
        elif weighted_score < complex_reasoning:
            tier = ComplexityTier.COMPLEX
        else:
            tier = ComplexityTier.REASONING

        return tier, weighted_score, tuple(signals)

    def get_candidates(self, tier: ComplexityTier) -> tuple[str, ...]:
        tier_key: Final = tier.value if isinstance(tier, ComplexityTier) else tier
        candidates = self._tier_candidates.get(tier_key)
        if candidates:
            return candidates
        for fallback_key in ("MEDIUM", "COMPLEX", "SIMPLE", "REASONING"):
            if fallback_key in self._tier_candidates:
                return self._tier_candidates[fallback_key]
        return self._all_models

    def pick_model(self, tier: ComplexityTier) -> str:
        tier_key: Final = tier.value if isinstance(tier, ComplexityTier) else tier
        candidates: Final = self.get_candidates(tier)

        if len(candidates) == 1:
            return candidates[0]

        bandit = self._tier_bandits.get(tier_key)
        if bandit is None:
            return candidates[0]

        return bandit.pick_from(candidates)

    def route(self, user_message: str, system_prompt: str | None = None) -> tuple[ComplexityTier, str]:
        tier, _score, _signals = self.classify(user_message, system_prompt)
        model: Final = self.pick_model(tier)
        return tier, model

    def _compute_reward(self, latency_seconds: float, completion_tokens: int) -> float:
        time_per_token: Final = latency_seconds / max(completion_tokens, 1)
        return self._config.target_tpt / (self._config.target_tpt + time_per_token)

    def record_success(
        self, tier: ComplexityTier, model: str, latency_seconds: float, completion_tokens: int = 0
    ) -> float:
        tier_key: Final = tier.value if isinstance(tier, ComplexityTier) else tier
        reward: Final = self._compute_reward(latency_seconds, completion_tokens)
        bandit = self._tier_bandits.get(tier_key)
        if bandit is not None:
            bandit.update(model, reward)
        return reward

    def record_quality(
        self,
        tier: ComplexityTier,
        model: str,
        latency_seconds: float,
        completion_tokens: int,
        quality_score: float,
        quality_weight: float = 0.5,
    ) -> float:
        """Record a composite reward blending efficiency and quality.

        quality_score: Judge/quality score in [0, 1].
        quality_weight: How much to weight quality vs efficiency. 0.5 = equal.
        """
        tier_key: Final = tier.value if isinstance(tier, ComplexityTier) else tier
        eff_reward: Final = self._compute_reward(latency_seconds, completion_tokens)
        composite: Final = (1.0 - quality_weight) * eff_reward + quality_weight * quality_score
        bandit = self._tier_bandits.get(tier_key)
        if bandit is not None:
            bandit.update(model, composite)
        return composite

    def record_failure(self, tier: ComplexityTier, model: str) -> None:
        tier_key: Final = tier.value if isinstance(tier, ComplexityTier) else tier
        bandit = self._tier_bandits.get(tier_key)
        if bandit is not None:
            bandit.update(model, 0.0)

    def state(self) -> dict[str, object]:
        return {
            "config": {
                "bandit": self._config.bandit,
                "tier_candidates": {k: list(v) for k, v in self._config.tier_candidates.items()},
                "target_tpt": self._config.target_tpt,
            },
            "tier_bandits": {tier: bandit.state() for tier, bandit in self._tier_bandits.items()},
        }


def _keyword_matches(text: str, keyword: str) -> bool:
    kw_lower: Final = keyword.lower()
    if " " in kw_lower:
        return kw_lower in text
    return bool(re.search(r"\b" + re.escape(kw_lower) + r"\b", text))
