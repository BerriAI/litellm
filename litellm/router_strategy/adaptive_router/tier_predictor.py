from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from litellm.types.router import (
    AdaptiveRouterTierArtifact,
    AdaptiveRouterTierCohortStatistic,
    AdaptiveRouterTierDomainStatistic,
    AdaptiveRouterTierGlobalStatistic,
    RequestType,
)

_CODE_PATTERN: Final = re.compile(
    r"```|\b(def|class|function|python|javascript|typescript|sql|code)\b",
    re.IGNORECASE,
)
_MATH_PATTERN: Final = re.compile(
    r"\b(solve|calculate|equation|probability|theorem|proof|integral)\b|[$=]",
    re.IGNORECASE,
)
_MULTIPLE_CHOICE_PATTERN: Final = re.compile(r"(?:^|\s)[A-D][.)]\s")
_TIERS: Final = (1, 2, 3, 4)
_BUILTIN_ARTIFACTS: Final = MappingProxyType({"ultrafeedback": "ultrafeedback_tiers.json"})


def resolve_tier_artifact(artifact: AdaptiveRouterTierArtifact | str) -> AdaptiveRouterTierArtifact:
    if isinstance(artifact, AdaptiveRouterTierArtifact):
        return artifact
    filename: Final = _BUILTIN_ARTIFACTS.get(artifact)
    if filename is None:
        raise ValueError(f"unknown adaptive router tier artifact: {artifact}")
    path: Final = Path(__file__).with_name("artifacts") / filename
    return AdaptiveRouterTierArtifact.model_validate_json(path.read_text())


def similarity_cohort(prompt: str, request_type: RequestType) -> str:
    length: Final = len(prompt)
    length_bucket: Final = (
        "short" if length < 200 else "medium" if length < 800 else "long" if length < 2000 else "very_long"
    )
    code: Final = int(bool(_CODE_PATTERN.search(prompt)))
    math: Final = int(bool(_MATH_PATTERN.search(prompt)))
    multiple_choice: Final = int(bool(_MULTIPLE_CHOICE_PATTERN.search(prompt)))
    non_ascii: Final = int(sum(ord(character) > 127 for character in prompt) / max(1, length) > 0.1)
    return f"{request_type.value}|{length_bucket}|code={code}|math={math}|mc={multiple_choice}|intl={non_ascii}"


@dataclass(frozen=True, slots=True)
class TierPrediction:
    probabilities: Mapping[int, float]
    required_tier: int


class TierSuccessPredictor:
    def __init__(self, artifact: AdaptiveRouterTierArtifact) -> None:
        self._artifact = artifact
        self._global: Mapping[int, AdaptiveRouterTierGlobalStatistic] = MappingProxyType(
            {stat.tier: stat for stat in artifact.global_statistics}
        )
        self._domain: Mapping[tuple[RequestType, int], AdaptiveRouterTierDomainStatistic] = MappingProxyType(
            {(stat.request_type, stat.tier): stat for stat in artifact.domain_statistics}
        )
        self._cohort: Mapping[tuple[str, int], AdaptiveRouterTierCohortStatistic] = MappingProxyType(
            {(stat.cohort, stat.tier): stat for stat in artifact.cohort_statistics}
        )

    @property
    def routing_threshold(self) -> float:
        return self._artifact.routing_threshold

    def predict(self, prompt: str, request_type: RequestType) -> TierPrediction:
        cohort: Final = similarity_cohort(prompt, request_type)
        raw: Final = tuple(self._probability(tier, request_type, cohort) for tier in _TIERS)
        monotonic: Final = tuple(max(raw[:index]) for index in range(1, len(raw) + 1))
        probabilities: Final[Mapping[int, float]] = MappingProxyType(
            {int(tier): probability for tier, probability in zip(_TIERS, monotonic)}
        )
        required_tier: Final = next(
            (tier for tier in _TIERS if probabilities[tier] >= self._artifact.routing_threshold),
            4,
        )
        return TierPrediction(probabilities=probabilities, required_tier=required_tier)

    def _probability(self, tier: int, request_type: RequestType, cohort: str) -> float:
        global_stat: Final = self._global[tier]
        global_mean: Final = (global_stat.successes + 1.0) / (global_stat.observations + 2.0)
        domain_stat: Final = self._domain.get((request_type, tier))
        domain_mean: Final = self._posterior_mean(domain_stat, self._artifact.domain_prior_mass, global_mean)
        cohort_stat: Final = self._cohort.get((cohort, tier))
        return self._posterior_mean(cohort_stat, self._artifact.cohort_prior_mass, domain_mean)

    @staticmethod
    def _posterior_mean(
        statistic: AdaptiveRouterTierGlobalStatistic | None,
        prior_mass: float,
        prior_mean: float,
    ) -> float:
        if statistic is None:
            return prior_mean
        return (statistic.successes + prior_mass * prior_mean) / (statistic.observations + prior_mass)
