from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, Field, model_validator

from litellm.types.router import RequestType


class TierGlobalStatistic(BaseModel):
    tier: int = Field(ge=1, le=4)
    successes: float = Field(ge=0.0)
    observations: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _successes_do_not_exceed_observations(self) -> TierGlobalStatistic:
        if self.successes > self.observations:
            raise ValueError("successes cannot exceed observations")
        return self


class TierDomainStatistic(TierGlobalStatistic):
    request_type: RequestType


class TierCohortStatistic(TierGlobalStatistic):
    cohort: str = Field(min_length=1)


class TierDataset(BaseModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    rows: int = Field(gt=0)
    success_definition: str = Field(default="quality score meets the dataset success threshold", min_length=1)


class TrainedTierArtifact(BaseModel):
    schema_version: Literal[1] = 1
    global_statistics: tuple[TierGlobalStatistic, ...]
    domain_statistics: tuple[TierDomainStatistic, ...] = ()
    cohort_statistics: tuple[TierCohortStatistic, ...] = ()
    domain_prior_mass: float = Field(default=200.0, gt=0.0)
    cohort_prior_mass: float = Field(default=20.0, gt=0.0)
    routing_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    datasets: tuple[TierDataset, ...] = ()
    success_definition: str = Field(default="quality score meets the dataset success threshold", min_length=1)
    split_method: str = Field(default="sha256(prompt): 70% train, 15% validation, 15% test", min_length=1)

    @model_validator(mode="after")
    def _statistics_are_unique(self) -> TrainedTierArtifact:
        global_tiers: Final = tuple(stat.tier for stat in self.global_statistics)
        if frozenset(global_tiers) != frozenset((1, 2, 3, 4)) or len(global_tiers) != 4:
            raise ValueError("global statistics must contain each tier exactly once")
        domain_keys: Final = tuple((stat.request_type, stat.tier) for stat in self.domain_statistics)
        if len(domain_keys) != len(frozenset(domain_keys)):
            raise ValueError("domain statistics must contain unique request_type and tier pairs")
        cohort_keys: Final = tuple((stat.cohort, stat.tier) for stat in self.cohort_statistics)
        if len(cohort_keys) != len(frozenset(cohort_keys)):
            raise ValueError("cohort statistics must contain unique cohort and tier pairs")
        return self

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


def resolve_tier_artifact(artifact: TrainedTierArtifact | str) -> TrainedTierArtifact:
    if isinstance(artifact, TrainedTierArtifact):
        return artifact
    filename: Final = _BUILTIN_ARTIFACTS.get(artifact)
    if filename is None:
        raise ValueError(f"unknown complexity router tier artifact: {artifact}")
    path: Final = Path(__file__).with_name("artifacts") / filename
    return TrainedTierArtifact.model_validate_json(path.read_text())


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
    def __init__(self, artifact: TrainedTierArtifact) -> None:
        self._artifact = artifact
        self._global: Mapping[int, TierGlobalStatistic] = MappingProxyType(
            {stat.tier: stat for stat in artifact.global_statistics}
        )
        self._domain: Mapping[tuple[RequestType, int], TierDomainStatistic] = MappingProxyType(
            {(stat.request_type, stat.tier): stat for stat in artifact.domain_statistics}
        )
        self._cohort: Mapping[tuple[str, int], TierCohortStatistic] = MappingProxyType(
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
        statistic: TierGlobalStatistic | None,
        prior_mass: float,
        prior_mean: float,
    ) -> float:
        if statistic is None:
            return prior_mean
        return (statistic.successes + prior_mass * prior_mean) / (statistic.observations + prior_mass)
