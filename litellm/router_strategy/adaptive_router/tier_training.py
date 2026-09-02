from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, Field

from litellm.router_strategy.adaptive_router.classifier import classify_prompt
from litellm.router_strategy.adaptive_router.tier_predictor import (
    TierSuccessPredictor,
    similarity_cohort,
)
from litellm.types.router import (
    AdaptiveRouterTierArtifact,
    AdaptiveRouterTierCohortStatistic,
    AdaptiveRouterTierDataset,
    AdaptiveRouterTierDomainStatistic,
    AdaptiveRouterTierGlobalStatistic,
    RequestType,
)

_TIERS: Final = (1, 2, 3, 4)
_TIER_COSTS: Final = MappingProxyType({1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0})
_DOMAIN_MASS_CANDIDATES: Final = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
_COHORT_MASS_CANDIDATES: Final = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
_THRESHOLD_CANDIDATES: Final = (0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95)


class TierTrainingRecord(BaseModel):
    prompt: str = Field(min_length=1)
    tier: int = Field(ge=1, le=4)
    success: float = Field(ge=0.0, le=1.0)


class _Arguments(BaseModel):
    records: Path
    artifact_output: Path
    dataset_name: str
    dataset_url: str
    dataset_license: str
    success_definition: str


class ProbabilityMetrics(BaseModel):
    rows: float
    brier: float
    log_loss: float
    ece: float


class RouteMetrics(BaseModel):
    prompts: float
    success_rate: float
    relative_tier_cost: float
    quality_cost_utility: float


class RouterBenchmark(BaseModel):
    objective: str
    validation: RouteMetrics
    test: RouteMetrics
    test_baselines: Mapping[int, RouteMetrics]


class TierTrainingReport(BaseModel):
    split: str
    probability_metrics: Mapping[str, ProbabilityMetrics]
    router_benchmark: RouterBenchmark


@dataclass(frozen=True, slots=True)
class _Record:
    prompt: str
    tier: int
    success: float
    split: Literal["train", "validation", "test"]
    request_type: RequestType
    cohort: str


@dataclass(frozen=True, slots=True)
class TierTrainingResult:
    artifact: AdaptiveRouterTierArtifact
    report: TierTrainingReport


def stable_split(prompt: str) -> Literal["train", "validation", "test"]:
    bucket: Final = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def train_tier_artifact(
    records: Sequence[TierTrainingRecord],
    dataset: AdaptiveRouterTierDataset,
) -> TierTrainingResult:
    enriched: Final = tuple(_enrich(record) for record in records)
    training: Final = tuple(record for record in enriched if record.split == "train")
    global_statistics: Final = _global_statistics(training)
    domain_statistics: Final = _domain_statistics(training)
    cohort_statistics: Final = _cohort_statistics(training)
    domain_mass, cohort_mass = _tune_masses(
        enriched,
        global_statistics,
        domain_statistics,
        cohort_statistics,
        dataset,
    )
    threshold: Final = _tune_threshold(
        enriched,
        _artifact(
            global_statistics,
            domain_statistics,
            cohort_statistics,
            domain_mass,
            cohort_mass,
            0.75,
            dataset,
        ),
    )
    artifact: Final = _artifact(
        global_statistics,
        domain_statistics,
        cohort_statistics,
        domain_mass,
        cohort_mass,
        threshold,
        dataset,
    )
    return TierTrainingResult(
        artifact=artifact,
        report=TierTrainingReport(
            split="sha256(prompt): 70% train, 15% validation, 15% test",
            probability_metrics=MappingProxyType(
                {
                    "validation": _probability_metrics(enriched, "validation", artifact),
                    "test": _probability_metrics(enriched, "test", artifact),
                }
            ),
            router_benchmark=_router_benchmark(enriched, artifact),
        ),
    )


def _enrich(record: TierTrainingRecord) -> _Record:
    request_type: Final = classify_prompt(record.prompt)
    return _Record(
        prompt=record.prompt,
        tier=record.tier,
        success=record.success,
        split=stable_split(record.prompt),
        request_type=request_type,
        cohort=similarity_cohort(record.prompt, request_type),
    )


def _grouped_stats(
    records: Sequence[_Record],
    key: Callable[[_Record], str],
) -> tuple[tuple[str, float, float], ...]:
    ordered: Final = sorted(records, key=key)
    return tuple(
        (group_key, sum(record.success for record in group_records), float(len(group_records)))
        for group_key, grouped in groupby(ordered, key=key)
        for group_records in (tuple(grouped),)
    )


def _global_statistics(records: Sequence[_Record]) -> tuple[AdaptiveRouterTierGlobalStatistic, ...]:
    by_tier: Final = MappingProxyType(
        {
            int(key): (successes, observations)
            for key, successes, observations in _grouped_stats(records, lambda row: str(row.tier))
        }
    )
    return tuple(
        AdaptiveRouterTierGlobalStatistic(
            tier=tier,
            successes=by_tier[tier][0],
            observations=by_tier[tier][1],
        )
        for tier in _TIERS
    )


def _domain_statistics(records: Sequence[_Record]) -> tuple[AdaptiveRouterTierDomainStatistic, ...]:
    return tuple(
        AdaptiveRouterTierDomainStatistic(
            request_type=RequestType(key.rsplit("\0", 1)[0]),
            tier=int(key.rsplit("\0", 1)[1]),
            successes=successes,
            observations=observations,
        )
        for key, successes, observations in _grouped_stats(records, lambda row: f"{row.request_type.value}\0{row.tier}")
    )


def _cohort_statistics(records: Sequence[_Record]) -> tuple[AdaptiveRouterTierCohortStatistic, ...]:
    return tuple(
        AdaptiveRouterTierCohortStatistic(
            cohort=key.rsplit("\0", 1)[0],
            tier=int(key.rsplit("\0", 1)[1]),
            successes=successes,
            observations=observations,
        )
        for key, successes, observations in _grouped_stats(records, lambda row: f"{row.cohort}\0{row.tier}")
    )


def _artifact(
    global_statistics: tuple[AdaptiveRouterTierGlobalStatistic, ...],
    domain_statistics: tuple[AdaptiveRouterTierDomainStatistic, ...],
    cohort_statistics: tuple[AdaptiveRouterTierCohortStatistic, ...],
    domain_mass: float,
    cohort_mass: float,
    threshold: float,
    dataset: AdaptiveRouterTierDataset,
) -> AdaptiveRouterTierArtifact:
    return AdaptiveRouterTierArtifact(
        global_statistics=global_statistics,
        domain_statistics=domain_statistics,
        cohort_statistics=cohort_statistics,
        domain_prior_mass=domain_mass,
        cohort_prior_mass=cohort_mass,
        routing_threshold=threshold,
        datasets=(dataset,),
        success_definition=dataset.success_definition,
    )


def _tune_masses(
    records: Sequence[_Record],
    global_statistics: tuple[AdaptiveRouterTierGlobalStatistic, ...],
    domain_statistics: tuple[AdaptiveRouterTierDomainStatistic, ...],
    cohort_statistics: tuple[AdaptiveRouterTierCohortStatistic, ...],
    dataset: AdaptiveRouterTierDataset,
) -> tuple[float, float]:
    candidates: Final = tuple(
        (
            _probability_metrics(
                records,
                "validation",
                _artifact(
                    global_statistics,
                    domain_statistics,
                    cohort_statistics,
                    domain_mass,
                    cohort_mass,
                    0.75,
                    dataset,
                ),
            ).brier,
            domain_mass,
            cohort_mass,
        )
        for domain_mass in _DOMAIN_MASS_CANDIDATES
        for cohort_mass in _COHORT_MASS_CANDIDATES
    )
    _, domain_mass, cohort_mass = min(candidates)
    return domain_mass, cohort_mass


def _probability_metrics(
    records: Sequence[_Record],
    split: Literal["validation", "test"],
    artifact: AdaptiveRouterTierArtifact,
) -> ProbabilityMetrics:
    rows: Final = tuple(record for record in records if record.split == split)
    predictor: Final = TierSuccessPredictor(artifact)
    outcomes: Final = tuple(record.success for record in rows)
    predictions: Final = tuple(
        predictor.predict(record.prompt, record.request_type).probabilities[record.tier] for record in rows
    )
    brier: Final = sum((prediction - outcome) ** 2 for prediction, outcome in zip(predictions, outcomes)) / len(rows)
    log_loss: Final = -sum(
        outcome * math.log(max(1e-9, prediction)) + (1.0 - outcome) * math.log(max(1e-9, 1.0 - prediction))
        for prediction, outcome in zip(predictions, outcomes)
    ) / len(rows)
    ece: Final = sum(
        len(bucket)
        / len(rows)
        * abs(sum(item[0] for item in bucket) / len(bucket) - sum(item[1] for item in bucket) / len(bucket))
        for index in range(10)
        for bucket in (
            tuple(
                (prediction, outcome)
                for prediction, outcome in zip(predictions, outcomes)
                if min(9, int(prediction * 10)) == index
            ),
        )
        if bucket
    )
    return ProbabilityMetrics(rows=float(len(rows)), brier=brier, log_loss=log_loss, ece=ece)


def _prompt_outcomes(
    records: Sequence[_Record], split: Literal["validation", "test"]
) -> tuple[tuple[str, RequestType, Mapping[int, float]], ...]:
    rows: Final = sorted((record for record in records if record.split == split), key=lambda row: row.prompt)
    return tuple(
        (prompt, prompt_records[0].request_type, _tier_outcomes(prompt_records))
        for prompt, grouped in groupby(rows, key=lambda row: row.prompt)
        for prompt_records in (tuple(grouped),)
    )


def _tier_outcomes(records: Sequence[_Record]) -> Mapping[int, float]:
    ordered: Final = sorted(records, key=lambda row: row.tier)
    return MappingProxyType(
        {
            tier: sum(record.success for record in tier_records) / len(tier_records)
            for tier, grouped in groupby(ordered, key=lambda row: row.tier)
            for tier_records in (tuple(grouped),)
        }
    )


def _route_metrics(
    records: Sequence[_Record],
    split: Literal["validation", "test"],
    artifact: AdaptiveRouterTierArtifact,
) -> RouteMetrics:
    predictor: Final = TierSuccessPredictor(artifact)
    prompts: Final = tuple(row for row in _prompt_outcomes(records, split) if frozenset(row[2]) == frozenset(_TIERS))
    chosen: Final = tuple(
        (predictor.predict(prompt, request_type).required_tier, outcomes) for prompt, request_type, outcomes in prompts
    )
    success_rate: Final = sum(outcomes[tier] for tier, outcomes in chosen) / len(chosen)
    relative_cost: Final = sum(_TIER_COSTS[tier] for tier, _ in chosen) / len(chosen)
    utility: Final = sum(0.7 * outcomes[tier] + 0.3 * (1.0 - (tier - 1) / 3.0) for tier, outcomes in chosen) / len(
        chosen
    )
    return RouteMetrics(
        prompts=float(len(prompts)),
        success_rate=success_rate,
        relative_tier_cost=relative_cost,
        quality_cost_utility=utility,
    )


def _tune_threshold(records: Sequence[_Record], artifact: AdaptiveRouterTierArtifact) -> float:
    candidates: Final = tuple(
        (
            _route_metrics(
                records,
                "validation",
                artifact.model_copy(
                    update={"routing_threshold": threshold}  # mutable-ok: Pydantic model_copy requires a dict
                ),
            ).quality_cost_utility,
            threshold,
        )
        for threshold in _THRESHOLD_CANDIDATES
    )
    return max(candidates)[1]


def _router_benchmark(records: Sequence[_Record], artifact: AdaptiveRouterTierArtifact) -> RouterBenchmark:
    test_prompts: Final = tuple(
        row for row in _prompt_outcomes(records, "test") if frozenset(row[2]) == frozenset(_TIERS)
    )
    return RouterBenchmark(
        objective="0.7 * observed success + 0.3 * normalized tier cost",
        validation=_route_metrics(records, "validation", artifact),
        test=_route_metrics(records, "test", artifact),
        test_baselines=MappingProxyType(
            {
                tier: RouteMetrics(
                    prompts=float(len(test_prompts)),
                    success_rate=sum(outcomes[tier] for _, _, outcomes in test_prompts) / len(test_prompts),
                    relative_tier_cost=_TIER_COSTS[tier],
                    quality_cost_utility=sum(
                        0.7 * outcomes[tier] + 0.3 * (1.0 - (tier - 1) / 3.0) for _, _, outcomes in test_prompts
                    )
                    / len(test_prompts),
                )
                for tier in _TIERS
            }
        ),
    )


def _read_records(path: Path) -> tuple[TierTrainingRecord, ...]:
    with path.open() as record_file:
        return tuple(TierTrainingRecord.model_validate_json(line) for line in record_file if line.strip())


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(description="Train and benchmark a tier-based adaptive-router artifact")
    parser.add_argument("records", type=Path)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--dataset-license", required=True)
    parser.add_argument("--success-definition", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    raw_args: Final[object] = vars(_parser().parse_args(argv))
    args: Final = _Arguments.model_validate(raw_args)
    records: Final = _read_records(args.records)
    dataset: Final = AdaptiveRouterTierDataset(
        name=args.dataset_name,
        url=args.dataset_url,
        license=args.dataset_license,
        rows=len(records),
        success_definition=args.success_definition,
    )
    result: Final = train_tier_artifact(records, dataset)
    args.artifact_output.write_text(result.artifact.model_dump_json(indent=2) + "\n")
    print(result.report.model_dump_json(indent=2))  # noqa: T201  # CLI emits benchmark JSON
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
