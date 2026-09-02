from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import (
    CapabilityBoundary,
    CapabilityCalibrationBin,
    CapabilityRouterCandidate,
    CapabilityRouterConfig,
    CapabilityRule,
    CapabilityRuleBoundary,
    indexed_rules,
)
from .policy import BOUNDARY_THRESHOLD_STEPS, calibrated_probability

_THRESHOLDS: Final = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
_THRESHOLD_STEPS: Final = (0.0, 0.05, 0.1, 0.15)


class CapabilityTrainingRecord(BaseModel):
    benchmark: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    model: str = Field(min_length=1)
    primary_rule: str = Field(default="none", min_length=1)
    raw_p_solve: float = Field(ge=0.0, le=1.0)
    success: float = Field(ge=0.0, le=1.0)
    estimated_cost: float = Field(ge=0.0)

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CapabilityRuleStatistic(BaseModel):
    model: str
    rule_id: str
    observations: int
    success_rate: float
    interval_low: float
    interval_high: float
    learned_boundary: CapabilityBoundary

    model_config = ConfigDict(frozen=True)


class CapabilityProbabilityMetrics(BaseModel):
    observations: int
    brier: float
    log_loss: float
    ece: float

    model_config = ConfigDict(frozen=True)


class CapabilityRouteMetrics(BaseModel):
    tasks: int
    success_rate: float
    mean_cost: float
    normalized_cost: float
    quality_cost_utility: float

    model_config = ConfigDict(frozen=True)


class CapabilityThresholdPoint(BaseModel):
    probability_threshold: float
    threshold_step: float
    metrics: CapabilityRouteMetrics

    model_config = ConfigDict(frozen=True)


class CapabilityTrainingReport(BaseModel):
    objective: str
    validation: CapabilityRouteMetrics
    test: CapabilityRouteMetrics
    test_untrained: CapabilityRouteMetrics
    test_always_candidates: Mapping[str, CapabilityRouteMetrics]
    test_oracle: CapabilityRouteMetrics
    test_threshold_sweep: tuple[CapabilityThresholdPoint, ...]
    test_probability_raw: CapabilityProbabilityMetrics
    test_probability_calibrated: CapabilityProbabilityMetrics

    model_config = ConfigDict(frozen=True)


class CapabilityTrainingArtifact(BaseModel):
    config: CapabilityRouterConfig
    rule_statistics: tuple[CapabilityRuleStatistic, ...]
    datasets: tuple[str, ...]
    records: int
    split_counts: Mapping[str, int]
    records_sha256: str
    split_contract: str

    model_config = ConfigDict(frozen=True)


class _Arguments(BaseModel):
    records: Path
    config: Path
    artifact_output: Path
    quality_weight: float


@dataclass(frozen=True, slots=True)
class CapabilityTrainingResult:
    artifact: CapabilityTrainingArtifact
    report: CapabilityTrainingReport


@dataclass(frozen=True, slots=True)
class _CalibrationBlock:
    upper_bound: float
    successes: float
    observations: int

    @property
    def probability(self) -> float:
        return (self.successes + 1.0) / (self.observations + 2.0)


@dataclass(frozen=True, slots=True)
class _TaskCandidate:
    model: str
    probability: float
    success: float
    cost: float
    boundary: CapabilityBoundary


def _pool_adjacent_violators(blocks: tuple[_CalibrationBlock, ...]) -> tuple[_CalibrationBlock, ...]:
    violation: Final = next(
        (index for index in range(len(blocks) - 1) if blocks[index].probability > blocks[index + 1].probability),
        None,
    )
    if violation is None:
        return blocks
    left: Final = blocks[violation]
    right: Final = blocks[violation + 1]
    merged: Final = _CalibrationBlock(
        upper_bound=right.upper_bound,
        successes=left.successes + right.successes,
        observations=left.observations + right.observations,
    )
    return _pool_adjacent_violators((*blocks[:violation], merged, *blocks[violation + 2 :]))


def fit_probability_calibration(
    records: Sequence[CapabilityTrainingRecord], max_bins: int = 10
) -> tuple[CapabilityCalibrationBin, ...]:
    if max_bins < 1:
        raise ValueError("max_bins must be at least 1")
    if not records:
        return ()
    ordered: Final = tuple(sorted(records, key=lambda record: record.raw_p_solve))
    bin_count: Final = min(max_bins, max(1, int(math.sqrt(len(ordered)))))
    cutpoints: Final = tuple(
        sorted(
            frozenset(
                ordered[min(len(ordered) - 1, math.ceil(len(ordered) * index / bin_count) - 1)].raw_p_solve
                for index in range(1, bin_count + 1)
            )
        )
    )
    chunks: Final = tuple(
        tuple(
            record
            for record in ordered
            if (index == 0 or record.raw_p_solve > cutpoints[index - 1]) and record.raw_p_solve <= cutpoint
        )
        for index, cutpoint in enumerate(cutpoints)
    )
    blocks: Final = _pool_adjacent_violators(
        tuple(
            _CalibrationBlock(
                upper_bound=max(record.raw_p_solve for record in chunk),
                successes=sum(record.success for record in chunk),
                observations=len(chunk),
            )
            for chunk in chunks
        )
    )
    return tuple(
        CapabilityCalibrationBin(
            upper_bound=1.0 if index == len(blocks) - 1 else block.upper_bound,
            probability=block.probability,
        )
        for index, block in enumerate(blocks)
    )


def _wilson_interval(successes: float, observations: int) -> tuple[float, float]:
    if observations == 0:
        return 0.0, 1.0
    z: Final = 1.959963984540054
    mean: Final = successes / observations
    denominator: Final = 1.0 + z**2 / observations
    center: Final = (mean + z**2 / (2.0 * observations)) / denominator
    margin: Final = z * math.sqrt(mean * (1.0 - mean) / observations + z**2 / (4.0 * observations**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _learned_boundary(
    current: CapabilityRuleBoundary, successes: float, observations: int, target_success: float
) -> tuple[CapabilityRuleBoundary, float, float]:
    low, high = _wilson_interval(successes, observations)
    if observations == 0:
        return current, low, high
    if low >= target_success:
        return "supported", low, high
    if high < target_success:
        return "unsupported", low, high
    return "uncertain", low, high


def _train_candidate(
    candidate: CapabilityRouterCandidate,
    records: Sequence[CapabilityTrainingRecord],
    target_success: float,
) -> tuple[CapabilityRouterCandidate, tuple[CapabilityRuleStatistic, ...]]:
    candidate_records: Final = tuple(record for record in records if record.model == candidate.model)
    calibration: Final = fit_probability_calibration(candidate_records)
    learned: Final = tuple(
        (
            rule_id,
            rule,
            tuple(record for record in candidate_records if record.primary_rule == rule_id),
        )
        for rule_id, rule in indexed_rules(candidate)
    )
    boundaries: Final = tuple(
        _learned_boundary(
            rule.boundary,
            sum(record.success for record in rule_records),
            len(rule_records),
            target_success,
        )
        for _, rule, rule_records in learned
    )
    rules: Final = tuple(
        CapabilityRule(boundary=boundary[0], rule=rule.rule)
        for (_, rule, _), boundary in zip(learned, boundaries)
    )
    statistics: Final = tuple(
        CapabilityRuleStatistic(
            model=candidate.model,
            rule_id=rule_id,
            observations=len(rule_records),
            success_rate=(
                sum(record.success for record in rule_records) / len(rule_records) if rule_records else 0.0
            ),
            interval_low=boundary[1],
            interval_high=boundary[2],
            learned_boundary=boundary[0],
        )
        for (rule_id, _, rule_records), boundary in zip(learned, boundaries)
    )
    return candidate.model_copy(update={"rules": rules, "probability_calibration": calibration}), statistics


def _effective_boundary(candidate: CapabilityRouterCandidate, primary_rule: str) -> CapabilityBoundary:
    if not candidate.rules:
        return "unmatched"
    boundaries: Final[dict[str, CapabilityBoundary]] = {
        rule_id: rule.boundary for rule_id, rule in indexed_rules(candidate)
    }
    return boundaries.get(primary_rule, "unmatched")


def _task_candidates(
    records: Sequence[CapabilityTrainingRecord],
    split: Literal["validation", "test"],
    config: CapabilityRouterConfig,
    calibrated: bool,
) -> tuple[tuple[_TaskCandidate, ...], ...]:
    candidates: Final = {candidate.model: candidate for candidate in config.candidates}
    selected: Final = sorted(
        (record for record in records if record.split == split and record.model in candidates),
        key=lambda record: (record.benchmark, record.task_id, record.model),
    )
    grouped_tasks: Final = tuple(
        tuple(group)
        for _, group in groupby(selected, key=lambda record: (record.benchmark, record.task_id))
    )
    return tuple(
        tuple(
            _aggregate_candidate(tuple(model_records), candidates[model], calibrated)
            for model, model_records in groupby(task_records, key=lambda record: record.model)
        )
        for task_records in grouped_tasks
        if frozenset(record.model for record in task_records) == frozenset(candidates)
    )


def _aggregate_candidate(
    records: tuple[CapabilityTrainingRecord, ...],
    candidate: CapabilityRouterCandidate,
    calibrated: bool,
) -> _TaskCandidate:
    raw_probability: Final = sum(record.raw_p_solve for record in records) / len(records)
    primary_rule: Final = min(Counter(record.primary_rule for record in records).items(), key=lambda item: (-item[1], item[0]))[0]
    return _TaskCandidate(
        model=candidate.model,
        probability=calibrated_probability(candidate, raw_probability) if calibrated else raw_probability,
        success=sum(record.success for record in records) / len(records),
        cost=sum(record.estimated_cost for record in records) / len(records),
        boundary=_effective_boundary(candidate, primary_rule),
    )


def _route_metrics(
    records: Sequence[CapabilityTrainingRecord],
    split: Literal["validation", "test"],
    config: CapabilityRouterConfig,
    quality_weight: float,
    calibrated: bool,
) -> CapabilityRouteMetrics:
    tasks: Final = _task_candidates(records, split, config, calibrated)
    if not tasks:
        raise ValueError(f"{split} has no tasks with outcomes for every configured candidate")
    order: Final = {candidate.model: index for index, candidate in enumerate(config.candidates)}
    selected: Final = tuple(_select_task_candidate(task, config, order) for task in tasks)
    return _summarize_routes(tasks, selected, quality_weight)


def _summarize_routes(
    tasks: tuple[tuple[_TaskCandidate, ...], ...],
    selected: tuple[_TaskCandidate, ...],
    quality_weight: float,
) -> CapabilityRouteMetrics:
    normalized_costs: Final = tuple(_normalized_task_cost(choice, task) for choice, task in zip(selected, tasks))
    success_rate: Final = sum(choice.success for choice in selected) / len(selected)
    normalized_cost: Final = sum(normalized_costs) / len(normalized_costs)
    return CapabilityRouteMetrics(
        tasks=len(tasks),
        success_rate=success_rate,
        mean_cost=sum(choice.cost for choice in selected) / len(selected),
        normalized_cost=normalized_cost,
        quality_cost_utility=quality_weight * success_rate + (1.0 - quality_weight) * (1.0 - normalized_cost),
    )


def _always_candidate_metrics(
    records: Sequence[CapabilityTrainingRecord],
    config: CapabilityRouterConfig,
    quality_weight: float,
) -> Mapping[str, CapabilityRouteMetrics]:
    tasks: Final = _task_candidates(records, "test", config, True)
    return {
        model: _summarize_routes(
            tasks,
            tuple(next(candidate for candidate in task if candidate.model == model) for task in tasks),
            quality_weight,
        )
        for model in (candidate.model for candidate in config.candidates)
    }


def _oracle_metrics(
    records: Sequence[CapabilityTrainingRecord],
    config: CapabilityRouterConfig,
    quality_weight: float,
) -> CapabilityRouteMetrics:
    tasks: Final = _task_candidates(records, "test", config, True)
    selected: Final = tuple(
        max(
            task,
            key=lambda candidate: (
                quality_weight * candidate.success
                + (1.0 - quality_weight) * (1.0 - _normalized_task_cost(candidate, task)),
                -candidate.cost,
            ),
        )
        for task in tasks
    )
    return _summarize_routes(tasks, selected, quality_weight)


def _select_task_candidate(
    task: tuple[_TaskCandidate, ...], config: CapabilityRouterConfig, order: dict[str, int]
) -> _TaskCandidate:
    qualified: Final = tuple(
        candidate
        for candidate in task
        if candidate.probability
        > round(
            config.probability_threshold
            + BOUNDARY_THRESHOLD_STEPS[candidate.boundary] * config.threshold_step,
            9,
        )
    )
    fallback: Final = next(candidate for candidate in task if candidate.model == config.fallback_model)
    return min(qualified, key=lambda candidate: (candidate.cost, order[candidate.model])) if qualified else fallback


def _normalized_task_cost(selected: _TaskCandidate, task: tuple[_TaskCandidate, ...]) -> float:
    low: Final = min(candidate.cost for candidate in task)
    high: Final = max(candidate.cost for candidate in task)
    return 0.0 if high == low else (selected.cost - low) / (high - low)


def _probability_metrics(
    records: Sequence[CapabilityTrainingRecord],
    config: CapabilityRouterConfig,
    calibrated: bool,
) -> CapabilityProbabilityMetrics:
    candidates: Final = {candidate.model: candidate for candidate in config.candidates}
    rows: Final = tuple(record for record in records if record.split == "test" and record.model in candidates)
    predictions: Final = tuple(
        calibrated_probability(candidates[record.model], record.raw_p_solve) if calibrated else record.raw_p_solve
        for record in rows
    )
    outcomes: Final = tuple(record.success for record in rows)
    return CapabilityProbabilityMetrics(
        observations=len(rows),
        brier=sum((prediction - outcome) ** 2 for prediction, outcome in zip(predictions, outcomes)) / len(rows),
        log_loss=-sum(
            outcome * math.log(max(1e-9, prediction))
            + (1.0 - outcome) * math.log(max(1e-9, 1.0 - prediction))
            for prediction, outcome in zip(predictions, outcomes)
        )
        / len(rows),
        ece=sum(
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
        ),
    )


def train_capability_artifact(
    records: Sequence[CapabilityTrainingRecord],
    config: CapabilityRouterConfig,
    quality_weight: float = 0.7,
) -> CapabilityTrainingResult:
    if not 0.0 <= quality_weight <= 1.0:
        raise ValueError("quality_weight must be between 0 and 1")
    required_splits: Final = frozenset(record.split for record in records)
    if required_splits != frozenset(("train", "validation", "test")):
        raise ValueError("records must explicitly contain train, validation, and test splits")
    task_splits: Final = tuple(
        frozenset(record.split for record in task_records)
        for _, grouped in groupby(
            sorted(records, key=lambda record: (record.benchmark, record.task_id)),
            key=lambda record: (record.benchmark, record.task_id),
        )
        for task_records in (tuple(grouped),)
    )
    if any(len(splits) != 1 for splits in task_splits):
        raise ValueError("a benchmark task must not cross splits")
    configured_models: Final = frozenset(candidate.model for candidate in config.candidates)
    if frozenset(record.model for record in records) != configured_models:
        raise ValueError("record models must exactly match configured candidates")
    training: Final = tuple(record for record in records if record.split == "train")
    trained_rows: Final = tuple(
        _train_candidate(candidate, training, config.probability_threshold) for candidate in config.candidates
    )
    trained_candidates: Final = tuple(candidate for candidate, _ in trained_rows)
    rule_statistics: Final = tuple(statistic for _, statistics in trained_rows for statistic in statistics)
    calibrated_config: Final = config.model_copy(update={"candidates": trained_candidates})
    candidates: Final = tuple(
        calibrated_config.model_copy(update={"probability_threshold": threshold, "threshold_step": step})
        for threshold in _THRESHOLDS
        for step in _THRESHOLD_STEPS
        if threshold + 2.0 * step <= 1.0
    )
    scored: Final = tuple(
        (_route_metrics(records, "validation", candidate, quality_weight, True), candidate)
        for candidate in candidates
    )
    validation, trained_config = max(
        scored,
        key=lambda item: (
            item[0].quality_cost_utility,
            -item[0].mean_cost,
            item[0].success_rate,
        ),
    )
    datasets: Final = tuple(sorted(frozenset(record.benchmark for record in records)))
    artifact: Final = CapabilityTrainingArtifact(
        config=trained_config,
        rule_statistics=rule_statistics,
        datasets=datasets,
        records=len(records),
        split_counts={split: sum(record.split == split for record in records) for split in sorted(required_splits)},
        records_sha256=hashlib.sha256(
            "\n".join(
                record.model_dump_json()
                for record in sorted(
                    records,
                    key=lambda record: (
                        record.benchmark,
                        record.task_id,
                        record.split,
                        record.model,
                        record.primary_rule,
                        record.raw_p_solve,
                        record.success,
                        record.estimated_cost,
                    ),
                )
            ).encode()
        ).hexdigest(),
        split_contract="split is explicit; task_id must not cross splits",
    )
    return CapabilityTrainingResult(
        artifact=artifact,
        report=CapabilityTrainingReport(
            objective=(
                f"{quality_weight:g} * observed success + {1.0 - quality_weight:g} * normalized cost score"
            ),
            validation=validation,
            test=_route_metrics(records, "test", trained_config, quality_weight, True),
            test_untrained=_route_metrics(records, "test", config, quality_weight, False),
            test_always_candidates=_always_candidate_metrics(records, trained_config, quality_weight),
            test_oracle=_oracle_metrics(records, trained_config, quality_weight),
            test_threshold_sweep=tuple(
                CapabilityThresholdPoint(
                    probability_threshold=candidate.probability_threshold,
                    threshold_step=candidate.threshold_step,
                    metrics=_route_metrics(records, "test", candidate, quality_weight, True),
                )
                for candidate in candidates
            ),
            test_probability_raw=_probability_metrics(records, config, False),
            test_probability_calibrated=_probability_metrics(records, trained_config, True),
        ),
    )


def _read_records(path: Path) -> tuple[CapabilityTrainingRecord, ...]:
    with path.open() as record_file:
        return tuple(CapabilityTrainingRecord.model_validate_json(line) for line in record_file if line.strip())


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(description="Train and benchmark capability-router cards")
    parser.add_argument("records", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--quality-weight", type=float, default=0.7)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args: Final = _Arguments.model_validate(vars(_parser().parse_args(argv)))
    records: Final = _read_records(args.records)
    config: Final = CapabilityRouterConfig.model_validate(json.loads(args.config.read_text()))
    result: Final = train_capability_artifact(records, config, args.quality_weight)
    args.artifact_output.write_text(result.artifact.model_dump_json(indent=2) + "\n")
    print(result.report.model_dump_json(indent=2))  # noqa: T201  # CLI emits benchmark JSON
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
