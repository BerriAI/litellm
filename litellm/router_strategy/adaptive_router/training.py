from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from itertools import groupby
from pathlib import Path
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.router import (
    AdaptiveRouterEvaluationPrior,
    RequestType,
)

EvaluationKey: TypeAlias = tuple[RequestType, str]


class AdaptiveRouterEvaluationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str | None = None
    request_type: RequestType
    model: str
    quality: float = Field(ge=0.0, le=1.0)
    cost: float | None = Field(default=None, ge=0.0)
    latency_ms: float | None = Field(default=None, ge=0.0)


class AdaptiveRouterTrainingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluation_priors: tuple[AdaptiveRouterEvaluationPrior, ...]
    exploration_rate: float = 0.05


def _aggregate_group(
    key: EvaluationKey,
    records: Iterator[AdaptiveRouterEvaluationRecord],
) -> AdaptiveRouterEvaluationPrior:
    request_type, model = key
    grouped_records: Final = tuple(records)
    return AdaptiveRouterEvaluationPrior(
        request_type=request_type,
        model=model,
        successes=sum(record.quality for record in grouped_records),
        failures=sum(1.0 - record.quality for record in grouped_records),
    )


def aggregate_evaluation_records(
    records: Iterable[AdaptiveRouterEvaluationRecord],
) -> tuple[AdaptiveRouterEvaluationPrior, ...]:
    ordered: Final = tuple(sorted(records, key=lambda record: (record.request_type.value, record.model)))
    grouped: Final = groupby(ordered, key=lambda record: (record.request_type, record.model))
    return tuple(_aggregate_group(key, records_for_key) for key, records_for_key in grouped)


def load_evaluation_records(path: Path) -> tuple[AdaptiveRouterEvaluationRecord, ...]:
    with path.open(encoding="utf-8") as input_file:
        return tuple(AdaptiveRouterEvaluationRecord.model_validate_json(line) for line in input_file if line.strip())


def training_config_fragment(path: Path) -> AdaptiveRouterTrainingResult:
    return training_config(load_evaluation_records(path))


def training_config(records: Iterable[AdaptiveRouterEvaluationRecord]) -> AdaptiveRouterTrainingResult:
    priors: Final = aggregate_evaluation_records(records)
    return AdaptiveRouterTrainingResult(evaluation_priors=priors)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m litellm.router_strategy.adaptive_router.training EVALUATIONS.jsonl")
    evaluation_records: Final = Path(sys.argv[1])
    result: Final = training_config_fragment(evaluation_records)
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
