import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel

from litellm.router_strategy.adaptive_router.evaluation import (
    AdaptiveRouterEvaluationCase,
    EvaluationCompletion,
    EvaluationGrade,
    EvaluationMessage,
    evaluate_suite,
    load_evaluation_cases,
)
from litellm.router_strategy.adaptive_router.training import training_config
from litellm.types.router import RequestType


async def _fake_completion(
    model: str,
    messages: tuple[EvaluationMessage, ...],
    response_schema: type[BaseModel] | None,
) -> EvaluationCompletion:
    if response_schema is EvaluationGrade:
        quality: Final = 0.9 if "smart answer" in messages[-1].content else 0.4
        return EvaluationCompletion(content=json.dumps({"quality": quality}), cost=0.001)
    answer: Final = "smart answer" if model == "smart" else "fast answer"
    cost: Final = 0.02 if model == "smart" else 0.01
    return EvaluationCompletion(content=answer, cost=cost)


def test_load_evaluation_cases(tmp_path: Path) -> None:
    dataset: Final = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"case_id":"code-1","request_type":"code_generation","prompt":"Write a sort",'
        '"reference_answer":"sorted(xs)","grading_criteria":"Must handle duplicates"}\n',
        encoding="utf-8",
    )

    cases: Final = load_evaluation_cases(dataset)

    assert cases == (
        AdaptiveRouterEvaluationCase(
            case_id="code-1",
            request_type=RequestType.CODE_GENERATION,
            prompt="Write a sort",
            reference_answer="sorted(xs)",
            grading_criteria="Must handle duplicates",
        ),
    )


@pytest.mark.asyncio
async def test_evaluate_suite_runs_every_model_and_trains_priors() -> None:
    cases: Final = (
        AdaptiveRouterEvaluationCase(
            case_id="code-1",
            request_type=RequestType.CODE_GENERATION,
            prompt="Write a sort",
            reference_answer="sorted(xs)",
        ),
    )

    records: Final = await evaluate_suite(
        cases=cases,
        candidate_models=("fast", "smart"),
        judge_model="judge",
        completion=_fake_completion,
        concurrency=2,
    )
    config: Final = training_config(records).model_dump(mode="json")

    assert [(record.case_id, record.model, record.quality, record.cost) for record in records] == [
        ("code-1", "fast", 0.4, 0.01),
        ("code-1", "smart", 0.9, 0.02),
    ]
    assert all(record.latency_ms is not None and record.latency_ms >= 0 for record in records)
    assert config == {
        "evaluation_priors": [
            {
                "request_type": "code_generation",
                "model": "fast",
                "successes": 0.4,
                "failures": 0.6,
            },
            {
                "request_type": "code_generation",
                "model": "smart",
                "successes": 0.9,
                "failures": 1.0 - 0.9,
            },
        ],
        "exploration_rate": 0.05,
    }


@pytest.mark.asyncio
async def test_evaluate_suite_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        await evaluate_suite((), (), "judge", completion=_fake_completion, concurrency=0)
