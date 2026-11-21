from types import MappingProxyType
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.model_recommender.benchmarks import BenchmarkScore
from litellm.proxy.model_recommender.models import Candidate, Priorities, RecommendationRequest, Requirements, Speed
from litellm.proxy.model_recommender.scoring import blended_cost, cost_signal, matches, normalize, rank


def test_cost_weight_ranks_cheapest_and_reports_unknown_neutrally() -> None:
    cheap: Final = Candidate(
        model_group="cheap", mode="chat", input_cost_per_token=0.000001, output_cost_per_token=0.000005
    )
    expensive: Final = cheap.model_copy(update={"model_group": "expensive", "input_cost_per_token": 0.000009})
    unknown: Final = Candidate(model_group="unknown", mode="chat")
    result: Final = rank(
        (expensive, unknown, cheap),
        RecommendationRequest(priorities=Priorities(intelligence=0, speed=0, cost=3)),
        (),
        MappingProxyType({}),
    )
    assert tuple((r.model_group, r.score) for r in result) == (("cheap", 100), ("unknown", 50), ("expensive", 0))
    assert blended_cost(cheap) == pytest.approx(2)
    assert result[1].signals.cost.unknown is True
    assert result[1].signals.cost.raw is None


@pytest.mark.parametrize(
    "requirements",
    [
        Requirements(reasoning=True),
        Requirements(vision=True),
        Requirements(function_calling=True),
        Requirements(min_context_tokens=100),
    ],
)
def test_requirements_exclude_unsupported_and_unknown_capabilities(requirements: Requirements) -> None:
    assert not matches(Candidate(model_group="unknown", mode="chat"), requirements)
    assert matches(
        Candidate(
            model_group="capable",
            mode="chat",
            supports_reasoning=True,
            supports_vision=True,
            supports_function_calling=True,
            max_input_tokens=100,
        ),
        requirements,
    )


def test_non_chat_models_are_excluded() -> None:
    assert not matches(Candidate(model_group="embedding", mode="embedding"), Requirements())


def test_speed_inverts_latency_and_keeps_throughput() -> None:
    result: Final = rank(
        (Candidate(model_group="slow"), Candidate(model_group="fast")),
        RecommendationRequest(priorities=Priorities(intelligence=0, cost=0, speed=3)),
        (),
        {
            "slow": Speed(model_group="slow", latency_seconds=10),
            "fast": Speed(model_group="fast", latency_seconds=1, output_tokens_per_second=42),
        },
    )
    assert tuple(r.model_group for r in result) == ("fast", "slow")
    assert result[0].score == 100
    assert result[0].output_tokens_per_second == 42


def test_empty_zero_weights_ties_and_limit() -> None:
    request: Final = RecommendationRequest(priorities=Priorities(intelligence=0, speed=0, cost=0), limit=1)
    assert rank((), request, (), {}) == ()
    result: Final = rank((Candidate(model_group="b"), Candidate(model_group="a")), request, (), {})
    assert tuple((r.model_group, r.score) for r in result) == (("a", 50),)
    assert normalize(2, (2, 2)).normalized == 0.5


def test_request_rejects_invalid_priority_and_limit() -> None:
    with pytest.raises(ValidationError):
        Priorities(cost=4)
    with pytest.raises(ValidationError):
        RecommendationRequest(limit=0)


def test_intelligence_priority_outweighs_cost_with_exact_underlying_model_match() -> None:
    capable: Final = Candidate(
        model_group="capable",
        underlying_models=("provider/example",),
        input_cost_per_token=0.01,
        output_cost_per_token=0.01,
    )
    economical: Final = Candidate(
        model_group="economical", underlying_models=("provider/basic",), input_cost_per_token=0, output_cost_per_token=0
    )
    result: Final = rank(
        (economical, capable),
        RecommendationRequest(priorities=Priorities(intelligence=3, speed=0, cost=1)),
        (
            BenchmarkScore(model="example", focus="general", score=0.8),
            BenchmarkScore(model="basic", focus="general", score=0.2),
        ),
        {},
    )
    assert tuple((row.model_group, row.score) for row in result) == (("capable", 60), ("economical", 40))
    assert result[0].signals.intelligence.raw == 0.8
    assert result[0].signals.intelligence.normalized == 0.8
    assert result[1].signals.intelligence.normalized == 0.2


def test_intelligence_is_absolute_even_for_one_candidate() -> None:
    result: Final = rank(
        (Candidate(model_group="only", underlying_models=("example",)),),
        RecommendationRequest(priorities=Priorities(intelligence=3, speed=0, cost=0)),
        (BenchmarkScore(model="example", focus="general", score=0.9),),
        {},
    )
    assert result[0].score == 90
    assert result[0].signals.intelligence.normalized == 0.9


def test_cost_uses_logarithmic_distance_and_preserves_raw_prices() -> None:
    signal: Final = cost_signal(10, (1, 10, 100, None))
    assert signal.normalized == pytest.approx(0.5)
    assert signal.raw == 10
    assert signal.unknown is False
    assert cost_signal(0, (0, 1, 100)).normalized == 1
    assert cost_signal(0, (0, 0)).normalized == 0.5
    assert cost_signal(None, (1, 100)).unknown is True
    assert cost_signal(None, (None,)).normalized == 0.5

    candidates: Final = tuple(
        Candidate(model_group=str(price), input_cost_per_token=price / 1_000_000, output_cost_per_token=price / 1_000_000)
        for price in (1, 10, 100)
    )
    result: Final = rank(
        candidates, RecommendationRequest(priorities=Priorities(intelligence=0, speed=0, cost=3)), (), {}
    )
    assert result[1].model_group == "10"
    assert result[1].signals.cost.normalized == pytest.approx(0.5)
    assert result[1].score == 50
