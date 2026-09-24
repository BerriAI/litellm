"""Jev route-down gate: a below-top verdict is applied only when every configured gate passes.

The gate is fail-expensive: a refused route-down keeps the request on the configured
expensive fallback (here classifier_fallback='default_model' so the destination is
deterministic), the Jev verdict and its cost stay on the outcome for provenance, and a
refusal counts as a classifier success for the circuit breaker. All tests mock the
classifier client; no TYPESAFE_API_KEY or network is involved.
"""

from collections.abc import Mapping
from typing import Any, Final
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import ComplexityTier, JevClassifierConfig
from litellm.router_strategy.complexity_router.jev_classifier import (
    JevChoiceAnswer,
    JevSystemOneRequest,
    JevSystemOneResponse,
    JevUsage,
)

_TIERS: Final[Mapping[str, str]] = {
    "SIMPLE": "cheap",
    "MEDIUM": "mid",
    "COMPLEX": "strong",
    "REASONING": "fallback-strong",
}


def _tier_answer(choice: str, confidence: float) -> JevChoiceAnswer:
    return JevChoiceAnswer(
        type="choice",
        choice=choice,
        probabilities={choice: confidence},
        confidence=confidence,
    )


def _response(choice: str, confidence: float) -> JevSystemOneResponse:
    return JevSystemOneResponse(
        model="jev-1.13.0",
        answers={"tier": _tier_answer(choice, confidence)},
        usage=JevUsage(input_tokens=3, output_tokens=4),
    )


class _StaticJevClient:
    def __init__(self, response: JevSystemOneResponse) -> None:
        self.response = response
        self.calls = 0

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        self.calls += 1
        return self.response


class _WireJevClient:
    """Validates a raw wire payload the way HttpJevClassifierClient does."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = payload

    async def evaluate(
        self,
        request: JevSystemOneRequest,
        timeout_s: float,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> JevSystemOneResponse:
        return TypeAdapter(JevSystemOneResponse).validate_python(self.payload)


def _router(client: Any, jev_overrides: Mapping[str, object] | None = None) -> ComplexityRouter:
    jev: Final[dict[str, object]] = {"api_key": "test", "timeout_ms": 100, **(dict(jev_overrides or {}))}
    return ComplexityRouter(
        "test-router",
        MagicMock(),
        {
            "classifier_type": "jev",
            "jev_classifier_config": jev,
            "classifier_fallback": "default_model",
            "default_model": "fallback-strong",
            "tiers": dict(_TIERS),
        },
        derive_savings_baseline=False,
        jev_client=client,
    )


@pytest.mark.asyncio
async def test_below_threshold_refuses_route_down_and_fails_over_to_the_expensive_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "typesafe/jev-1.13.0",
        {"input_cost_per_token": 0.0001, "output_cost_per_token": 0.0002},
    )
    router = _router(_StaticJevClient(_response("MEDIUM", 0.42)), {"confidence_threshold": 0.7})

    outcome = await router.aclassify("Explain this")

    assert outcome.tier == ComplexityTier.REASONING  # the tier whose pool holds default_model
    assert outcome.cause == "default_model_fallback"
    assert outcome.signals == (
        "classifier-failed:default-model",
        "jev-refused:low-confidence",
        "jev-classifier:MEDIUM",
    )
    assert outcome.jev_verdict is not None
    assert outcome.jev_verdict.confidence == 0.42
    assert outcome.classifier_cost == pytest.approx(0.0011)


@pytest.mark.asyncio
async def test_above_threshold_applies_the_route_down() -> None:
    router = _router(_StaticJevClient(_response("MEDIUM", 0.8)), {"confidence_threshold": 0.7})

    outcome = await router.aclassify("Explain this")

    assert outcome.tier == ComplexityTier.MEDIUM
    assert outcome.cause == "jev_classifier"
    assert outcome.signals == (
        "jev-classifier:MEDIUM",
        "jev-confidence=0.800000",
        "tier-probability:MEDIUM=0.800000",
    )


@pytest.mark.asyncio
async def test_confidence_exactly_at_threshold_applies_the_route_down() -> None:
    router = _router(_StaticJevClient(_response("MEDIUM", 0.7)), {"confidence_threshold": 0.7})

    outcome = await router.aclassify("Explain this")

    assert outcome.tier == ComplexityTier.MEDIUM
    assert outcome.cause == "jev_classifier"


@pytest.mark.asyncio
async def test_gate_disabled_keeps_current_behavior_of_applying_every_verdict() -> None:
    router = _router(_StaticJevClient(_response("MEDIUM", 0.05)))

    outcome = await router.aclassify("Explain this")

    assert outcome.tier == ComplexityTier.MEDIUM
    assert outcome.cause == "jev_classifier"


@pytest.mark.asyncio
async def test_verdict_naming_the_strongest_tier_is_never_gated() -> None:
    router = _router(_StaticJevClient(_response("REASONING", 0.1)), {"confidence_threshold": 0.99})

    outcome = await router.aclassify("Explain this")

    assert outcome.tier == ComplexityTier.REASONING
    assert outcome.cause == "jev_classifier"


@pytest.mark.asyncio
async def test_wire_answer_without_confidence_falls_back_conservatively() -> None:
    client = _WireJevClient(
        {
            "model": "jev-1.13.0",
            "answers": {"tier": {"type": "choice", "choice": "MEDIUM", "probabilities": {"MEDIUM": 0.9}}},
        }
    )
    router = _router(client, {"confidence_threshold": 0.7})

    outcome = await router.aclassify("Explain this")

    assert outcome.tier == ComplexityTier.REASONING
    assert outcome.cause == "default_model_fallback"
    assert outcome.jev_verdict is None


@pytest.mark.asyncio
async def test_gate_refusal_is_a_classifier_success_for_the_circuit_breaker() -> None:
    client = _StaticJevClient(_response("MEDIUM", 0.1))
    router = _router(client, {"confidence_threshold": 0.7})

    first = await router.aclassify("Explain this")
    second = await router.aclassify("Explain this")

    assert client.calls == 2  # the breaker stayed closed: refusals are policy, not health
    assert first.cause == second.cause == "default_model_fallback"


@pytest.mark.asyncio
async def test_complexity_max_refuses_a_route_down_when_no_complexity_answer_exists() -> None:
    router = _router(_StaticJevClient(_response("MEDIUM", 0.99)), {"complexity_max": 0.5})

    outcome = await router.aclassify("Explain this")

    assert outcome.cause == "default_model_fallback"
    assert "jev-refused:high-complexity" in outcome.signals


@pytest.mark.asyncio
async def test_complexity_max_of_one_keeps_route_down_available_without_a_complexity_answer() -> None:
    router = _router(_StaticJevClient(_response("MEDIUM", 0.5)), {"complexity_max": 1.0})

    outcome = await router.aclassify("Explain this")

    assert outcome.tier == ComplexityTier.MEDIUM
    assert outcome.cause == "jev_classifier"


@pytest.mark.asyncio
async def test_complexity_confidence_min_refuses_a_route_down_when_no_complexity_answer_exists() -> None:
    router = _router(_StaticJevClient(_response("MEDIUM", 0.99)), {"complexity_confidence_min": 0.5})

    outcome = await router.aclassify("Explain this")

    assert outcome.cause == "default_model_fallback"
    assert "jev-refused:low-complexity-confidence" in outcome.signals


@pytest.mark.asyncio
async def test_complexity_confidence_min_of_zero_keeps_route_down_available() -> None:
    router = _router(_StaticJevClient(_response("MEDIUM", 0.5)), {"complexity_confidence_min": 0.0})

    outcome = await router.aclassify("Explain this")

    assert outcome.tier == ComplexityTier.MEDIUM
    assert outcome.cause == "jev_classifier"


@pytest.mark.asyncio
async def test_all_three_gates_configured_refuse_with_the_first_failure_reported() -> None:
    router = _router(
        _StaticJevClient(_response("MEDIUM", 0.1)),
        {"confidence_threshold": 0.7, "complexity_max": 0.5, "complexity_confidence_min": 0.5},
    )

    outcome = await router.aclassify("Explain this")

    assert outcome.cause == "default_model_fallback"
    assert "jev-refused:low-confidence" in outcome.signals


def test_gate_refusal_reasons_cover_every_gate_independently() -> None:
    config = JevClassifierConfig(
        api_key="test", confidence_threshold=0.7, complexity_max=0.5, complexity_confidence_min=0.5
    )
    reason = ComplexityRouter._jev_route_down_refusal_reason

    # Not a route-down: the strongest tier is never gated, whatever the confidence.
    assert reason(config, "REASONING", "REASONING", 0.0) is None
    # Confidence gate: closed below the threshold, open at it (complexity gates follow).
    assert reason(config, "MEDIUM", "REASONING", 0.69) == "low-confidence"
    assert reason(config, "MEDIUM", "REASONING", 0.7) == "high-complexity"  # missing score is 1.0 > 0.5
    # Complexity gates pass when the wire carries a qualifying answer.
    assert reason(config, "MEDIUM", "REASONING", 0.7, complexity_score=0.3, complexity_confidence=0.6) is None
    # Missing complexity data is hard: unknown score 1.0 refuses, unknown confidence 0.0 refuses.
    assert reason(config, "MEDIUM", "REASONING", 0.9, complexity_score=0.6) == "high-complexity"
    assert reason(config, "MEDIUM", "REASONING", 0.9, complexity_score=0.3) == "low-complexity-confidence"


def test_threshold_fields_default_to_off_and_reject_out_of_range_values() -> None:
    defaults = JevClassifierConfig(api_key="test")
    assert defaults.confidence_threshold is None
    assert defaults.complexity_max is None
    assert defaults.complexity_confidence_min is None

    with pytest.raises(ValidationError):
        JevClassifierConfig(api_key="test", confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        JevClassifierConfig(api_key="test", complexity_max=-0.1)
    with pytest.raises(ValidationError):
        JevClassifierConfig(api_key="test", complexity_confidence_min=1.5)
