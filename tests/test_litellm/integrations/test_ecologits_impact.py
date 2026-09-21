import sys
from dataclasses import dataclass, field
from types import ModuleType
from unittest.mock import patch

import pytest

from litellm.impact_calculator import ImpactRequest


@dataclass
class _Range:
    min: float
    max: float


@dataclass
class _Criterion:
    value: float | _Range
    unit: str


@dataclass
class _Error:
    message: str


@dataclass
class _Impacts:
    energy: _Criterion | None = None
    gwp: _Criterion | None = None
    adpe: _Criterion | None = None
    pe: _Criterion | None = None
    wcf: _Criterion | None = None
    errors: list = field(default_factory=list)


def _build_estimator(impacts):
    calls = []

    def llm_impacts(provider, model_name, output_token_count, request_latency, electricity_mix_zone=None):
        calls.append(
            {
                "provider": provider,
                "model_name": model_name,
                "output_token_count": output_token_count,
                "request_latency": request_latency,
            }
        )
        return impacts

    utils = ModuleType("ecologits.tracers.utils")
    utils.llm_impacts = llm_impacts
    modules = {
        "ecologits": ModuleType("ecologits"),
        "ecologits.tracers": ModuleType("ecologits.tracers"),
        "ecologits.tracers.utils": utils,
    }
    with patch.dict(sys.modules, modules):
        from litellm.integrations.ecologits_impact import EcoLogitsImpactEstimator

        return EcoLogitsImpactEstimator(), calls


def _request(**overrides) -> ImpactRequest:
    defaults = {
        "model": "gpt-4o",
        "custom_llm_provider": "openai",
        "completion_tokens": 250,
        "response_time": 3.5,
    }
    return ImpactRequest(**{**defaults, **overrides})


@pytest.fixture
def energy_only() -> _Impacts:
    return _Impacts(energy=_Criterion(value=1.0, unit="Wh"))


def test_a_modelled_interval_survives_the_conversion():
    """An interval is the estimator saying it does not know precisely, so it must not be flattened."""
    estimator, _ = _build_estimator(_Impacts(energy=_Criterion(value=_Range(min=2.0, max=6.0), unit="Wh")))

    energy = estimator.estimate(_request())["energy"]

    assert energy["min"] == 2.0
    assert energy["max"] == 6.0
    assert energy["min"] < energy["value"] < energy["max"]


def test_a_point_estimate_stays_a_point():
    estimator, _ = _build_estimator(_Impacts(energy=_Criterion(value=1.25, unit="Wh")))

    energy = estimator.estimate(_request())["energy"]

    assert energy["value"] == 1.25
    assert energy["min"] is None
    assert energy["max"] is None


def test_water_comes_from_the_same_total_as_the_other_figures():
    """Water read from a narrower phase than the rest would not match the boundary they all claim."""
    estimator, _ = _build_estimator(_Impacts(wcf=_Criterion(value=0.4, unit="L")))

    result = estimator.estimate(_request())

    assert result["water"]["value"] == 0.4
    assert result["water"]["unit"] == "L"


def test_the_boundary_and_estimator_are_recorded(energy_only):
    estimator, _ = _build_estimator(energy_only)

    result = estimator.estimate(_request())

    assert result["boundary"] == "B"
    assert result["estimator"] == "ecologits"


def test_a_result_carrying_errors_is_not_an_estimate():
    """ecologits reports an unknown model as an error on a filled-in result, not as a missing one."""
    estimator, _ = _build_estimator(_Impacts(errors=[_Error(message="Could not find model")]))

    assert estimator.estimate(_request()) is None


def test_generated_tokens_and_latency_drive_the_estimate(energy_only):
    estimator, calls = _build_estimator(energy_only)
    request = _request()

    estimator.estimate(request)

    assert calls[0]["output_token_count"] == request.completion_tokens
    assert calls[0]["request_latency"] == request.response_time


def test_the_routing_prefix_is_stripped_from_the_model_name(energy_only):
    """Under the router a model arrives provider-prefixed, which no estimator registry knows."""
    estimator, calls = _build_estimator(energy_only)

    estimator.estimate(_request(model="openai/gpt-4o"))

    assert calls[0]["model_name"] == "gpt-4o"


def test_provider_names_are_translated(energy_only):
    """litellm names a provider by the API it calls, the estimator by who trained the model."""
    estimator, calls = _build_estimator(energy_only)

    estimator.estimate(_request(custom_llm_provider="mistral"))

    assert calls[0]["provider"] == "mistralai"


def test_an_untranslatable_provider_yields_no_estimate(energy_only):
    estimator, calls = _build_estimator(energy_only)

    assert estimator.estimate(_request(custom_llm_provider="some_new_provider")) is None
    assert calls == []


def test_no_estimate_without_a_latency_measurement(energy_only):
    """Energy is modelled from how long generation ran, so an unmeasured request has no estimate."""
    estimator, calls = _build_estimator(energy_only)

    assert estimator.estimate(_request(response_time=None)) is None
    assert calls == []


def test_no_estimate_when_nothing_was_generated(energy_only):
    estimator, calls = _build_estimator(energy_only)

    assert estimator.estimate(_request(completion_tokens=0)) is None
    assert calls == []
