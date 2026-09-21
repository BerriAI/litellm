import pytest

import litellm
from litellm.impact_calculator import (
    ImpactRequest,
    calculate_impact,
    current_estimator_name,
    get_impact_estimator,
    interval_impact_value,
    point_impact_value,
    register_impact_estimator,
    reset_default_estimator_cache,
)
from litellm.types.utils import ImpactInformation


@pytest.fixture
def impact_request() -> ImpactRequest:
    return ImpactRequest(
        model="gpt-4o",
        custom_llm_provider="openai",
        completion_tokens=100,
        response_time=2.0,
    )


@pytest.fixture(autouse=True)
def clear_registration():
    yield
    register_impact_estimator(None)
    reset_default_estimator_cache()


class _StubEstimator:
    name = "stub"

    def __init__(self) -> None:
        self.impact = ImpactInformation(estimator="stub")
        self.calls = 0

    def estimate(self, request: ImpactRequest) -> ImpactInformation | None:
        self.calls += 1
        return self.impact


class _RaisingEstimator:
    name = "raising"

    def estimate(self, request: ImpactRequest) -> ImpactInformation | None:
        raise ValueError("no model data")


def test_tracking_off_never_calls_the_estimator(monkeypatch, impact_request):
    """The flag gates estimation itself, so an opted-out deployment pays nothing for it."""
    estimator = _StubEstimator()
    register_impact_estimator(estimator)
    monkeypatch.setattr(litellm, "track_impact", False)

    assert calculate_impact(impact_request) is None
    assert estimator.calls == 0


def test_tracking_on_returns_the_registered_estimators_result(monkeypatch, impact_request):
    estimator = _StubEstimator()
    register_impact_estimator(estimator)
    monkeypatch.setattr(litellm, "track_impact", True)

    result = calculate_impact(impact_request)

    assert estimator.calls == 1
    assert result is estimator.impact


def test_registration_takes_precedence_over_default_resolution():
    estimator = _StubEstimator()
    register_impact_estimator(estimator)

    assert get_impact_estimator() is estimator
    assert current_estimator_name() == "stub"


def test_clearing_registration_stops_routing_to_the_cleared_estimator(monkeypatch, impact_request):
    estimator = _StubEstimator()
    register_impact_estimator(estimator)
    monkeypatch.setattr(litellm, "track_impact", True)

    register_impact_estimator(None)
    calculate_impact(impact_request)

    assert estimator.calls == 0


def test_estimator_failure_reaches_the_caller(monkeypatch, impact_request):
    """calculate_impact does not swallow, so the logging layer can record the failure."""
    register_impact_estimator(_RaisingEstimator())
    monkeypatch.setattr(litellm, "track_impact", True)

    with pytest.raises(ValueError, match="no model data"):
        calculate_impact(impact_request)


def test_a_point_value_carries_no_interval():
    value = point_impact_value(unit="Wh", value=1.5)

    assert value["value"] == 1.5
    assert value["min"] is None
    assert value["max"] is None


def test_an_interval_value_reports_its_midpoint():
    """A consumer summing `value` must read a figure inside the modelled interval."""
    value = interval_impact_value(unit="Wh", minimum=2.0, maximum=6.0)

    assert value["min"] < value["value"] < value["max"]
    assert value["value"] == 4.0
