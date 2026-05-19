"""Tests for capability-dimension Prometheus counters (S6-02)."""

import pytest

prometheus_client = pytest.importorskip("prometheus_client")


@pytest.fixture(autouse=True)
def _reset_metrics():
    from litellm.integrations.prometheus_helpers import capability_metrics

    capability_metrics._reset_for_tests()
    yield
    capability_metrics._reset_for_tests()


def _get_counter_value(counter, **labels) -> float:
    sample_value = counter.labels(**labels)._value.get()
    return float(sample_value)


def test_record_increments_request_and_spend_counters():
    from litellm.integrations.prometheus_helpers import capability_metrics

    capability_metrics.record_capability_call(
        app_id="xct-chat",
        entity_type="agent",
        entity_id="agt-1",
        spend=0.01,
    )
    capability_metrics.record_capability_call(
        app_id="xct-chat",
        entity_type="agent",
        entity_id="agt-1",
        spend=0.02,
    )

    requests = _get_counter_value(
        capability_metrics._requests_counter,
        app_id="xct-chat",
        entity_type="agent",
        entity_id="agt-1",
    )
    spend = _get_counter_value(
        capability_metrics._spend_counter,
        app_id="xct-chat",
        entity_type="agent",
        entity_id="agt-1",
    )
    assert requests == 2.0
    assert spend == pytest.approx(0.03)


def test_record_missing_entity_type_is_noop():
    """No entity_type -> nothing attributed (the row was a non-capability call)."""
    from litellm.integrations.prometheus_helpers import capability_metrics

    capability_metrics.record_capability_call(
        app_id="xct-chat",
        entity_type=None,
        entity_id="anything",
        spend=0.05,
    )
    # Counters were never created (lazy init waits on first eligible call).
    assert capability_metrics._requests_counter is None
    assert capability_metrics._spend_counter is None


def test_record_separate_apps_get_separate_series():
    from litellm.integrations.prometheus_helpers import capability_metrics

    capability_metrics.record_capability_call(
        app_id="xct-chat", entity_type="model", entity_id="gpt-4o", spend=0.01
    )
    capability_metrics.record_capability_call(
        app_id="xct-home", entity_type="model", entity_id="gpt-4o", spend=0.02
    )
    chat = _get_counter_value(
        capability_metrics._requests_counter,
        app_id="xct-chat",
        entity_type="model",
        entity_id="gpt-4o",
    )
    home = _get_counter_value(
        capability_metrics._requests_counter,
        app_id="xct-home",
        entity_type="model",
        entity_id="gpt-4o",
    )
    assert chat == 1.0
    assert home == 1.0


def test_record_app_id_none_collapses_to_literal_none_label():
    """Pre-S4 traffic has no app_id; bucket it under "none" so cardinality stays bounded."""
    from litellm.integrations.prometheus_helpers import capability_metrics

    capability_metrics.record_capability_call(
        app_id=None, entity_type="skill", entity_id="fact-check", spend=0
    )
    value = _get_counter_value(
        capability_metrics._requests_counter,
        app_id="none",
        entity_type="skill",
        entity_id="fact-check",
    )
    assert value == 1.0


def test_record_zero_spend_does_not_inc_spend_counter():
    from litellm.integrations.prometheus_helpers import capability_metrics

    capability_metrics.record_capability_call(
        app_id="xct-chat", entity_type="model", entity_id="gpt-4o", spend=0
    )
    # Requests counter incremented; spend counter NOT touched (no `.inc(0)`
    # call that would still create the child series with value 0).
    requests = _get_counter_value(
        capability_metrics._requests_counter,
        app_id="xct-chat",
        entity_type="model",
        entity_id="gpt-4o",
    )
    assert requests == 1.0
