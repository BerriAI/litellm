from typing import Final

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.rust_bridge.host import redis_events


def _sample(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    redis_events._service_logging.cache_clear()
    yield
    redis_events._service_logging.cache_clear()


@pytest.mark.parametrize(
    ("failure", "failure_class"),
    [("timeout", "timeout"), ("transport", "connectivity")],
)
def test_a_health_failure_counts_as_a_failed_request_and_a_breaker_failure(
    prometheus: None, failure: redis_events.Failure, failure_class: str
):
    failed: Final = (
        "litellm_redis_failed_requests_total",
        {"redis": "redis", "error_class": "RedisCallError", "function_name": "async_increment"},
    )
    breaker: Final = ("litellm_redis_circuit_breaker_failures_total", {"failure_class": failure_class})
    before: Final = (_sample(*failed), _sample(*breaker))

    redis_events.record_redis_call("async_increment", 0.25, failure, "caller.py")

    assert (_sample(*failed), _sample(*breaker)) == (before[0] + 1, before[1] + 1)


@pytest.mark.parametrize("failure", ["server", "decode"])
def test_a_rejected_command_is_a_failed_request_but_not_a_breaker_failure(
    prometheus: None, failure: redis_events.Failure
):
    breaker: Final = {
        failure_class: _sample("litellm_redis_circuit_breaker_failures_total", {"failure_class": failure_class})
        for failure_class in ("timeout", "connectivity")
    }
    failed: Final = (
        "litellm_redis_failed_requests_total",
        {"redis": "redis", "error_class": "RedisCallError", "function_name": "async_increment"},
    )
    before: Final = _sample(*failed)

    redis_events.record_redis_call("async_increment", 0.25, failure, None)

    assert _sample(*failed) == before + 1
    assert {
        failure_class: _sample("litellm_redis_circuit_breaker_failures_total", {"failure_class": failure_class})
        for failure_class in ("timeout", "connectivity")
    } == breaker


def test_a_success_is_counted_and_a_breaker_rejection_is_not(prometheus: None):
    total: Final = ("litellm_redis_total_requests_total", {"redis": "redis"})
    before: Final = _sample(*total)

    redis_events.record_redis_call("async_get_cache", 0.01, None, None)
    redis_events.record_redis_call("async_get_cache", 0.0, "breaker_open", None)

    assert _sample(*total) == before + 1


@pytest.mark.parametrize("state", ["open", "half_open", "closed"])
def test_breaker_transitions_are_counted_by_state(prometheus: None, state: redis_events.BreakerState):
    labels: Final = {"state": state}
    before: Final = _sample("litellm_redis_circuit_breaker_transitions_total", labels)

    redis_events.record_redis_breaker(state)

    assert _sample("litellm_redis_circuit_breaker_transitions_total", labels) == before + 1
