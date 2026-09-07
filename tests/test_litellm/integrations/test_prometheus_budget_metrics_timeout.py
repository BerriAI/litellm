"""
Unit tests for the per-request budget-metric emission timeout in
PrometheusLogger._increment_remaining_budget_metrics.

A slow Redis/DB lookup in one of the budget branches must not let the gather run
unbounded; it is wrapped in asyncio.wait_for so the success-logging coroutine
cannot exceed the LoggingWorker watchdog and get the whole event cancelled.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import (
    PrometheusLogger,
    _DEFAULT_BUDGET_METRICS_PER_REQUEST_TIMEOUT,
    _get_budget_metrics_per_request_timeout,
)

TIMEOUT_ENV = "PROMETHEUS_BUDGET_METRICS_PER_REQUEST_TIMEOUT"


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def prometheus_logger():
    return PrometheusLogger()


def _call_increment(logger: PrometheusLogger):
    return logger._increment_remaining_budget_metrics(
        user_api_team="team-1",
        user_api_team_alias="team-alias",
        user_api_key="key-1",
        user_api_key_alias="key-alias",
        litellm_params={"metadata": {}},
        response_cost=0.01,
        user_id="user-1",
        user_api_key_org_id="org-1",
    )


def _skip_logged(debug_mock) -> bool:
    return any("skipping" in str(call.args[0]) for call in debug_mock.call_args_list if call.args)


@pytest.mark.asyncio
async def test_budget_metric_emission_skips_on_timeout(prometheus_logger, monkeypatch):
    """A branch slower than the timeout is skipped without propagating, and the
    skip is logged instead of cancelling the success-logging event."""
    monkeypatch.setenv(TIMEOUT_ENV, "0.05")

    async def _slow_branch(**kwargs):
        await asyncio.sleep(30)

    prometheus_logger._set_api_key_budget_metrics_after_api_request = _slow_branch
    prometheus_logger._set_team_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_user_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_org_budget_metrics_after_api_request = AsyncMock()

    with patch("litellm.integrations.prometheus.verbose_logger") as mock_logger:
        await _call_increment(prometheus_logger)

    assert _skip_logged(mock_logger.debug)


@pytest.mark.asyncio
async def test_budget_metric_emission_completes_within_timeout(prometheus_logger, monkeypatch):
    """With a generous timeout every branch is awaited and no skip is logged."""
    monkeypatch.setenv(TIMEOUT_ENV, "5.0")

    prometheus_logger._set_api_key_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_team_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_user_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_org_budget_metrics_after_api_request = AsyncMock()

    with patch("litellm.integrations.prometheus.verbose_logger") as mock_logger:
        await _call_increment(prometheus_logger)

    assert prometheus_logger._set_api_key_budget_metrics_after_api_request.await_count == 1
    assert prometheus_logger._set_team_budget_metrics_after_api_request.await_count == 1
    assert prometheus_logger._set_user_budget_metrics_after_api_request.await_count == 1
    assert prometheus_logger._set_org_budget_metrics_after_api_request.await_count == 1
    assert not _skip_logged(mock_logger.debug)


@pytest.mark.asyncio
async def test_invalid_timeout_env_falls_back_to_default(prometheus_logger, monkeypatch):
    """A malformed timeout env value must not raise (which would recreate the
    failure mode); it falls back to the default and every branch still runs."""
    monkeypatch.setenv(TIMEOUT_ENV, "not-a-number")

    prometheus_logger._set_api_key_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_team_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_user_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_org_budget_metrics_after_api_request = AsyncMock()

    await _call_increment(prometheus_logger)

    assert prometheus_logger._set_api_key_budget_metrics_after_api_request.await_count == 1
    assert prometheus_logger._set_org_budget_metrics_after_api_request.await_count == 1


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1", "nan", "inf", "-inf"])
def test_unusable_timeout_env_falls_back_to_default(value, monkeypatch):
    """Values that parse but disable or unbound the timeout (0, negative, nan,
    inf) must fall back to the default instead of being used; otherwise they
    either skip every emission or recreate the unbounded-wait failure mode."""
    monkeypatch.setenv(TIMEOUT_ENV, value)

    assert _get_budget_metrics_per_request_timeout() == _DEFAULT_BUDGET_METRICS_PER_REQUEST_TIMEOUT


@pytest.mark.parametrize("value,expected", [("0.05", 0.05), ("5.0", 5.0), ("30", 30.0)])
def test_valid_timeout_env_is_used(value, expected, monkeypatch):
    """A finite positive value is parsed and returned unchanged."""
    monkeypatch.setenv(TIMEOUT_ENV, value)

    assert _get_budget_metrics_per_request_timeout() == expected


def test_missing_timeout_env_uses_default(monkeypatch):
    """With the env unset the default is returned."""
    monkeypatch.delenv(TIMEOUT_ENV, raising=False)

    assert _get_budget_metrics_per_request_timeout() == _DEFAULT_BUDGET_METRICS_PER_REQUEST_TIMEOUT


@pytest.mark.asyncio
async def test_outer_cancellation_still_propagates(prometheus_logger, monkeypatch):
    """Only asyncio.TimeoutError is swallowed; an outer cancellation (cooperative
    shutdown / watchdog) injected while awaiting must still propagate."""
    monkeypatch.setenv(TIMEOUT_ENV, "30")

    started = asyncio.Event()

    async def _slow_branch(**kwargs):
        started.set()
        await asyncio.sleep(30)

    prometheus_logger._set_api_key_budget_metrics_after_api_request = _slow_branch
    prometheus_logger._set_team_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_user_budget_metrics_after_api_request = AsyncMock()
    prometheus_logger._set_org_budget_metrics_after_api_request = AsyncMock()

    task = asyncio.create_task(_call_increment(prometheus_logger))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
