"""
Tests for health-check reachability transition logging and the no-log flag on
internal health probes (issue #34281): an offline deployment should produce a
single log line per state change instead of a full stack trace per poll cycle,
and health probes should not be routed through user logging callbacks.
"""

import logging
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers
from litellm.proxy import health_check as hc


@pytest.fixture(autouse=True)
def _reset_reachability_state():
    """Isolate the module-level reachability state between tests."""
    hc._deployment_reachability_state.clear()
    if hasattr(litellm, "health_check_unreachable_relog_seconds"):
        delattr(litellm, "health_check_unreachable_relog_seconds")
    yield
    hc._deployment_reachability_state.clear()


# ---------------------------------------------------------------------------
# no-log flag on internal health probes
# ---------------------------------------------------------------------------


def test_health_check_tracking_sets_no_log():
    """Health probes must be marked no-log so failures skip user logging
    callbacks (proxy cost/DB callbacks still run)."""
    updated = HealthCheckHelpers._update_model_params_with_health_check_tracking_information(
        model_params={"model": "gpt-4", "api_base": "http://localhost:1234"}
    )
    assert updated["no-log"] is True


# ---------------------------------------------------------------------------
# transport-error classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (ConnectionRefusedError("[Errno 111] Connection refused"), True),
        (httpx.ConnectError("Cannot connect to host"), True),
        (httpx.ConnectTimeout("timed out"), True),
        (TimeoutError(), True),
        (OSError("Network is unreachable"), True),
        (Exception("AuthenticationError: invalid api key - status 401"), False),
        (Exception("BadRequestError: unsupported parameter"), False),
        (None, False),
    ],
)
def test_is_transport_error_classification(exc, expected):
    assert hc._is_transport_error(exc) is expected


# ---------------------------------------------------------------------------
# transition logging: down -> quiet -> up
# ---------------------------------------------------------------------------


def _unhealthy(model_id="d1"):
    return [{"model_id": model_id, "model": "qwen3", "api_base": "http://ollama:8443"}]


def _healthy(model_id="d1"):
    return [{"model_id": model_id, "model": "qwen3", "api_base": "http://ollama:8443"}]


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def _infos(caplog):
    return [r for r in caplog.records if r.levelno == logging.INFO]


def test_first_failure_logs_single_warning(caplog):
    caplog.set_level(logging.INFO, logger="litellm.proxy.health_check")
    exc = ConnectionRefusedError("[Errno 111] Connection refused")

    hc._log_deployment_health_transitions(
        healthy_endpoints=[],
        unhealthy_endpoints=_unhealthy(),
        exceptions_by_model_id={"d1": exc},
    )

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "unreachable" in msg
    assert "qwen3" in msg and "http://ollama:8443" in msg
    assert hc._deployment_reachability_state["d1"]["reachable"] is False


def test_still_unhealthy_does_not_relog(caplog):
    exc = ConnectionRefusedError("[Errno 111] Connection refused")
    # cycle 1: transition down
    hc._log_deployment_health_transitions([], _unhealthy(), {"d1": exc})

    # cycle 2: still down -> must be silent with default relog (0)
    caplog.clear()
    caplog.set_level(logging.INFO, logger="litellm.proxy.health_check")
    hc._log_deployment_health_transitions([], _unhealthy(), {"d1": exc})

    assert _warnings(caplog) == []


def test_recovery_logs_single_info(caplog):
    exc = ConnectionRefusedError("[Errno 111] Connection refused")
    hc._log_deployment_health_transitions([], _unhealthy(), {"d1": exc})

    caplog.clear()
    caplog.set_level(logging.INFO, logger="litellm.proxy.health_check")
    hc._log_deployment_health_transitions(_healthy(), [], {})

    infos = _infos(caplog)
    assert len(infos) == 1
    assert "reachable again" in infos[0].getMessage()
    assert hc._deployment_reachability_state["d1"]["reachable"] is True


def test_real_error_keeps_detail(caplog):
    caplog.set_level(logging.INFO, logger="litellm.proxy.health_check")
    exc = Exception("AuthenticationError: invalid api key - status 401")

    hc._log_deployment_health_transitions([], _unhealthy(), {"d1": exc})

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "failed its health check" in msg
    assert "invalid api key" in msg


def test_relog_cooldown_reemits_after_interval(caplog):
    litellm.health_check_unreachable_relog_seconds = 1
    exc = ConnectionRefusedError("[Errno 111] Connection refused")
    # transition down
    hc._log_deployment_health_transitions([], _unhealthy(), {"d1": exc})

    # simulate the cooldown having elapsed
    hc._deployment_reachability_state["d1"]["last_logged"] -= 100

    caplog.clear()
    caplog.set_level(logging.INFO, logger="litellm.proxy.health_check")
    hc._log_deployment_health_transitions([], _unhealthy(), {"d1": exc})

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "still unreachable" in warnings[0].getMessage()


def test_maybe_log_gated_to_background_loop(caplog):
    caplog.set_level(logging.INFO, logger="litellm.proxy.health_check")
    exc = ConnectionRefusedError("[Errno 111] Connection refused")

    # on-demand /health source: must not log or touch shared state
    hc._maybe_log_health_transitions("endpoint", [], _unhealthy(), {"d1": exc})
    assert _warnings(caplog) == []
    assert hc._deployment_reachability_state == {}

    # background loop source: logs and records state
    hc._maybe_log_health_transitions("proxy_background_loop", [], _unhealthy(), {"d1": exc})
    assert len(_warnings(caplog)) == 1
    assert hc._deployment_reachability_state["d1"]["reachable"] is False


def test_endpoint_without_model_id_is_ignored(caplog):
    caplog.set_level(logging.INFO, logger="litellm.proxy.health_check")
    hc._log_deployment_health_transitions(
        healthy_endpoints=[],
        unhealthy_endpoints=[{"model": "no-id"}],
        exceptions_by_model_id={},
    )
    assert _warnings(caplog) == []
    assert hc._deployment_reachability_state == {}
