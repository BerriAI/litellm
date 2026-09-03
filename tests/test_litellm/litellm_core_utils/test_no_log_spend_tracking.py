"""`no-log` must not disable spend tracking.

`no-log` means "do not send this request to logging integrations". It is not a
way for a caller to opt out of being billed, and `Logging.should_run_callback`
has always intended to exempt the proxy's cost-tracking callback from it.

The exemption used to be a case-sensitive substring test for `"_PROXY_"` in the
callback's class name, which never matched the cost callback's actual class
(`_ProxyDBLogger`), so spend tracking was silently skipped for every `no-log`
request. Callbacks now declare the behaviour explicitly via `runs_on_no_log`.
"""

from datetime import datetime

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger


class _ObservabilityLogger(CustomLogger):
    """Stand-in for langfuse/datadog/etc -- must be skipped on no-log."""


class _MeteringLogger(CustomLogger):
    """Stand-in for a billing callback -- must still run on no-log."""

    runs_on_no_log = True


@pytest.fixture
def logging_obj() -> Logging:
    return Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="my-unique-call-id",
        function_id="1234",
    )


def _should_run(logging_obj: Logging, callback, litellm_params: dict) -> bool:
    return logging_obj.should_run_callback(
        callback=callback,
        litellm_params=litellm_params,
        event_hook="success_handler",
    )


def test_cost_callback_still_runs_on_no_log(logging_obj):
    """The regression: `_ProxyDBLogger` must survive a no-log request."""
    assert _ProxyDBLogger.runs_on_no_log is True
    assert _should_run(logging_obj, _ProxyDBLogger(), {"no-log": True}) is True


def test_observability_callback_is_skipped_on_no_log(logging_obj):
    """no-log still does what it says for logging integrations."""
    assert _ObservabilityLogger.runs_on_no_log is False
    assert _should_run(logging_obj, _ObservabilityLogger(), {"no-log": True}) is False


def test_string_callbacks_are_still_skipped_on_no_log(logging_obj):
    """Callbacks referenced by name are not CustomLogger instances."""
    assert _should_run(logging_obj, "langfuse", {"no-log": True}) is False


def test_runs_on_no_log_is_opt_in_for_any_callback(logging_obj):
    """Any callback can declare itself infrastructure, not just the proxy's."""
    assert _should_run(logging_obj, _MeteringLogger(), {"no-log": True}) is True


@pytest.mark.parametrize("litellm_params", [{}, {"no-log": False}])
def test_everything_runs_when_no_log_absent_or_false(logging_obj, litellm_params):
    assert _should_run(logging_obj, _ObservabilityLogger(), litellm_params) is True
    assert _should_run(logging_obj, _ProxyDBLogger(), litellm_params) is True


def test_legacy_underscore_proxy_naming_still_exempt(logging_obj):
    """Callbacks named _PROXY_* keep working without setting the new flag."""

    class _PROXY_LegacyHandler(CustomLogger):
        pass

    assert _PROXY_LegacyHandler.runs_on_no_log is False
    assert _should_run(logging_obj, _PROXY_LegacyHandler(), {"no-log": True}) is True


def test_global_disable_no_log_param_overrides_everything(monkeypatch, logging_obj):
    monkeypatch.setattr(litellm, "global_disable_no_log_param", True)
    assert _should_run(logging_obj, _ObservabilityLogger(), {"no-log": True}) is True
