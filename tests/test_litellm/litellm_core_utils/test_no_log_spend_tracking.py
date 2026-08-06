"""`no-log` must not disable spend tracking.

`no-log` means "do not send this request to logging integrations". It is not a
way for a caller to opt out of being billed, and `Logging.should_run_callback`
has always intended to exempt the proxy's cost-tracking callback from it. The
exemption used to be a case-sensitive substring test for `"_PROXY_"` in the
callback's class name, which never matched the cost callback's actual class
(`_ProxyDBLogger`), so spend tracking was silently skipped for every `no-log`
request. Callbacks now declare the behaviour explicitly via `runs_on_no_log`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger


class _ObservabilityLogger(CustomLogger):
    """A stand-in for langfuse/datadog/etc. -- must be skipped on no-log."""


class _MeteringLogger(CustomLogger):
    """A stand-in for a billing callback -- must still run on no-log."""

    runs_on_no_log = True


def _logging_obj() -> LiteLLMLogging:
    return LiteLLMLogging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        start_time=None,
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )


def test_cost_callback_still_runs_on_no_log():
    """The regression: _ProxyDBLogger must survive a no-log request."""
    assert _ProxyDBLogger.runs_on_no_log is True
    assert (
        _logging_obj().should_run_callback(
            callback=_ProxyDBLogger(),
            litellm_params={"no-log": True},
            event_hook="async_log_success_event",
        )
        is True
    )


def test_observability_callback_is_skipped_on_no_log():
    """no-log still does what it says for logging integrations."""
    assert _ObservabilityLogger.runs_on_no_log is False
    assert (
        _logging_obj().should_run_callback(
            callback=_ObservabilityLogger(),
            litellm_params={"no-log": True},
            event_hook="async_log_success_event",
        )
        is False
    )


def test_runs_on_no_log_is_opt_in_for_any_callback():
    """Any callback can declare itself as infrastructure, not just the proxy's."""
    assert (
        _logging_obj().should_run_callback(
            callback=_MeteringLogger(),
            litellm_params={"no-log": True},
            event_hook="async_log_success_event",
        )
        is True
    )


@pytest.mark.parametrize("litellm_params", [{}, {"no-log": False}])
def test_everything_runs_when_no_log_is_absent_or_false(litellm_params):
    for callback in (_ObservabilityLogger(), _ProxyDBLogger()):
        assert (
            _logging_obj().should_run_callback(
                callback=callback,
                litellm_params=litellm_params,
                event_hook="async_log_success_event",
            )
            is True
        )


def test_legacy_underscore_proxy_naming_still_exempt():
    """Callbacks named _PROXY_* keep working without setting the new flag."""

    class _PROXY_LegacyHandler(CustomLogger):
        pass

    assert _PROXY_LegacyHandler.runs_on_no_log is False
    assert (
        _logging_obj().should_run_callback(
            callback=_PROXY_LegacyHandler(),
            litellm_params={"no-log": True},
            event_hook="async_log_success_event",
        )
        is True
    )


def test_global_disable_no_log_param_overrides_everything():
    original = litellm.global_disable_no_log_param
    litellm.global_disable_no_log_param = True
    try:
        assert (
            _logging_obj().should_run_callback(
                callback=_ObservabilityLogger(),
                litellm_params={"no-log": True},
                event_hook="async_log_success_event",
            )
            is True
        )
    finally:
        litellm.global_disable_no_log_param = original
