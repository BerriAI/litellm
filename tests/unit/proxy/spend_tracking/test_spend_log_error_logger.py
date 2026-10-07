"""
Unit tests for ``litellm.proxy.spend_tracking.spend_log_error_logger``.

The helper exists to let proxy operators silence the multi-line stack traces
that the spend-tracking machinery normally emits on 4xx/5xx and DB errors.
These tests cover:

  * the env-var gating behavior (opt-in, off by default),
  * the interaction between the env var and the proxy log level (DEBUG always
    keeps the traceback, INFO/WARNING honors the opt-in), and
  * the fact that ``spend_log_error`` always emits an ERROR-level record so
    operators can still see the failure summary.
"""

import logging

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy.spend_tracking.spend_log_error_logger import (
    SUPPRESS_SPEND_LOG_TRACEBACKS_ENV,
    should_suppress_spend_log_tracebacks,
    spend_log_error,
)


@pytest.fixture
def reset_env_and_level(monkeypatch):
    """Restore both the env var and proxy logger level after each test."""
    monkeypatch.delenv(SUPPRESS_SPEND_LOG_TRACEBACKS_ENV, raising=False)
    original_level = verbose_proxy_logger.level
    yield monkeypatch
    verbose_proxy_logger.setLevel(original_level)


def test_should_suppress_default_is_false(reset_env_and_level):
    """With no env var set, suppression is off so existing operators see no change."""
    verbose_proxy_logger.setLevel(logging.INFO)
    assert should_suppress_spend_log_tracebacks() is False


@pytest.mark.parametrize("value", ["true", "True", "TRUE"])
def test_should_suppress_when_env_true_at_info(reset_env_and_level, value):
    reset_env_and_level.setenv(SUPPRESS_SPEND_LOG_TRACEBACKS_ENV, value)
    verbose_proxy_logger.setLevel(logging.INFO)
    assert should_suppress_spend_log_tracebacks() is True


@pytest.mark.parametrize("value", ["false", "False", "no", "0", "", "garbage"])
def test_should_not_suppress_when_env_falsy(reset_env_and_level, value):
    if value == "":
        # ``""`` would be ambiguous; ensure the var is genuinely unset.
        reset_env_and_level.delenv(SUPPRESS_SPEND_LOG_TRACEBACKS_ENV, raising=False)
    else:
        reset_env_and_level.setenv(SUPPRESS_SPEND_LOG_TRACEBACKS_ENV, value)
    verbose_proxy_logger.setLevel(logging.INFO)
    assert should_suppress_spend_log_tracebacks() is False


def test_debug_level_overrides_suppression(reset_env_and_level):
    """DEBUG always shows the traceback even when the env var is set."""
    reset_env_and_level.setenv(SUPPRESS_SPEND_LOG_TRACEBACKS_ENV, "true")
    verbose_proxy_logger.setLevel(logging.DEBUG)
    assert should_suppress_spend_log_tracebacks() is False


def test_spend_log_error_includes_traceback_by_default(reset_env_and_level, caplog):
    """Default behavior: ERROR record carries exc_info so the formatter renders it."""
    verbose_proxy_logger.setLevel(logging.INFO)
    caplog.set_level(logging.ERROR, logger=verbose_proxy_logger.name)

    try:
        raise ValueError("boom")
    except ValueError as e:
        spend_log_error("update failed: %s", str(e), exc=e)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert "update failed: boom" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[0] is ValueError


def test_spend_log_error_drops_traceback_when_env_set(reset_env_and_level, caplog):
    """Opt-in path: ERROR record still emitted, but exc_info is stripped."""
    reset_env_and_level.setenv(SUPPRESS_SPEND_LOG_TRACEBACKS_ENV, "true")
    verbose_proxy_logger.setLevel(logging.INFO)
    caplog.set_level(logging.ERROR, logger=verbose_proxy_logger.name)

    try:
        raise ValueError("boom")
    except ValueError as e:
        spend_log_error("update failed: %s", str(e), exc=e)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert "update failed: boom" in record.getMessage()
    assert record.exc_info is None


def test_spend_log_error_keeps_traceback_at_debug_even_with_env(
    reset_env_and_level, caplog
):
    """DEBUG operators always get tracebacks; the env var doesn't apply."""
    reset_env_and_level.setenv(SUPPRESS_SPEND_LOG_TRACEBACKS_ENV, "true")
    verbose_proxy_logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger=verbose_proxy_logger.name)

    try:
        raise RuntimeError("boom-at-debug")
    except RuntimeError as e:
        spend_log_error("update failed: %s", str(e), exc=e)

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 1
    record = error_records[0]
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_spend_log_error_uses_active_exception_when_exc_omitted(
    reset_env_and_level, caplog
):
    """When called inside an ``except`` block without ``exc=``, the active
    exception's traceback should still be attached."""
    verbose_proxy_logger.setLevel(logging.INFO)
    caplog.set_level(logging.ERROR, logger=verbose_proxy_logger.name)

    try:
        raise KeyError("missing")
    except KeyError:
        spend_log_error("update failed without exc kwarg")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.exc_info is not None
    assert record.exc_info[0] is KeyError
