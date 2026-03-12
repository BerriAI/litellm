import logging
from io import StringIO
from unittest.mock import patch

import pytest

from litellm._logging import (
    _redact_string,
    _secret_filter,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)

SECRET = "sk-proj-abc123def456ghi789jklmnopqrst"


@pytest.fixture(autouse=True)
def _enable_redaction():
    """Enable secret redaction for all tests in this module."""
    with patch("litellm._logging._ENABLE_SECRET_REDACTION", True):
        yield


def _capture_logger_output(fn):
    """Run fn with all litellm loggers wired to a StringIO buffer, return output."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.addFilter(_secret_filter)
    loggers = [verbose_logger, verbose_proxy_logger, verbose_router_logger]
    saved = [(lg, lg.handlers[:], lg.level) for lg in loggers]
    for lg in loggers:
        lg.handlers.clear()
        lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
    try:
        fn()
        return buf.getvalue()
    finally:
        for lg, handlers, level in saved:
            lg.handlers.clear()
            for old_h in handlers:
                lg.addHandler(old_h)
            lg.setLevel(level)


def test_redact_string_catches_secret_patterns():
    """Core regex patterns redact known secret formats."""
    cases = [
        "Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig",
        "api_key=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
        "password=supersecretpassword123",
        "postgresql://admin:s3cretpass@db.example.com:5432/mydb",
        SECRET,
    ]
    for secret in cases:
        result = _redact_string("msg: " + secret)
        assert secret not in result, f"{secret!r} was not redacted"
        assert "REDACTED" in result

    normal = "Loaded model gpt-4 with 3 replicas on us-east-1"
    assert _redact_string(normal) == normal


def test_filter_redacts_secrets_in_logger_output():
    def log_messages():
        verbose_logger.debug("Key: " + SECRET)
        verbose_logger.debug("Normal message with no secrets")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output
    assert "Normal message with no secrets" in output


def test_filter_redacts_percent_style_args():
    """Secrets passed as %-style args should be redacted."""

    def log_messages():
        verbose_logger.debug("key=%s region=%s", SECRET, "us-east-1")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "us-east-1" in output


def test_filter_redacts_non_string_args():
    """Secrets inside dicts/lists passed as %-style args should be redacted."""

    def log_messages():
        verbose_logger.debug("Config: %s", {"nested": {"key": SECRET}})
        verbose_logger.debug("Keys: %s", [SECRET])

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output


def test_filter_redacts_exception_tracebacks():
    """Secrets embedded in exception messages must be redacted in tracebacks."""

    def log_messages():
        try:
            raise ValueError(f"Auth failed with key {SECRET}")
        except ValueError:
            verbose_logger.exception("Something went wrong")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output
    assert "Something went wrong" in output


def test_filter_redacts_extra_fields():
    """Secrets passed via extra={...} must be redacted on the record."""
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="request completed",
        args=(),
        exc_info=None,
    )
    record.api_key = SECRET
    record.region = "us-east-1"

    _secret_filter.filter(record)

    assert SECRET not in record.api_key
    assert "REDACTED" in record.api_key
    assert record.region == "us-east-1"


def test_disable_redaction_passes_secrets_through():
    """When _ENABLE_SECRET_REDACTION is False, secrets pass through."""
    with patch("litellm._logging._ENABLE_SECRET_REDACTION", False):
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="key=" + SECRET,
            args=(),
            exc_info=None,
        )
        _secret_filter.filter(record)
        assert "sk-proj-" in record.msg
