import asyncio
import json
import os
import sys
from typing import List

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path
import logging
import sys

import litellm
from litellm._logging import (
    ALL_LOGGERS,
    JsonFormatter,
    SensitiveLogFilter,
    _initialize_loggers_with_handler,
    _turn_on_json,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


class CacheHitCustomLogger(CustomLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_standard_logging_payloads: List[StandardLoggingPayload] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload = kwargs.get("standard_logging_object", None)
        if standard_logging_payload:
            self.logged_standard_logging_payloads.append(standard_logging_payload)


def test_json_mode_emits_one_record_per_logger(capfd):
    # Turn on JSON logging
    _turn_on_json()
    # Make sure our loggers will emit INFO-level records
    for lg in (verbose_logger, verbose_router_logger, verbose_proxy_logger):
        lg.setLevel(logging.INFO)

    # Log one message from each logger at different levels
    verbose_logger.info("first info")
    verbose_router_logger.info("second info from router")
    verbose_proxy_logger.info("third info from proxy")

    # Capture stdout
    out, err = capfd.readouterr()
    print("out", out)
    print("err", err)
    lines = [l for l in err.splitlines() if l.strip()]

    # Expect exactly three JSON lines
    assert len(lines) == 3, f"got {len(lines)} lines, want 3: {lines!r}"

    # Each line must be valid JSON with the required fields
    for line in lines:
        obj = json.loads(line)
        assert "message" in obj, "`message` key missing"
        assert "level" in obj, "`level` key missing"
        assert "timestamp" in obj, "`timestamp` key missing"


def test_json_formatter_parses_embedded_json_message():
    """
    Test that JsonFormatter parses embedded JSON in the message field and promotes
    sub-fields to first-class JSON properties for downstream querying.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg='{"event": "giveup", "exception": "Connection failed", "model_name": "gpt-4"}',
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    obj = json.loads(output)
    # Standard fields preserved
    assert "message" in obj
    assert obj["level"] == "DEBUG"
    assert "timestamp" in obj
    # Embedded JSON fields promoted to top-level for querying
    assert obj["event"] == "giveup"
    assert obj["exception"] == "Connection failed"
    assert obj["model_name"] == "gpt-4"


def test_json_formatter_includes_extra_attributes():
    """
    Test that JsonFormatter includes extra attributes from logger.debug("msg", extra={...}).
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="POST Request Sent from LiteLLM",
        args=(),
        exc_info=None,
    )
    record.api_base = "https://api.openai.com"
    record.authorization = "Bearer sk-***"
    output = formatter.format(record)
    obj = json.loads(output)
    assert obj["message"] == "POST Request Sent from LiteLLM"
    assert obj["api_base"] == "https://api.openai.com"
    assert obj["authorization"] == "Bearer sk-***"


def test_json_formatter_plain_message_unchanged():
    """
    Test that non-JSON messages are passed through as-is in the message field.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Cache hit!",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    obj = json.loads(output)
    assert obj["message"] == "Cache hit!"
    assert "event" not in obj
    assert "exception" not in obj


def test_json_formatter_parses_embedded_python_dict_repr():
    """
    Test that JsonFormatter parses Python dict repr (str/deployment) embedded in
    plain text, e.g. from get_available_deployment logs.
    Reproduces Roni's reported case.
    """
    formatter = JsonFormatter()
    msg = (
        "get_available_deployment for model: text-embedding-3-large, "
        "Selected deployment: {'model_name': 'text-embedding-3-large', "
        "'litellm_params': {'api_key': 'sk**********', 'tpm': 1000000, 'rpm': 2000, "
        "'use_in_pass_through': False, 'use_litellm_proxy': False, "
        "'merge_reasoning_content_in_choices': False, 'model': 'text-embedding-3-large'}, "
        "'model_info': {'id': 'a624b057aec64ada48311', 'db_model': False}} "
        "for model: text-embedding-3-large"
    )
    record = logging.LogRecord(
        name="LiteLLM Router",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    obj = json.loads(output)
    assert "message" in obj
    assert obj["level"] == "INFO"
    # Python dict parsed and promoted to first-class properties
    assert obj["model_name"] == "text-embedding-3-large"
    assert "litellm_params" in obj
    assert obj["litellm_params"]["api_key"] == "sk**********"
    assert obj["litellm_params"]["tpm"] == 1000000
    assert obj["litellm_params"]["use_in_pass_through"] is False
    assert "model_info" in obj
    assert obj["model_info"]["id"] == "a624b057aec64ada48311"
    assert obj["model_info"]["db_model"] is False


def test_initialize_loggers_with_handler_sets_propagate_false():
    """
    Test that the initialize_loggers_with_handler function sets propagate to False for all loggers
    """
    # Initialize loggers with the test handler
    _initialize_loggers_with_handler(logging.StreamHandler())

    # Check that propagate is set to False for all loggers
    for logger in ALL_LOGGERS:
        assert (
            logger.propagate is False
        ), f"Logger {logger.name} has propagate set to {logger.propagate}, expected False"


@pytest.mark.asyncio
async def test_cache_hit_includes_custom_llm_provider():
    """
    Test that when there's a cache hit, the standard logging payload includes the custom_llm_provider
    """
    # Set up caching and custom logger
    litellm.cache = litellm.Cache()
    test_custom_logger = CacheHitCustomLogger()
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    litellm.callbacks = [test_custom_logger]

    try:
        # First call - should be a cache miss
        response1 = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test cache hit message"}],
            mock_response="test response",
            caching=True,
        )

        # Wait for logging to complete
        await asyncio.sleep(0.5)

        # Second identical call - should be a cache hit
        response2 = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test cache hit message"}],
            mock_response="test response",
            caching=True,
        )

        # Wait for logging to complete
        await asyncio.sleep(0.5)

        # Verify we have logged events
        assert (
            len(test_custom_logger.logged_standard_logging_payloads) >= 2
        ), f"Expected at least 2 logged events, got {len(test_custom_logger.logged_standard_logging_payloads)}"

        # Find the cache hit event (should be the second call)
        cache_hit_payload = None
        for payload in test_custom_logger.logged_standard_logging_payloads:
            if payload.get("cache_hit") is True:
                cache_hit_payload = payload
                break

        # Verify cache hit event was found
        assert (
            cache_hit_payload is not None
        ), "No cache hit event found in logged payloads"

        # Verify custom_llm_provider is included in the cache hit payload
        assert (
            "custom_llm_provider" in cache_hit_payload
        ), "custom_llm_provider missing from cache hit standard logging payload"

        # Verify custom_llm_provider has a valid value (should be "openai" for gpt-3.5-turbo)
        custom_llm_provider = cache_hit_payload["custom_llm_provider"]
        assert (
            custom_llm_provider is not None and custom_llm_provider != ""
        ), f"custom_llm_provider should not be None or empty, got: {custom_llm_provider}"

        print(
            f"Cache hit standard logging payload with custom_llm_provider: {custom_llm_provider}",
            json.dumps(cache_hit_payload, indent=2),
        )

    finally:
        # Clean up
        litellm.callbacks = original_callbacks
        litellm.cache = None


# ---------------------------------------------------------------------------
# SensitiveLogFilter tests
# ---------------------------------------------------------------------------

def _make_record(msg, args=(), level=logging.DEBUG, name="LiteLLM"):
    """Helper to create a LogRecord for filter tests."""
    return logging.LogRecord(
        name=name,
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=args,
        exc_info=None,
    )


class TestSensitiveLogFilter:
    """Tests for the SensitiveLogFilter that redacts Bearer tokens."""

    def setup_method(self):
        self.filt = SensitiveLogFilter()

    def test_should_redact_bearer_token_in_plain_message(self):
        record = _make_record(
            "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.sig"
        )
        self.filt.filter(record)
        msg = record.getMessage()
        assert "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9" not in msg
        assert "Bearer [REDACTED]" in msg

    def test_should_redact_bearer_in_dict_format_arg(self):
        data = {
            "model": "gpt-4",
            "secret_fields": {
                "raw_headers": {
                    "authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxIn0.abc"
                }
            },
        }
        record = _make_record("receiving data: %s", (data,))
        self.filt.filter(record)
        msg = record.getMessage()
        assert "eyJhbGciOiJSUzI1NiJ9" not in msg
        assert "Bearer [REDACTED]" in msg
        assert "gpt-4" in msg

    def test_should_leave_non_sensitive_messages_unchanged(self):
        record = _make_record("Cache hit for model gpt-4, returning cached response")
        self.filt.filter(record)
        assert record.getMessage() == "Cache hit for model gpt-4, returning cached response"

    def test_should_handle_nested_dict_with_authorization(self):
        data = {
            "proxy_server_request": {
                "url": "http://localhost:4000/v1/chat/completions",
                "method": "POST",
                "headers": {
                    "authorization": "Bearer sk-proj-abc123def456"
                },
            }
        }
        record = _make_record("receiving data: %s", (data,))
        self.filt.filter(record)
        msg = record.getMessage()
        assert "sk-proj-abc123def456" not in msg
        assert "Bearer [REDACTED]" in msg

    def test_should_not_redact_bearer_word_alone(self):
        record = _make_record("The bearer of this message is authorized")
        self.filt.filter(record)
        assert record.getMessage() == "The bearer of this message is authorized"

    def test_should_redact_multiple_tokens_in_one_message(self):
        record = _make_record(
            "Header1: Bearer tokenAAA, Header2: Bearer tokenBBB"
        )
        self.filt.filter(record)
        msg = record.getMessage()
        assert "tokenAAA" not in msg
        assert "tokenBBB" not in msg
        assert msg.count("Bearer [REDACTED]") == 2

    def test_should_be_case_insensitive(self):
        record = _make_record("auth: bearer eyJsecretPayload.data.sig")
        self.filt.filter(record)
        msg = record.getMessage()
        assert "eyJsecretPayload" not in msg
        assert "[REDACTED]" in msg

    def test_should_always_return_true(self):
        record = _make_record("Bearer supersecrettoken123")
        result = self.filt.filter(record)
        assert result is True

    def test_should_preserve_args_when_no_token_present(self):
        record = _make_record("model=%s tokens=%d", ("gpt-4", 150))
        self.filt.filter(record)
        assert record.args == ("gpt-4", 150)
        assert record.getMessage() == "model=gpt-4 tokens=150"

    def test_should_clear_args_after_redaction(self):
        record = _make_record("auth: %s", ("Bearer my_jwt_token_here",))
        self.filt.filter(record)
        assert record.args is None
        msg = record.getMessage()
        assert "my_jwt_token_here" not in msg
        assert "Bearer [REDACTED]" in msg
