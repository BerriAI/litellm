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
    _initialize_loggers_with_handler,
    _turn_on_json,
    session_id_var,
    set_session_id,
    set_trace_id,
    trace_id_var,
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


def test_json_formatter_includes_component_field():
    """
    Test that JsonFormatter always emits a 'component' field equal to the logger name.
    This allows filtering by component (e.g. "LiteLLM Proxy") in Datadog / third-party log services.
    """
    formatter = JsonFormatter()
    for logger_name in ("LiteLLM Proxy", "LiteLLM Router", "LiteLLM"):
        record = logging.LogRecord(
            name=logger_name,
            level=logging.ERROR,
            pathname="proxy_server.py",
            lineno=42,
            msg="something went wrong",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        obj = json.loads(output)
        assert (
            obj["component"] == logger_name
        ), f"Expected component={logger_name!r}, got {obj.get('component')!r}"


def test_json_formatter_includes_logger_field():
    """
    Test that JsonFormatter always emits a 'logger' field with filename:lineno.
    This allows pinpointing the exact source of a log line in third-party services.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM Proxy",
        level=logging.INFO,
        pathname="/app/litellm/proxy/proxy_server.py",
        lineno=123,
        msg="request received",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    obj = json.loads(output)
    assert (
        obj["logger"] == "proxy_server.py:123"
    ), f"Expected logger='proxy_server.py:123', got {obj['logger']!r}"


def test_json_formatter_extra_component_not_overwritten():
    """
    User-supplied extra={"component": "..."} must not be silently dropped.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM Proxy",
        level=logging.INFO,
        pathname="proxy_server.py",
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    record.component = "auth-service"
    obj = json.loads(formatter.format(record))
    assert (
        obj["component"] == "auth-service"
    ), f"User-supplied component was overwritten, got {obj['component']!r}"


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


class _JsonCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.formatter = JsonFormatter()
        self.records: list[dict] = []

    def emit(self, record):
        self.records.append(json.loads(self.formatter.format(record)))


def _make_capture_logger(name: str) -> tuple[logging.Logger, _JsonCapture]:
    lg = logging.getLogger(name)
    cap = _JsonCapture()
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    return lg, cap


def test_trace_id_injected_into_json_record():
    """trace_id set via set_trace_id() appears in every JSON record in that context."""
    lg, cap = _make_capture_logger("test.trace_inject")
    set_trace_id("trace-abc-123")
    try:
        lg.info("test message")
        assert len(cap.records) == 1
        assert cap.records[0]["trace_id"] == "trace-abc-123"
    finally:
        trace_id_var.set("")


def test_session_id_injected_when_set():
    """session_id set via set_session_id() appears in JSON record."""
    lg, cap = _make_capture_logger("test.session_inject")
    set_session_id("sess-xyz-456")
    try:
        lg.info("another message")
        assert cap.records[0]["session_id"] == "sess-xyz-456"
    finally:
        session_id_var.set("")


def test_session_id_absent_when_not_set():
    """session_id must NOT appear in JSON record when not set for this context."""
    lg, cap = _make_capture_logger("test.no_session")
    session_id_var.set("")
    lg.info("no session message")
    assert "session_id" not in cap.records[0]


def test_trace_id_absent_when_not_set():
    """trace_id must NOT appear when not set."""
    lg, cap = _make_capture_logger("test.no_trace")
    trace_id_var.set("")
    lg.info("no trace message")
    assert "trace_id" not in cap.records[0]


@pytest.mark.asyncio
async def test_contextvar_isolation_between_tasks():
    """Two concurrent async tasks each see only their own trace_id."""
    results: dict[str, str] = {}

    async def task(task_id: str, trace_id: str) -> None:
        set_trace_id(trace_id)
        await asyncio.sleep(0)
        results[task_id] = trace_id_var.get()

    await asyncio.gather(
        task("A", "trace-for-A"),
        task("B", "trace-for-B"),
    )

    assert results["A"] == "trace-for-A"
    assert results["B"] == "trace-for-B"


def test_logging_init_sets_trace_id():
    """Logging.__init__() must call set_trace_id with self.litellm_trace_id."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    trace_id_var.set("")

    log_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="call-001",
        function_id="fn-001",
        kwargs={},
    )
    assert trace_id_var.get() == log_obj.litellm_trace_id


def test_logging_init_sets_session_id_when_provided():
    """Logging.__init__() must call set_session_id when litellm_session_id is in kwargs."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    session_id_var.set("")

    Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="call-002",
        function_id="fn-002",
        kwargs={"litellm_session_id": "my-session-99"},
    )
    assert session_id_var.get() == "my-session-99"


def test_logging_init_resets_session_id_to_empty_when_absent():
    """When no session_id is in kwargs, Logging.__init__() must reset session_id_var to ""
    so a prior request's session_id does not leak into subsequent log records."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    session_id_var.set("preexisting-sid")

    Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="call-003",
        function_id="fn-003",
        kwargs={},
    )
    assert session_id_var.get() == ""
