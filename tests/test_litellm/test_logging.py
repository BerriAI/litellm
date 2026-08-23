import ast
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import List

import pytest

import logging

import litellm
from litellm._logging import (
    ALL_LOGGERS,
    CorrelationContextFilter,
    CorrelationPlainFormatter,
    JsonFormatter,
    SecretRedactionFilter,
    StdoutLogTruncationFilter,
    _initialize_loggers_with_handler,
    _stdout_truncation_marker,
    _turn_on_json,
    session_id_var,
    set_session_id,
    set_trace_id,
    trace_id_var,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)
from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD
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
    # Redacted, not passed through: SecretRedactionFilter already collapses this
    # pair in the plain path before any formatter sees it, so the JSON path matching
    # it is production parity. The key survives because redaction is per-value here.
    assert obj["litellm_params"]["api_key"] == "REDACTED"
    assert obj["litellm_params"]["tpm"] == 1000000
    assert obj["litellm_params"]["use_in_pass_through"] is False
    assert "model_info" in obj
    assert obj["model_info"]["id"] == "a624b057aec64ada48311"
    assert obj["model_info"]["db_model"] is False


def test_json_formatter_output_stays_parseable_when_a_secret_is_redacted():
    """Redaction must collapse the value only, never the surrounding JSON member.

    Redacting the serialized document turned '"api_key": "sk-..."' into a bare
    REDACTED token, so the line stopped being valid JSON entirely.
    """
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="calling deployment",
        args=(),
        exc_info=None,
    )
    record.deployment = {
        "api_key": "sk-abcdefghijklmnopqrstuvwxyz0123456789",
        "aws_secret_access_key": "wJalrXUtnFEMIQAfakeKEYbPxRfiCYEXAMPLEKEY",
        "aws_region_name": "us-east-1",
        "nested": {"tokens": ["Bearer abcdefghijklmnop", "keep-me"]},
    }

    obj = json.loads(formatter.format(record))

    assert obj["deployment"]["api_key"] == "REDACTED"
    assert obj["deployment"]["aws_secret_access_key"] == "REDACTED"
    # Non-secret siblings stay legible so the logs remain useful
    assert obj["deployment"]["aws_region_name"] == "us-east-1"
    assert obj["deployment"]["nested"]["tokens"] == ["REDACTED", "keep-me"]


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
        assert obj["component"] == logger_name, f"Expected component={logger_name!r}, got {obj.get('component')!r}"


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
    assert obj["logger"] == "proxy_server.py:123", f"Expected logger='proxy_server.py:123', got {obj['logger']!r}"


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
    assert obj["component"] == "auth-service", f"User-supplied component was overwritten, got {obj['component']!r}"


def test_initialize_loggers_with_handler_sets_propagate_false():
    """
    Test that the initialize_loggers_with_handler function sets propagate to False for all loggers
    """
    # Initialize loggers with the test handler
    _initialize_loggers_with_handler(logging.StreamHandler())

    # Check that propagate is set to False for all loggers
    for logger in ALL_LOGGERS:
        assert logger.propagate is False, (
            f"Logger {logger.name} has propagate set to {logger.propagate}, expected False"
        )


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
        assert len(test_custom_logger.logged_standard_logging_payloads) >= 2, (
            f"Expected at least 2 logged events, got {len(test_custom_logger.logged_standard_logging_payloads)}"
        )

        # Find the cache hit event (should be the second call)
        cache_hit_payload = None
        for payload in test_custom_logger.logged_standard_logging_payloads:
            if payload.get("cache_hit") is True:
                cache_hit_payload = payload
                break

        # Verify cache hit event was found
        assert cache_hit_payload is not None, "No cache hit event found in logged payloads"

        # Verify custom_llm_provider is included in the cache hit payload
        assert "custom_llm_provider" in cache_hit_payload, (
            "custom_llm_provider missing from cache hit standard logging payload"
        )

        # Verify custom_llm_provider has a valid value (should be "openai" for gpt-3.5-turbo)
        custom_llm_provider = cache_hit_payload["custom_llm_provider"]
        assert custom_llm_provider is not None and custom_llm_provider != "", (
            f"custom_llm_provider should not be None or empty, got: {custom_llm_provider}"
        )

        print(
            f"Cache hit standard logging payload with custom_llm_provider: {custom_llm_provider}",
            json.dumps(cache_hit_payload, indent=2),
        )

    finally:
        # Clean up
        litellm.callbacks = original_callbacks
        litellm.cache = None


LITELLM_LOGGER_NAMES = frozenset(
    {"verbose_logger", "verbose_proxy_logger", "verbose_router_logger", "logger", "logging"}
)
LOG_LEVEL_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})
LITELLM_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "litellm"


def _receiver_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_logging_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOG_LEVEL_METHODS
        and _receiver_name(node.func.value) in LITELLM_LOGGER_NAMES
    )


def _has_format_spec(message: ast.JoinedStr) -> bool:
    return any(isinstance(value, ast.FormattedValue) and value.format_spec is not None for value in message.values)


def _eager_logging_calls(source: str, path: Path) -> tuple[str, ...]:
    return tuple(
        f"{path}:{node.lineno}"
        for node in ast.walk(ast.parse(source))
        if _is_logging_call(node)
        and node.args
        and isinstance(node.args[0], ast.JoinedStr)
        and not _has_format_spec(node.args[0])
    )


def test_logging_calls_do_not_build_their_message_eagerly():
    """A discarded log record must not have cost anything to build.

    `log.debug(f"payload: {body}")` interpolates before the call runs, so the message is
    built and thrown away on every request the level filters out; `log.debug("payload: %s", body)`
    defers that to `record.getMessage()`, which only runs once the record passes the level check.

    f-strings carrying a format spec are exempt: `%`-style has no faithful equivalent for
    specs like `{ratio:.1%}`, and those sites interpolate scalars rather than payloads.
    """
    offenders = tuple(
        offender
        for path in sorted(LITELLM_PACKAGE_ROOT.rglob("*.py"))
        for offender in _eager_logging_calls(
            path.read_text(encoding="utf-8"), path.relative_to(LITELLM_PACKAGE_ROOT.parent)
        )
    )

    assert offenders == (), (
        "these logging calls build their message eagerly; pass the values as %-style arguments instead:\n"
        + "\n".join(offenders)
    )


class _JsonCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.formatter = JsonFormatter()
        self.records: list[dict] = []
        self.addFilter(CorrelationContextFilter())

    def emit(self, record):
        self.records.append(json.loads(self.formatter.format(record)))


def _make_capture_logger(name: str) -> tuple[logging.Logger, _JsonCapture]:
    lg = logging.getLogger(name)
    cap = _JsonCapture()
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    return lg, cap


def test_trace_id_injected_into_json_record(monkeypatch):
    """trace_id set via set_trace_id() appears in every JSON record in that context."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    lg, cap = _make_capture_logger("test.trace_inject")
    set_trace_id("trace-abc-123")
    try:
        lg.info("test message")
        assert len(cap.records) == 1
        assert cap.records[0]["trace_id"] == "trace-abc-123"
    finally:
        trace_id_var.set("")


def test_session_id_injected_when_set(monkeypatch):
    """session_id set via set_session_id() appears in JSON record."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    lg, cap = _make_capture_logger("test.session_inject")
    set_session_id("sess-xyz-456")
    try:
        lg.info("another message")
        assert cap.records[0]["session_id"] == "sess-xyz-456"
    finally:
        session_id_var.set("")


def test_trace_id_and_session_id_cannot_be_spoofed_by_message_content(monkeypatch):
    """A log message that happens to parse as JSON/dict with "trace_id"/"session_id"
    keys (e.g. the proxy logging a raw request-header dict) must not override the
    real correlation ids set via set_trace_id()/set_session_id()."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    lg, cap = _make_capture_logger("test.spoof_attempt")
    set_trace_id("real-trace-id")
    set_session_id("real-session-id")
    try:
        lg.info('{"trace_id": "attacker-supplied-trace", "session_id": "attacker-supplied-session"}')
        assert cap.records[0]["trace_id"] == "real-trace-id"
        assert cap.records[0]["session_id"] == "real-session-id"
    finally:
        trace_id_var.set("")
        session_id_var.set("")


def test_trace_id_and_session_id_cannot_be_injected_with_no_active_context(monkeypatch):
    """A message that happens to parse as JSON/dict with "trace_id"/"session_id" keys
    must not surface those fields at all when CorrelationContextFilter hasn't stamped
    this record - e.g. a log line emitted before Logging.__init__() runs for a request
    (request_correlation_in_logs on, but no genuine trace/session id active yet)."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    lg, cap = _make_capture_logger("test.no_context_spoof_attempt")
    trace_id_var.set("")
    session_id_var.set("")
    lg.info('{"trace_id": "attacker-supplied-trace", "session_id": "attacker-supplied-session"}')
    assert "trace_id" not in cap.records[0]
    assert "session_id" not in cap.records[0]


def test_trace_id_and_session_id_are_redacted_when_credential_shaped(monkeypatch):
    """A caller-controlled trace_id/session_id (e.g. from x-litellm-trace-id or a W3C
    baggage header) that happens to look like a real credential must not reach log
    records unredacted. CorrelationContextFilter stamps trace_id/session_id onto the
    record after SecretRedactionFilter has already run, so those two fields would
    otherwise bypass credential redaction entirely - the fix redacts at set_trace_id()/
    set_session_id() time instead, before the value ever reaches a log record."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    lg, cap = _make_capture_logger("test.credential_shaped_correlation_id")
    poisoned_trace_id = "sk-ant-api03-" + "A" * 40
    poisoned_session_id = "AKIA" + "B" * 16
    set_trace_id(poisoned_trace_id)
    set_session_id(poisoned_session_id)
    try:
        lg.info("some benign log line")
        assert cap.records[0]["trace_id"] == "REDACTED"
        assert cap.records[0]["session_id"] == "REDACTED"
        assert poisoned_trace_id not in json.dumps(cap.records[0])
        assert poisoned_session_id not in json.dumps(cap.records[0])
    finally:
        trace_id_var.set("")
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


def test_trace_id_not_in_log_when_flag_disabled(monkeypatch):
    """When request_correlation_in_logs is False (default), trace_id must not appear in JSON records even when set."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", False)
    lg, cap = _make_capture_logger("test.no_trace_gated")
    set_trace_id("trace-should-not-appear")
    try:
        lg.info("message")
        assert "trace_id" not in cap.records[0]
    finally:
        trace_id_var.set("")


def test_session_id_not_in_log_when_flag_disabled(monkeypatch):
    """When request_correlation_in_logs is False (default), session_id must not appear in JSON records even when set."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", False)
    lg, cap = _make_capture_logger("test.no_session_gated")
    set_session_id("sess-should-not-appear")
    try:
        lg.info("message")
        assert "session_id" not in cap.records[0]
    finally:
        session_id_var.set("")


class _PlainCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.formatter = CorrelationPlainFormatter("%(message)s")
        self.records: list[str] = []
        self.addFilter(CorrelationContextFilter())

    def emit(self, record):
        self.records.append(self.formatter.format(record))


def _make_plain_capture_logger(name: str) -> tuple[logging.Logger, _PlainCapture]:
    lg = logging.getLogger(name)
    cap = _PlainCapture()
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    return lg, cap


def test_plain_formatter_appends_trace_id_and_session_id(monkeypatch):
    """CorrelationPlainFormatter must append trace_id/session_id to non-JSON log lines too."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    lg, cap = _make_plain_capture_logger("test.plain_trace_session")
    set_trace_id("plain-trace-1")
    set_session_id("plain-session-1")
    try:
        lg.info("plaintext message")
        assert cap.records[0] == "plaintext message [trace_id=plain-trace-1 session_id=plain-session-1]"
    finally:
        trace_id_var.set("")
        session_id_var.set("")


def test_plain_formatter_appends_only_trace_id_when_session_id_absent(monkeypatch):
    """Only trace_id is appended when session_id was never set."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", True)
    lg, cap = _make_plain_capture_logger("test.plain_trace_only")
    set_trace_id("plain-trace-2")
    session_id_var.set("")
    try:
        lg.info("plaintext message")
        assert cap.records[0] == "plaintext message [trace_id=plain-trace-2]"
    finally:
        trace_id_var.set("")


def test_plain_formatter_unchanged_when_flag_disabled(monkeypatch):
    """When request_correlation_in_logs is False, plain log lines are unmodified even if the contextvars are set."""
    monkeypatch.setattr(litellm, "request_correlation_in_logs", False)
    lg, cap = _make_plain_capture_logger("test.plain_flag_off")
    set_trace_id("should-not-appear")
    set_session_id("should-not-appear")
    try:
        lg.info("plaintext message")
        assert cap.records[0] == "plaintext message"
    finally:
        trace_id_var.set("")
        session_id_var.set("")


def test_set_trace_id_strips_control_characters():
    """set_trace_id() must strip \\r/\\n/escape sequences so a caller-controlled
    trace id can't forge fake log entries when interpolated into plain-text logs."""
    token = set_trace_id('evil\r\n{"level": "CRITICAL", "message": "forged"}')
    try:
        value = trace_id_var.get()
        assert "\r" not in value
        assert "\n" not in value
    finally:
        trace_id_var.reset(token)


_MARKER_RE = re.compile(rf"\.\.\. \({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped (\d+) chars\..*?\) \.\.\.", re.S)


def _extract_marker(text: str) -> "re.Match[str] | None":
    return _MARKER_RE.search(text)


def _make_record(level: int, msg: str, args=(), exc_info=None) -> logging.LogRecord:
    return logging.LogRecord(
        name="LiteLLM Router",
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )


def test_oversized_info_record_is_truncated(monkeypatch):
    """An error string echoing a huge request payload must not reach stdout in full."""
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "500")
    payload = "p" * 100_000
    record = _make_record(logging.INFO, "litellm.acompletion(model=%s) Exception %s", ("gpt-4", payload))

    assert StdoutLogTruncationFilter().filter(record) is True

    message = record.getMessage()
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD in message
    assert len(message) <= 500
    assert message.startswith("litellm.acompletion(model=gpt-4) Exception ppp")
    assert message.endswith("ppp")

    marker = _extract_marker(message)
    assert marker is not None
    kept, skipped = len(message) - len(marker.group(0)), int(marker.group(1))
    assert kept + skipped == 43 + len(payload)


def test_truncated_message_fits_the_configured_cap(monkeypatch):
    """The cap is the whole point of the setting, so the marker has to be paid for out of
    the budget instead of appended on top of a limit-sized head and tail."""
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "500")
    record = _make_record(logging.ERROR, "Exception %s", ("p" * 2000,))

    assert StdoutLogTruncationFilter().filter(record) is True

    message = record.getMessage()
    assert _extract_marker(message) is not None
    assert len(message) == 500


@pytest.mark.parametrize("payload_len", [501, 512, 1000, 9999, 100_000])
def test_truncated_message_never_exceeds_the_cap(monkeypatch, payload_len):
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "500")
    record = _make_record(logging.ERROR, "%s", ("p" * payload_len,))

    assert StdoutLogTruncationFilter().filter(record) is True

    assert len(record.getMessage()) <= 500


_NO_BUDGET_PAYLOAD = "p" * 2000
_MARKER_SIZED_CAP = len(_stdout_truncation_marker(len(_NO_BUDGET_PAYLOAD)))


@pytest.mark.parametrize("cap", [_MARKER_SIZED_CAP, _MARKER_SIZED_CAP - 1, 100])
def test_cap_leaving_no_room_for_the_marker_still_bounds_output(monkeypatch, cap):
    """An operator can set the cap at or below the marker's own length, leaving nothing to
    spend on a head and tail, and the output still has to fit."""
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", str(cap))
    record = _make_record(logging.ERROR, "%s", (_NO_BUDGET_PAYLOAD,))

    assert StdoutLogTruncationFilter().filter(record) is True

    assert len(record.getMessage()) == cap


def test_debug_record_is_not_truncated(monkeypatch):
    """--detailed_debug exists to dump full payloads, so DEBUG records pass through."""
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "500")
    payload = "p" * 100_000
    record = _make_record(logging.DEBUG, "raw request %s", (payload,))

    assert StdoutLogTruncationFilter().filter(record) is True

    assert record.getMessage() == f"raw request {payload}"


def test_truncation_disabled_by_zero_limit(monkeypatch):
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "0")
    payload = "p" * 100_000
    record = _make_record(logging.ERROR, "Exception %s", (payload,))

    assert StdoutLogTruncationFilter().filter(record) is True

    assert record.getMessage() == f"Exception {payload}"


def test_oversized_traceback_is_truncated(monkeypatch):
    """verbose_proxy_logger.exception() re-logs the payload inside the traceback too."""
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "500")
    try:
        raise ValueError("payload " + "p" * 100_000)
    except ValueError:
        exc_info = sys.exc_info()
    record = _make_record(logging.ERROR, "Exception occured", exc_info=exc_info)

    assert StdoutLogTruncationFilter().filter(record) is True

    assert record.exc_text is not None
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD in record.exc_text
    assert len(record.exc_text) <= 500
    assert "Traceback (most recent call last)" in record.exc_text


def test_falsy_exc_info_is_not_formatted(monkeypatch):
    """Callers pass exc_info=False, which logging leaves on the record as a bool."""
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "500")
    record = _make_record(logging.WARNING, "skipping malformed endpoint %s", ("p" * 100_000,), exc_info=False)

    assert StdoutLogTruncationFilter().filter(record) is True

    assert record.exc_text is None
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD in record.getMessage()


def test_secret_filter_keeps_truncated_traceback(monkeypatch):
    """SecretRedactionFilter runs after truncation, so it must redact the capped
    traceback instead of reformatting the full one from exc_info."""
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "500")
    try:
        raise ValueError("sk-1234567890abcdefghij payload " + "p" * 100_000)
    except ValueError:
        exc_info = sys.exc_info()
    record = _make_record(logging.ERROR, "Exception occured", exc_info=exc_info)

    assert StdoutLogTruncationFilter().filter(record) is True
    assert SecretRedactionFilter().filter(record) is True

    assert record.exc_text is not None
    assert len(record.exc_text) <= 500
    assert "sk-1234567890abcdefghij" not in record.exc_text


def test_truncation_filter_survives_json_reconfiguration():
    """The cap lives on the loggers, so swapping handlers (JSON mode) can't drop it."""
    _turn_on_json()

    for lg in (verbose_logger, verbose_router_logger, verbose_proxy_logger):
        assert any(isinstance(f, StdoutLogTruncationFilter) for f in lg.filters), f"{lg.name} lost stdout truncation"


def test_oversized_error_is_truncated_end_to_end(monkeypatch, caplog):
    """The router's own exception log line must come out bounded, not just the filter in isolation."""
    monkeypatch.setenv("MAX_STRING_LENGTH_STDOUT_LOG", "500")

    with caplog.at_level(logging.INFO, logger="LiteLLM Router"):
        verbose_router_logger.info("litellm.acompletion(model=%s) Exception %s", "gpt-4", "p" * 100_000)

    emitted = "".join(record.getMessage() for record in caplog.records)
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD in emitted
    assert len(emitted) <= 500


def test_set_session_id_bounds_length():
    """set_session_id() must bound length so an oversized caller-supplied value
    isn't repeated across every log line for the request."""
    token = set_session_id("a" * 1000)
    try:
        assert len(session_id_var.get()) == 256
    finally:
        session_id_var.reset(token)
