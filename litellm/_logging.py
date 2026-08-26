import ast
import contextvars
import logging
import os
import sys
from datetime import datetime
from logging import Formatter
from typing import Any, Final

import litellm
from litellm.constants import (
    LITELLM_TRUNCATED_PAYLOAD_FIELD,
    LITELLM_TRUNCATION_STDOUT_SAFEGUARD_NOTE,
    MAX_STRING_LENGTH_STDOUT_LOG,
)
from litellm.litellm_core_utils.env_utils import get_env_int
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.litellm_core_utils.secret_redaction import redact_string, redact_structured_value

set_verbose = False

session_id_var: Final[contextvars.ContextVar[str]] = contextvars.ContextVar("session_id", default="")
trace_id_var: Final[contextvars.ContextVar[str]] = contextvars.ContextVar("trace_id", default="")

_MAX_CORRELATION_ID_LENGTH: Final = 256


def _sanitize_correlation_id(value: str) -> str:
    """Strip control characters, bound length, and redact credential-shaped
    content before a caller-controlled trace_id/session_id (e.g.
    litellm_session_id, x-litellm-trace-id) is stamped into log lines.

    Without the first two, a caller could embed \\r/\\n or terminal escape
    sequences to forge fake log entries, or submit an oversized value repeated
    across every log line for the request. Without the redaction, a caller
    could smuggle a real credential (e.g. an sk-... key) through this field:
    CorrelationContextFilter stamps trace_id/session_id onto the record after
    SecretRedactionFilter has already run, so those two fields never otherwise
    pass through credential redaction.
    """
    stripped: Final = "".join(ch for ch in value if ch.isprintable())
    return _redact_string(stripped[:_MAX_CORRELATION_ID_LENGTH])


def set_session_id(session_id: str) -> "contextvars.Token[str]":
    return session_id_var.set(_sanitize_correlation_id(session_id))


def set_trace_id(trace_id: str) -> "contextvars.Token[str]":
    return trace_id_var.set(_sanitize_correlation_id(trace_id))


if set_verbose is True:
    logging.warning(
        "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
    )

_ENABLE_SECRET_REDACTION: Final = os.getenv("LITELLM_DISABLE_REDACT_SECRETS", "").lower() != "true"


def _redact_string(value: str) -> str:
    if not _ENABLE_SECRET_REDACTION:
        return value
    return redact_string(value)


def _redact_structured_value(key: str | None, value: str) -> str:
    if not _ENABLE_SECRET_REDACTION:
        return value
    return redact_structured_value(key, value)


def redact_secrets(value: str) -> str:
    """Public API: redact known secret/credential patterns from an arbitrary string.

    Use this for code paths that bypass the logging system — e.g. Slack/Teams
    alerting, HTTP error response bodies, or any other string that may contain
    secrets and will be sent to an external sink.

    Not to be confused with redact_message_input_output_from_logging() in
    litellm_core_utils/redact_messages.py, which redacts LLM prompt/response
    content for privacy — this function redacts credential patterns (API keys,
    PEM blocks, tokens, etc.) by shape.
    """
    if not _ENABLE_SECRET_REDACTION:
        return value
    return _redact_string(value)


def _substituted_color_message(record: logging.LogRecord) -> str | None:
    """Render a record's ``color_message`` against its args, or None if absent.

    uvicorn's colorized formatter re-renders `color_message` against
    record.args at emit time (see uvicorn.logging.ColourizedFormatter) instead
    of using the already-formatted record.msg, so it has to be substituted
    before args are cleared or it is later formatted with no args and prints
    the raw "%s://%s:%d" placeholders instead of the URL.
    """
    color_message: Final = record.__dict__.get("color_message")
    if not isinstance(color_message, str) or not record.args:
        return None
    try:
        return color_message % record.args
    except TypeError:
        return color_message


class SecretRedactionFilter(logging.Filter):
    """Scrubs known secret/credential patterns from log records."""

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        if not _ENABLE_SECRET_REDACTION:
            return True

        # Runs before args are cleared, and before the extra-field loop below
        # that redacts the substituted result.
        substituted_color_message: Final = _substituted_color_message(record)
        if substituted_color_message is not None:
            record.color_message = substituted_color_message  # rebind-ok: a Filter scrubs records in place

        try:
            record.msg = _redact_string(record.getMessage())
            record.args = None
        except Exception:
            if isinstance(record.msg, str):
                record.msg = _redact_string(record.msg)

        # Redact exception tracebacks
        if record.exc_info and record.exc_info[1] is not None:
            try:
                record.exc_text = _redact_string(record.exc_text or self._formatter.formatException(record.exc_info))
            except Exception:
                pass

        # Redact extra fields passed via logger.debug("msg", extra={...})
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_RECORD_ATTRS and isinstance(value, str):
                setattr(record, key, _redact_string(value))

        return True


_secret_filter: Final = SecretRedactionFilter()


def _get_max_string_length_stdout_log() -> int:
    """Read the limit per record so a value loaded later via proxy config
    environment_variables is honored."""
    return get_env_int("MAX_STRING_LENGTH_STDOUT_LOG", MAX_STRING_LENGTH_STDOUT_LOG)


def _stdout_truncation_marker(skipped_chars: int) -> str:
    return (
        f"... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped {skipped_chars} chars. "
        f"{LITELLM_TRUNCATION_STDOUT_SAFEGUARD_NOTE}) ..."
    )


def _truncate_for_stdout_log(text: str, limit: int) -> str:
    kept_chars: Final = limit - len(_stdout_truncation_marker(len(text)))
    if kept_chars <= 0:
        return text[:limit]
    head_chars: Final = kept_chars // 2
    tail_chars: Final = kept_chars - head_chars
    return f"{text[:head_chars]}{_stdout_truncation_marker(len(text) - kept_chars)}{text[-tail_chars:]}"


class StdoutLogTruncationFilter(logging.Filter):
    """Bounds how much of an oversized log line reaches stdout.

    A provider error string can echo the whole request payload, so one failed agentic
    request writes hundreds of KB to stdout, repeatedly as the exception propagates from
    the router to the proxy handler and into its traceback, all inline on the event loop.

    DEBUG records pass through untouched, since dumping full payloads is the point of
    `--detailed_debug`, and logging callbacks (OTEL, Datadog, etc.) don't run through
    logging filters at all, so they still get the untruncated error.
    """

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.INFO:
            return True

        limit: Final = _get_max_string_length_stdout_log()
        if limit <= 0:
            return True

        try:
            message: Final = record.getMessage()
        except (TypeError, ValueError):
            return True

        if len(message) > limit:
            record.msg = _truncate_for_stdout_log(message, limit)  # rebind-ok: the Filter interface mutates the record
            record.args = None  # rebind-ok: args are consumed by the truncated message above

        if isinstance(record.exc_info, tuple):
            exc_text: Final = record.exc_text or self._formatter.formatException(record.exc_info)
            if len(exc_text) > limit:
                record.exc_text = _truncate_for_stdout_log(  # rebind-ok: the Filter interface mutates the record
                    exc_text, limit
                )

        return True


_stdout_truncation_filter: Final = StdoutLogTruncationFilter()


class CorrelationContextFilter(logging.Filter):
    """Stamps each log record with the current request's trace_id and session_id from contextvars.

    Works in tandem with JsonFormatter: the formatter's record.__dict__ loop picks up these
    attributes as first-class JSON fields without any formatter-level code.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not litellm.request_correlation_in_logs:
            return True
        trace_id: Final = trace_id_var.get()
        if trace_id:
            record.trace_id = trace_id  # rebind-ok: stamping the LogRecord is the Filter interface's contract
        session_id: Final = session_id_var.get()
        if session_id:
            record.session_id = session_id  # rebind-ok: stamping the LogRecord is the Filter interface's contract
        return True


_correlation_filter: Final = CorrelationContextFilter()


json_logs = bool(os.getenv("JSON_LOGS", False))
# Create a handler for the logger (you may need to adapt this based on your needs)
log_level: Final = os.getenv("LITELLM_LOG", "DEBUG")
numeric_level: Final[str] = getattr(logging, log_level.upper())
handler: Final = logging.StreamHandler()
handler.setLevel(numeric_level)
handler.addFilter(_secret_filter)
handler.addFilter(_correlation_filter)


def _try_parse_json_message(message: str) -> dict[str, Any] | None:
    """
    Try to parse a log message as JSON. Returns parsed dict if valid, else None.
    Handles messages that are entirely valid JSON (e.g. json.dumps output).
    Uses shared safe_json_loads for consistent error handling.
    """
    if not message or not isinstance(message, str):
        return None
    msg_stripped: Final = message.strip()
    if not (msg_stripped.startswith("{") or msg_stripped.startswith("[")):
        return None
    parsed: Final = safe_json_loads(message, default=None)
    if parsed is None or not isinstance(parsed, dict):
        return None
    return parsed


def _try_parse_embedded_python_dict(message: str) -> dict[str, Any] | None:
    """
    Try to find and parse a Python dict repr (e.g. str(d) or repr(d)) embedded in
    the message. Handles patterns like:
    "get_available_deployment for model: X, Selected deployment: {'model_name': '...', ...} for model: X"
    Uses ast.literal_eval for safe parsing. Returns the parsed dict or None.
    """
    if not message or not isinstance(message, str) or "{" not in message:
        return None
    i = 0
    while i < len(message):
        start = message.find("{", i)
        if start == -1:
            break
        depth = 0
        for j in range(start, len(message)):
            c = message[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    substr = message[start : j + 1]
                    try:
                        result = ast.literal_eval(substr)
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                    except (ValueError, SyntaxError, TypeError):
                        pass
                    break
        i = start + 1
    return None


# Standard LogRecord attribute names - used to identify 'extra' fields.
# Derived at runtime so we automatically include version-specific attrs (e.g. taskName).
def _get_standard_record_attrs() -> frozenset:
    """Standard LogRecord attribute names - excludes extra keys from logger.debug(..., extra={...})."""
    return frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


_STANDARD_RECORD_ATTRS: Final = _get_standard_record_attrs()

# CorrelationContextFilter is the only legitimate source for these two JSON fields;
# see JsonFormatter.format() for why they're excluded from the generic message-content
# and extra-attribute promotion paths.
_RESERVED_CORRELATION_FIELDS: Final = frozenset(("trace_id", "session_id"))


class JsonFormatter(Formatter):
    def __init__(self):
        super().__init__()

    def formatTime(self, record, datefmt=None):
        # Use datetime to format the timestamp in ISO 8601 format
        dt: Final = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        message_str: Final = record.getMessage()
        json_record: Final[dict[str, Any]] = {
            "message": message_str,
            "level": record.levelname,
            "timestamp": self.formatTime(record),
        }

        # Parse embedded JSON or Python dict repr in message so sub-fields become first-class properties.
        # trace_id/session_id are excluded here unconditionally (not just "if not already
        # set") - CorrelationContextFilter is the only legitimate source for these two
        # fields, and a message that merely happens to parse as JSON/dict (e.g. a proxy
        # log line dumping raw request headers) must never be able to claim them, even on
        # a record the filter hasn't stamped yet (no correlation context active for it).
        parsed = _try_parse_json_message(message_str)
        if parsed is None:
            parsed = _try_parse_embedded_python_dict(message_str)
        if parsed is not None:
            for key, value in parsed.items():
                if key not in json_record and key not in _RESERVED_CORRELATION_FIELDS:
                    json_record[key] = value

        # Include extra attributes passed via logger.debug("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in json_record:
                json_record[key] = value

        # trace_id/session_id are reserved: CorrelationContextFilter is the only
        # legitimate source for these two fields. Without this, a message string
        # that happens to parse as JSON/dict (e.g. a proxy log line dumping raw
        # request headers) with a "trace_id"/"session_id" key would have already
        # claimed the key at the parsed-message step above, and the extra-attributes
        # loop's "key not in json_record" guard would then skip the real value -
        # letting a caller-supplied header spoof another request's correlation ids.
        for reserved_key in _RESERVED_CORRELATION_FIELDS:
            value = getattr(record, reserved_key, None)
            if value:
                json_record[reserved_key] = value

        # Set component/logger only if not already supplied via extra={...}
        if "component" not in json_record:
            json_record["component"] = record.name
        if "logger" not in json_record:
            json_record["logger"] = f"{record.filename}:{record.lineno}"

        if record.exc_info:
            json_record["stacktrace"] = record.exc_text or self.formatException(record.exc_info)

        return safe_dumps(json_record, value_transform=_redact_structured_value)


class CorrelationPlainFormatter(logging.Formatter):
    """Appends trace_id/session_id to plain-text log lines stamped by CorrelationContextFilter.

    Mirrors JsonFormatter's handling of these two fields so request_correlation_in_logs
    behaves the same whether or not json_logs is enabled.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted: Final = _redact_string(super().format(record))
        trace_id: Final = getattr(record, "trace_id", None)
        session_id: Final = getattr(record, "session_id", None)
        if not trace_id and not session_id:
            return formatted
        parts: Final = tuple(
            p
            for p in (f"trace_id={trace_id}" if trace_id else None, f"session_id={session_id}" if session_id else None)
            if p
        )
        return f"{formatted} [{' '.join(parts)}]"


# Function to set up exception handlers for JSON logging
def _setup_json_exception_handlers(formatter):
    # Create a handler with JSON formatting for exceptions
    error_handler: Final = logging.StreamHandler()
    error_handler.setFormatter(formatter)
    error_handler.addFilter(_secret_filter)
    error_handler.addFilter(_stdout_truncation_filter)
    error_handler.addFilter(_correlation_filter)

    # Setup excepthook for uncaught exceptions
    def json_excepthook(exc_type, exc_value, exc_traceback):
        record: Final = logging.LogRecord(
            name="LiteLLM",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=str(exc_value),
            args=(),
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        error_handler.handle(record)

    sys.excepthook = json_excepthook

    # Configure asyncio exception handler if possible
    try:
        import asyncio

        def async_json_exception_handler(loop, context):
            exception: Final = context.get("exception")
            if exception:
                exc_type: Final = type(exception)
                record: Final = logging.LogRecord(
                    name="LiteLLM",
                    level=logging.ERROR,
                    pathname="",
                    lineno=0,
                    msg=str(exception),
                    args=(),
                    exc_info=(exc_type, exception, exception.__traceback__),
                )
                error_handler.handle(record)
            else:
                loop.default_exception_handler(context)

        asyncio.get_event_loop().set_exception_handler(async_json_exception_handler)
    except Exception:
        pass


# Create a formatter and set it for the handler
if json_logs:
    handler.setFormatter(JsonFormatter())
    _setup_json_exception_handlers(JsonFormatter())
else:
    formatter: Final = CorrelationPlainFormatter(
        "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)

verbose_proxy_logger = logging.getLogger("LiteLLM Proxy")
verbose_router_logger = logging.getLogger("LiteLLM Router")
verbose_logger = logging.getLogger("LiteLLM")

# Add the handler to the loggers
verbose_router_logger.addHandler(handler)
verbose_proxy_logger.addHandler(handler)
verbose_logger.addHandler(handler)

# Filters attached to the logger, not the handler, survive callers swapping in their own
# handlers (JSON mode, uvicorn log config, a host app's root handler).
verbose_router_logger.addFilter(_stdout_truncation_filter)
verbose_proxy_logger.addFilter(_stdout_truncation_filter)
verbose_logger.addFilter(_stdout_truncation_filter)


def _suppress_loggers():
    """Suppress noisy loggers at INFO level"""
    # Suppress httpx request logging at INFO level
    httpx_logger: Final = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Suppress APScheduler logging at INFO level
    apscheduler_executors_logger: Final = logging.getLogger("apscheduler.executors.default")
    apscheduler_executors_logger.setLevel(logging.WARNING)
    apscheduler_scheduler_logger: Final = logging.getLogger("apscheduler.scheduler")
    apscheduler_scheduler_logger.setLevel(logging.WARNING)


_REDACTED_THIRD_PARTY_LOGGERS: Final[tuple[str, ...]] = (
    "apscheduler.executors.default",
    "apscheduler.scheduler",
    "asyncio",
    "backoff",
    "httpx",
    "uvicorn.error",
)


def _redact_third_party_loggers() -> None:
    """Extend secret redaction to records litellm does not emit directly.

    litellm's own loggers are covered by the filter on their shared handler, but a
    litellm value can also reach a log record through a dependency that logs on its
    own logger. Those records never pass through a litellm handler.

    The filter is attached to each emitting logger rather than to the root logger or
    to root's handlers. `Logger.handle` applies the emitting logger's filters before
    any handler runs, so redaction happens once, at the earliest point in the
    record's life, and covers every downstream handler regardless of who owns it.
    The alternatives do not hold: `callHandlers` consults ancestors for handlers but
    never for filters, so a filter on the root logger never sees these records at
    all, and a filter on a root handler only covers that one handler, leaving
    handlers registered earlier or on the emitting logger itself untouched.

    Each name is the exact logger a dependency emits on; a parent name would not
    cover its children, for the same reason the root logger does not.
    """
    for name in _REDACTED_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).addFilter(_secret_filter)


# Call the suppression function
_suppress_loggers()
_redact_third_party_loggers()

ALL_LOGGERS: Final = [
    logging.getLogger(),
    verbose_logger,
    verbose_router_logger,
    verbose_proxy_logger,
]


def _get_loggers_to_initialize():
    """
    Get all loggers that should be initialized with the JSON handler.

    Includes third-party integration loggers (like langfuse) if they are
    configured as callbacks.
    """
    import litellm

    loggers: Final = list(ALL_LOGGERS)

    # Add langfuse logger if langfuse is being used as a callback
    langfuse_callbacks: Final = {"langfuse", "langfuse_otel"}
    all_callbacks: Final = set(litellm.success_callback + litellm.failure_callback)
    if langfuse_callbacks & all_callbacks:
        loggers.append(logging.getLogger("langfuse"))

    return loggers


def _initialize_loggers_with_handler(handler: logging.Handler):
    """
    Initialize all loggers with a handler

    - Adds a handler to each logger
    - Prevents bubbling to parent/root (critical to prevent duplicate JSON logs)
    """
    handler.addFilter(_secret_filter)
    handler.addFilter(_correlation_filter)
    for lg in _get_loggers_to_initialize():
        lg.handlers.clear()  # remove any existing handlers
        lg.addHandler(handler)  # add JSON formatter handler
        lg.propagate = False  # prevent bubbling to parent/root


def _get_uvicorn_json_log_config():
    """
    Generate a uvicorn log_config dictionary that applies JSON formatting to all loggers.

    This ensures that uvicorn's access logs, error logs, and all application logs
    are formatted as JSON when json_logs is enabled.
    """
    json_formatter_class: Final = "litellm._logging.JsonFormatter"

    # Use the module-level log_level variable for consistency
    uvicorn_log_level: Final = log_level.upper()

    log_config: Final = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": json_formatter_class,
            },
            "default": {
                "()": json_formatter_class,
            },
            "access": {
                "()": json_formatter_class,
            },
        },
        "handlers": {
            "default": {
                "formatter": "json",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": uvicorn_log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": uvicorn_log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": uvicorn_log_level,
                "propagate": False,
            },
        },
    }

    return log_config


def _turn_on_json():
    """
    Turn on JSON logging

    - Adds a JSON formatter to all loggers
    """
    handler: Final = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    _initialize_loggers_with_handler(handler)
    # Set up exception handlers
    _setup_json_exception_handlers(JsonFormatter())


def _turn_on_debug():
    verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
    verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
    verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug


def _disable_debugging():
    """Disable the package, router, and proxy verbose loggers."""
    verbose_logger.disabled = True
    verbose_router_logger.disabled = True
    verbose_proxy_logger.disabled = True


def _enable_debugging():
    verbose_logger.disabled = False
    verbose_router_logger.disabled = False
    verbose_proxy_logger.disabled = False


def print_verbose(print_statement):
    try:
        if set_verbose:
            print(redact_secrets(str(print_statement)))  # noqa: T201
    except Exception:
        pass


def _is_debugging_on() -> bool:
    """
    Returns True if debugging is on
    """
    return verbose_logger.isEnabledFor(logging.DEBUG) or set_verbose is True
