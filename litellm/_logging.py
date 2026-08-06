import ast
import logging
import os
import sys
from datetime import datetime
from logging import Formatter
from typing import Any, Final

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.litellm_core_utils.secret_redaction import redact_string

set_verbose = False

if set_verbose is True:
    logging.warning(
        "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
    )

_ENABLE_SECRET_REDACTION: Final = os.getenv("LITELLM_DISABLE_REDACT_SECRETS", "").lower() != "true"


def _redact_string(value: str) -> str:
    if not _ENABLE_SECRET_REDACTION:
        return value
    return redact_string(value)


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


class SecretRedactionFilter(logging.Filter):
    """Scrubs known secret/credential patterns from log records."""

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        if not _ENABLE_SECRET_REDACTION:
            return True

        try:
            record.msg = _redact_string(record.getMessage())
            record.args = None
        except Exception:
            if isinstance(record.msg, str):
                record.msg = _redact_string(record.msg)

        # Redact exception tracebacks
        if record.exc_info and record.exc_info[1] is not None:
            try:
                record.exc_text = _redact_string(self._formatter.formatException(record.exc_info))
            except Exception:
                pass

        # Redact extra fields passed via logger.debug("msg", extra={...})
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_RECORD_ATTRS and isinstance(value, str):
                setattr(record, key, _redact_string(value))

        return True


_secret_filter: Final = SecretRedactionFilter()


json_logs = bool(os.getenv("JSON_LOGS", False))
# Create a handler for the logger (you may need to adapt this based on your needs)
log_level: Final = os.getenv("LITELLM_LOG", "DEBUG")
numeric_level: Final[str] = getattr(logging, log_level.upper())
handler: Final = logging.StreamHandler()
handler.setLevel(numeric_level)
handler.addFilter(_secret_filter)


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

        # Parse embedded JSON or Python dict repr in message so sub-fields become first-class properties
        parsed = _try_parse_json_message(message_str)
        if parsed is None:
            parsed = _try_parse_embedded_python_dict(message_str)
        if parsed is not None:
            for key, value in parsed.items():
                if key not in json_record:
                    json_record[key] = value

        # Include extra attributes passed via logger.debug("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in json_record:
                json_record[key] = value

        # Set component/logger only if not already supplied via extra={...}
        if "component" not in json_record:
            json_record["component"] = record.name
        if "logger" not in json_record:
            json_record["logger"] = f"{record.filename}:{record.lineno}"

        if record.exc_info:
            json_record["stacktrace"] = record.exc_text or self.formatException(record.exc_info)

        return safe_dumps(json_record)


# Function to set up exception handlers for JSON logging
def _setup_json_exception_handlers(formatter):
    # Create a handler with JSON formatting for exceptions
    error_handler: Final = logging.StreamHandler()
    error_handler.setFormatter(formatter)
    error_handler.addFilter(_secret_filter)

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
    formatter: Final = logging.Formatter(
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
