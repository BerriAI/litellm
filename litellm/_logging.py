import ast
import logging
import os
import re
import sys
from datetime import datetime
from logging import Formatter
from typing import Any, Dict, List, Optional

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

set_verbose = False

if set_verbose is True:
    logging.warning(
        "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
    )

_ENABLE_SECRET_REDACTION = (
    os.getenv("LITELLM_DISABLE_REDACT_SECRETS", "").lower() != "true"
)

_REDACTED = "REDACTED"


def _build_secret_patterns() -> re.Pattern:
    patterns: List[str] = [
        # AWS access key IDs
        r"(?:AKIA|ASIA)[0-9A-Z]{16}",
        # AWS secrets / session tokens / access key IDs (key=value)
        r"(?:aws_secret_access_key|aws_session_token|aws_access_key_id)"
        r"\s*[:=]\s*[A-Za-z0-9/+=]{20,}",
        # Bearer tokens (OAuth, JWT, etc.)
        r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
        # Basic auth headers
        r"Basic\s+[A-Za-z0-9+/]{10,}={0,2}",
        # OpenAI / Anthropic sk- prefixed keys
        r"sk-[A-Za-z0-9\-_]{20,}",
        # Generic api_key / api-key / apikey (handles 'key': 'value' dict repr)
        r"(?:api[_-]?key)['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]{8,}",
        # x-api-key / api-key header values (handles 'key': 'value' dict repr)
        r"(?:x-api-key|api-key)['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # Anthropic internal header keys
        r"x-ak-[A-Za-z0-9\-_]{20,}",
        # Google API keys
        r"AIza[0-9A-Za-z\-_]{35}",
        # Password / secret params (handles key=value and 'key': 'value')
        r"\w*(?:password|passwd|client_secret|secret_key|_secret)"
        r"['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # Database connection string credentials (scheme://user:pass@host)
        r"(?<=://)[^\s'\"]*:[^\s'\"@]+(?=@)",
        # Databricks personal access tokens
        r"dapi[0-9a-f]{32}",
        # ── Key-name-based redaction ──
        # Catches secrets inside dicts/config dumps by matching on the KEY name
        # regardless of what the value looks like.
        # e.g. 'master_key': 'any-value-here', "database_url": "postgres://..."
        r"(?:master_key|database_url|db_url|connection_string|"
        r"private_key|signing_key|encryption_key|"
        r"auth_token|access_token|refresh_token|"
        r"slack_webhook_url|webhook_url|"
        r"database_connection_string|"
        r"huggingface_token|jwt_secret)"
        r"""['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+""",
    ]
    return re.compile("|".join(patterns), re.IGNORECASE)


_SECRET_RE = _build_secret_patterns()


def _redact_string(value: str) -> str:
    return _SECRET_RE.sub(_REDACTED, value)


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
                record.exc_text = _redact_string(
                    self._formatter.formatException(record.exc_info)
                )
            except Exception:
                pass

        # Redact extra fields passed via logger.debug("msg", extra={...})
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_RECORD_ATTRS and isinstance(value, str):
                setattr(record, key, _redact_string(value))

        return True


_secret_filter = SecretRedactionFilter()


json_logs = bool(os.getenv("JSON_LOGS", False))
# Create a handler for the logger (you may need to adapt this based on your needs)
log_level = os.getenv("LITELLM_LOG", "DEBUG")
numeric_level: str = getattr(logging, log_level.upper())
handler = logging.StreamHandler()
handler.setLevel(numeric_level)
handler.addFilter(_secret_filter)


def _try_parse_json_message(message: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a log message as JSON. Returns parsed dict if valid, else None.
    Handles messages that are entirely valid JSON (e.g. json.dumps output).
    Uses shared safe_json_loads for consistent error handling.
    """
    if not message or not isinstance(message, str):
        return None
    msg_stripped = message.strip()
    if not (msg_stripped.startswith("{") or msg_stripped.startswith("[")):
        return None
    parsed = safe_json_loads(message, default=None)
    if parsed is None or not isinstance(parsed, dict):
        return None
    return parsed


def _try_parse_embedded_python_dict(message: str) -> Optional[Dict[str, Any]]:
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


_STANDARD_RECORD_ATTRS = _get_standard_record_attrs()


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def formatTime(self, record, datefmt=None):
        # Use datetime to format the timestamp in ISO 8601 format
        dt = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        message_str = record.getMessage()
        json_record: Dict[str, Any] = {
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

        if record.exc_info:
            json_record["stacktrace"] = record.exc_text or self.formatException(
                record.exc_info
            )

        return safe_dumps(json_record)


# Function to set up exception handlers for JSON logging
def _setup_json_exception_handlers(formatter):
    # Create a handler with JSON formatting for exceptions
    error_handler = logging.StreamHandler()
    error_handler.setFormatter(formatter)
    error_handler.addFilter(_secret_filter)

    # Setup excepthook for uncaught exceptions
    def json_excepthook(exc_type, exc_value, exc_traceback):
        record = logging.LogRecord(
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
            exception = context.get("exception")
            if exception:
                exc_type = type(exception)
                record = logging.LogRecord(
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
    formatter = logging.Formatter(
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
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Suppress APScheduler logging at INFO level
    apscheduler_executors_logger = logging.getLogger("apscheduler.executors.default")
    apscheduler_executors_logger.setLevel(logging.WARNING)
    apscheduler_scheduler_logger = logging.getLogger("apscheduler.scheduler")
    apscheduler_scheduler_logger.setLevel(logging.WARNING)


# Call the suppression function
_suppress_loggers()

ALL_LOGGERS = [
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

    loggers = list(ALL_LOGGERS)

    # Add langfuse logger if langfuse is being used as a callback
    langfuse_callbacks = {"langfuse", "langfuse_otel"}
    all_callbacks = set(litellm.success_callback + litellm.failure_callback)
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
    json_formatter_class = "litellm._logging.JsonFormatter"

    # Use the module-level log_level variable for consistency
    uvicorn_log_level = log_level.upper()

    log_config = {
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
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    _initialize_loggers_with_handler(handler)
    # Set up exception handlers
    _setup_json_exception_handlers(JsonFormatter())


def _turn_on_debug():
    verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
    verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
    verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug


def _disable_debugging():
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
            print(print_statement)  # noqa
    except Exception:
        pass


def _is_debugging_on() -> bool:
    """
    Returns True if debugging is on
    """
    return verbose_logger.isEnabledFor(logging.DEBUG) or set_verbose is True
