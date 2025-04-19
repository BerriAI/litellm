import json
import logging
import os
import re
import sys
from datetime import datetime
from logging import Formatter

set_verbose = False

if set_verbose is True:
    logging.warning(
        "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
    )
json_logs = bool(os.getenv("JSON_LOGS", False))
# Create a handler for the logger (you may need to adapt this based on your needs)
log_level = os.getenv("LITELLM_LOG", "DEBUG")
numeric_level: str = getattr(logging, log_level.upper())
handler = logging.StreamHandler()
handler.setLevel(numeric_level)


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def formatTime(self, record, datefmt=None):
        # Use datetime to format the timestamp in ISO 8601 format
        dt = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        json_record = {
            "message": record.getMessage(),
            "level": record.levelname,
            "timestamp": self.formatTime(record),
        }

        if record.exc_info:
            json_record["stacktrace"] = self.formatException(record.exc_info)

        return json.dumps(json_record)


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""

    SENSITIVE_KEYS = [
        "credentials",
        "api_key",
        "key",
        "api_base",
        "password",
        "secret",
        "token",
        "private_key",  # Added for nested JSON case
    ]

    def filter(self, record):
        if not hasattr(record, "msg") or not record.msg:
            return True

        # If the message is a format string with args, we need to format it first
        if record.args:
            msg = record.msg % record.args
        else:
            msg = str(record.msg)

        # Redact sensitive information
        for key in self.SENSITIVE_KEYS:
            # Create patterns for compound keys (e.g., openai_api_key)
            key_pattern = f"[a-zA-Z0-9_/\\\\-]*{key}[a-zA-Z0-9_/\\\\-]*"

            # Handle JSON-like strings with double quotes
            json_pattern = f'"({key_pattern})":\\s*"[^"]*"'
            msg = re.sub(json_pattern, r'"\1": "REDACTED"', msg, flags=re.IGNORECASE)

            # Handle dictionary-like strings with single quotes
            dict_pattern = f"'({key_pattern})':\\s*'[^']*'"
            msg = re.sub(dict_pattern, r"'\1': 'REDACTED'", msg, flags=re.IGNORECASE)

            # Handle mixed quote styles
            mixed_pattern = f"\"({key_pattern})\":\\s*'[^']*'"
            msg = re.sub(mixed_pattern, r'"\1": \'REDACTED\'', msg, flags=re.IGNORECASE)

            # Handle key-value pairs in plain text
            # Convert snake_case and special characters to flexible matching
            display_key = key.replace("_", "[-_ ]")
            # Match both original and display versions of the key, preserving the separator and spacing
            plain_pattern = (
                f"\\b({key_pattern}|{display_key})\\s*([:=])\\s*[^,\\s][^,]*"
            )
            msg = re.sub(
                plain_pattern,
                lambda m: f"{m.group(1)}{m.group(2)}{' ' if m.group(2) == ':' else ''}REDACTED",
                msg,
                flags=re.IGNORECASE,
            )

            # Handle mixed quotes without escaping
            msg = msg.replace('\\"', '"').replace("\\'", "'")

        # Set the message and clear args since we've already formatted it
        record.msg = msg
        record.args = None
        return True


# Function to set up exception handlers for JSON logging
def _setup_json_exception_handlers(formatter):
    # Create a handler with JSON formatting for exceptions
    error_handler = logging.StreamHandler()
    error_handler.setFormatter(formatter)

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
                record = logging.LogRecord(
                    name="LiteLLM",
                    level=logging.ERROR,
                    pathname="",
                    lineno=0,
                    msg=str(exception),
                    args=(),
                    exc_info=None,
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

# Add the sensitive data filter to all loggers
sensitive_filter = SensitiveDataFilter()
verbose_router_logger.addFilter(sensitive_filter)
verbose_proxy_logger.addFilter(sensitive_filter)
verbose_logger.addFilter(sensitive_filter)

# Add the handler to the logger
verbose_router_logger.addHandler(handler)
verbose_proxy_logger.addHandler(handler)
verbose_logger.addHandler(handler)


def _turn_on_json():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    # Define all loggers to update, including root logger
    loggers = [logging.getLogger()] + [
        verbose_router_logger,
        verbose_proxy_logger,
        verbose_logger,
    ]

    # Iterate through each logger and update its handlers
    for logger in loggers:
        # Remove all existing handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # Add the new handler
        logger.addHandler(handler)

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
    if verbose_logger.isEnabledFor(logging.DEBUG) or set_verbose is True:
        return True
    return False
