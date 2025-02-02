import json
import logging
import os
import re
from datetime import datetime
from logging import Formatter

set_verbose = False

if set_verbose is True:
    logging.warning(
        "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
    )


class SensitiveMaskMixin:
    """Mixin class that provides sensitive data masking functionality"""

    SENSITIVE_PATTERNS = [
        (
            r'(api[_-]?key|apikey|api[_-]?token|access[_-]?token)(["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-\.]+)',
            r"\1\2*****",
        ),
        (
            r"Authorization:\s*Bearer\s+([a-zA-Z0-9_\-\.]+)",
            r"Authorization: Bearer *****",
        ),
        (r'(password|secret)(["\']?\s*[:=]\s*["\']?)([^"\'\s]+)', r"\1\2*****"),
        # Add OpenAI specific patterns
        (r"(sk-[a-zA-Z0-9]{48})", r"sk-****"),
        (r"(org-[a-zA-Z0-9]{24})", r"org-****"),
    ]

    def mask_sensitive_data(self, text):
        """Mask sensitive data in the given text"""
        if not isinstance(text, str):
            return text

        masked = text
        for pattern, repl in self.SENSITIVE_PATTERNS:
            masked = re.sub(pattern, repl, masked, flags=re.IGNORECASE)
        return masked


class JsonFormatter(Formatter, SensitiveMaskMixin):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        # Mask sensitive data in the message
        record.msg = self.mask_sensitive_data(record.msg)

        json_record = {
            "message": record.getMessage(),
            "level": record.levelname,
            "timestamp": self.formatTime(record),
        }

        if record.exc_info:
            json_record["stacktrace"] = self.formatException(record.exc_info)

        return json.dumps(json_record)


class ColoredFormatter(Formatter, SensitiveMaskMixin):
    def format(self, record):
        # Mask sensitive data in the message
        record.msg = self.mask_sensitive_data(record.msg)

        return super().format(record)


json_logs = bool(os.getenv("JSON_LOGS", False))
log_level = os.getenv("LITELLM_LOG", "DEBUG")
numeric_level = getattr(logging, log_level.upper())
handler = logging.StreamHandler()
handler.setLevel(numeric_level)

# Create a formatter and set it for the handler
if json_logs:
    handler.setFormatter(JsonFormatter())
else:
    formatter = ColoredFormatter(
        "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

verbose_proxy_logger = logging.getLogger("LiteLLM Proxy")
verbose_router_logger = logging.getLogger("LiteLLM Router")
verbose_logger = logging.getLogger("LiteLLM")

# Add the handler to the logger
verbose_router_logger.addHandler(handler)
verbose_proxy_logger.addHandler(handler)
verbose_logger.addHandler(handler)


def _turn_on_json():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    loggers = [verbose_router_logger, verbose_proxy_logger, verbose_logger]

    for logger in loggers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.addHandler(handler)


def _turn_on_debug():
    verbose_logger.setLevel(level=logging.DEBUG)
    verbose_router_logger.setLevel(level=logging.DEBUG)
    verbose_proxy_logger.setLevel(level=logging.DEBUG)


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
            masked_statement = JsonFormatter().mask_sensitive_data(str(print_statement))
            print(masked_statement)  # noqa
    except Exception:
        pass


def _is_debugging_on() -> bool:
    """
    Returns True if debugging is on
    """
    if verbose_logger.isEnabledFor(logging.DEBUG) or set_verbose is True:
        return True
    return False
