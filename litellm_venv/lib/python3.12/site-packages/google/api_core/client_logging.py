import logging
import json
import os

from typing import List, Optional

_LOGGING_INITIALIZED = False
_BASE_LOGGER_NAME = "google"

# Fields to be included in the StructuredLogFormatter.
#
# TODO(https://github.com/googleapis/python-api-core/issues/761): Update this list to support additional logging fields.
_recognized_logging_fields = [
    "httpRequest",
    "rpcName",
    "serviceName",
    "credentialsType",
    "credentialsInfo",
    "universeDomain",
    "request",
    "response",
    "metadata",
    "retryAttempt",
    "httpResponse",
]  # Additional fields to be Logged.


def logger_configured(logger) -> bool:
    """Determines whether `logger` has non-default configuration

    Args:
      logger: The logger to check.

    Returns:
      bool: Whether the logger has any non-default configuration.
    """
    return (
        logger.handlers != [] or logger.level != logging.NOTSET or not logger.propagate
    )


def initialize_logging():
    """Initializes "google" loggers, partly based on the environment variable

    Initializes the "google" logger and any loggers (at the "google"
    level or lower) specified by the environment variable
    GOOGLE_SDK_PYTHON_LOGGING_SCOPE, as long as none of these loggers
    were previously configured. If any such loggers (including the
    "google" logger) are initialized, they are set to NOT propagate
    log events up to their parent loggers.

    This initialization is executed only once, and hence the
    environment variable is only processed the first time this
    function is called.
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    scopes = os.getenv("GOOGLE_SDK_PYTHON_LOGGING_SCOPE", "")
    setup_logging(scopes)
    _LOGGING_INITIALIZED = True


def parse_logging_scopes(scopes: Optional[str] = None) -> List[str]:
    """Returns a list of logger names.

    Splits the single string of comma-separated logger names into a list of individual logger name strings.

    Args:
      scopes: The name of a single logger. (In the future, this will be a comma-separated list of multiple loggers.)

    Returns:
      A list of all the logger names in scopes.
    """
    if not scopes:
        return []
    # TODO(https://github.com/googleapis/python-api-core/issues/759): check if the namespace is a valid namespace.
    # TODO(b/380481951): Support logging multiple scopes.
    # TODO(b/380483756): Raise or log a warning for an invalid scope.
    namespaces = [scopes]
    return namespaces


def configure_defaults(logger):
    """Configures `logger` to emit structured info to stdout."""
    if not logger_configured(logger):
        console_handler = logging.StreamHandler()
        logger.setLevel("DEBUG")
        logger.propagate = False
        formatter = StructuredLogFormatter()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def setup_logging(scopes: str = ""):
    """Sets up logging for the specified `scopes`.

    If the loggers specified in `scopes` have not been previously
    configured, this will configure them to emit structured log
    entries to stdout, and to not propagate their log events to their
    parent loggers. Additionally, if the "google" logger (whether it
    was specified in `scopes` or not) was not previously configured,
    it will also configure it to not propagate log events to the root
    logger.

    Args:
      scopes: The name of a single logger. (In the future, this will be a comma-separated list of multiple loggers.)

    """

    # only returns valid logger scopes (namespaces)
    # this list has at most one element.
    logger_names = parse_logging_scopes(scopes)

    for namespace in logger_names:
        # This will either create a module level logger or get the reference of the base logger instantiated above.
        logger = logging.getLogger(namespace)

        # Configure default settings.
        configure_defaults(logger)

    # disable log propagation at base logger level to the root logger only if a base logger is not already configured via code changes.
    base_logger = logging.getLogger(_BASE_LOGGER_NAME)
    if not logger_configured(base_logger):
        base_logger.propagate = False


# TODO(https://github.com/googleapis/python-api-core/issues/763): Expand documentation.
class StructuredLogFormatter(logging.Formatter):
    # TODO(https://github.com/googleapis/python-api-core/issues/761): ensure that additional fields such as
    # function name, file name, and line no. appear in a log output.
    def format(self, record: logging.LogRecord):
        log_obj = {
            "timestamp": self.formatTime(record),
            "severity": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        for field_name in _recognized_logging_fields:
            value = getattr(record, field_name, None)
            if value is not None:
                log_obj[field_name] = value
        return json.dumps(log_obj)
