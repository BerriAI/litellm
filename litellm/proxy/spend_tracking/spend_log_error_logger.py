"""
Logging helpers for spend-tracking error paths.

Proxy operators have asked for a way to keep both their downstream log sinks
and the SpendLogs UI free of the stack traces that the spend-tracking
machinery emits when it hits 4xx/5xx or transient DB errors. The errors still
need to be logged (and still flow to Sentry via
``proxy_logging_obj.failure_handler``), but the multi-line stack traces
dominate log volume and clutter the per-row Metadata pane in the UI.

The opt-in is a single env var, ``LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS=true``,
gated by ``should_suppress_spend_log_tracebacks``. When it returns ``True``:
  * ``spend_log_error`` drops the traceback from the console / structured log
    record (this module), and
  * the failure callback in ``proxy_track_cost_callback`` drops the
    ``error_information.traceback`` field from the SpendLogs row before it is
    persisted, so the UI's per-row Metadata pane (which renders the metadata
    JSON verbatim) stays clean. The key is omitted entirely rather than set
    to ``""`` — ``StandardLoggingPayloadErrorInformation`` marks the field
    optional and every downstream consumer uses ``.get("traceback")``.

At DEBUG the full traceback is always preserved so operators can still
troubleshoot. The UI suppression follows the same gate.
"""

import logging
import os
from typing import Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.main import str_to_bool

SUPPRESS_SPEND_LOG_TRACEBACKS_ENV = "LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS"


def _is_suppression_env_enabled() -> bool:
    """Read the opt-in env var fresh each call so dynamic flips are honored.

    Kept separate from ``should_suppress_spend_log_tracebacks`` so tests and
    other call sites can introspect just the env-var state without also
    consulting the live logger level.
    """
    return str_to_bool(os.getenv(SUPPRESS_SPEND_LOG_TRACEBACKS_ENV)) is True


def should_suppress_spend_log_tracebacks() -> bool:
    """Return ``True`` when spend-log traceback suppression should apply.

    Suppression only kicks in when both:
      * the operator opted in via the env var, and
      * the proxy logger is at INFO or above (i.e. not DEBUG) — at DEBUG we
        still want full tracebacks for troubleshooting.
    """
    if not _is_suppression_env_enabled():
        return False
    return not verbose_proxy_logger.isEnabledFor(logging.DEBUG)


def spend_log_error(
    message: str,
    *args: Any,
    exc: Optional[BaseException] = None,
) -> None:
    """Log a spend-tracking error, with the traceback gated on the env var.

    By default this behaves like ``verbose_proxy_logger.exception`` — the
    active exception (or ``exc`` if supplied) is attached so the formatter
    renders its traceback. When ``LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS`` is
    truthy and the logger is at INFO or above, the traceback is dropped and
    only ``message % args`` is emitted.

    Sentry / ``proxy_logging_obj.failure_handler`` is NOT invoked here — call
    sites still own the alerting path. This helper is purely about console /
    structured-log output volume.
    """
    if should_suppress_spend_log_tracebacks():
        verbose_proxy_logger.error(message, *args)
        return

    if exc is not None:
        verbose_proxy_logger.error(
            message, *args, exc_info=(type(exc), exc, exc.__traceback__)
        )
    else:
        verbose_proxy_logger.error(message, *args, exc_info=True)
