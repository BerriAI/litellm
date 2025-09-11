import sys
from functools import wraps
from threading import Thread, current_thread

import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import use_isolation_scope, use_scope
from sentry_sdk.utils import (
    event_from_exception,
    capture_internal_exceptions,
    logger,
    reraise,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import TypeVar
    from typing import Callable
    from typing import Optional

    from sentry_sdk._types import ExcInfo

    F = TypeVar("F", bound=Callable[..., Any])


class ThreadingIntegration(Integration):
    identifier = "threading"

    def __init__(self, propagate_hub=None, propagate_scope=True):
        # type: (Optional[bool], bool) -> None
        if propagate_hub is not None:
            logger.warning(
                "Deprecated: propagate_hub is deprecated. This will be removed in the future."
            )

        # Note: propagate_hub did not have any effect on propagation of scope data
        # scope data was always propagated no matter what the value of propagate_hub was
        # This is why the default for propagate_scope is True

        self.propagate_scope = propagate_scope

        if propagate_hub is not None:
            self.propagate_scope = propagate_hub

    @staticmethod
    def setup_once():
        # type: () -> None
        old_start = Thread.start

        @wraps(old_start)
        def sentry_start(self, *a, **kw):
            # type: (Thread, *Any, **Any) -> Any
            integration = sentry_sdk.get_client().get_integration(ThreadingIntegration)
            if integration is None:
                return old_start(self, *a, **kw)

            if integration.propagate_scope:
                isolation_scope = sentry_sdk.get_isolation_scope()
                current_scope = sentry_sdk.get_current_scope()
            else:
                isolation_scope = None
                current_scope = None

            # Patching instance methods in `start()` creates a reference cycle if
            # done in a naive way. See
            # https://github.com/getsentry/sentry-python/pull/434
            #
            # In threading module, using current_thread API will access current thread instance
            # without holding it to avoid a reference cycle in an easier way.
            with capture_internal_exceptions():
                new_run = _wrap_run(
                    isolation_scope,
                    current_scope,
                    getattr(self.run, "__func__", self.run),
                )
                self.run = new_run  # type: ignore

            return old_start(self, *a, **kw)

        Thread.start = sentry_start  # type: ignore


def _wrap_run(isolation_scope_to_use, current_scope_to_use, old_run_func):
    # type: (Optional[sentry_sdk.Scope], Optional[sentry_sdk.Scope], F) -> F
    @wraps(old_run_func)
    def run(*a, **kw):
        # type: (*Any, **Any) -> Any
        def _run_old_run_func():
            # type: () -> Any
            try:
                self = current_thread()
                return old_run_func(self, *a, **kw)
            except Exception:
                reraise(*_capture_exception())

        if isolation_scope_to_use is not None and current_scope_to_use is not None:
            with use_isolation_scope(isolation_scope_to_use):
                with use_scope(current_scope_to_use):
                    return _run_old_run_func()
        else:
            return _run_old_run_func()

    return run  # type: ignore


def _capture_exception():
    # type: () -> ExcInfo
    exc_info = sys.exc_info()

    client = sentry_sdk.get_client()
    if client.get_integration(ThreadingIntegration) is not None:
        event, hint = event_from_exception(
            exc_info,
            client_options=client.options,
            mechanism={"type": "threading", "handled": False},
        )
        sentry_sdk.capture_event(event, hint=hint)

    return exc_info
