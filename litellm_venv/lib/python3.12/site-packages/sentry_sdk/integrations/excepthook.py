import sys

import sentry_sdk
from sentry_sdk.utils import (
    capture_internal_exceptions,
    event_from_exception,
)
from sentry_sdk.integrations import Integration

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from typing import Any
    from typing import Type
    from typing import Optional

    from types import TracebackType

    Excepthook = Callable[
        [Type[BaseException], BaseException, Optional[TracebackType]],
        Any,
    ]


class ExcepthookIntegration(Integration):
    identifier = "excepthook"

    always_run = False

    def __init__(self, always_run=False):
        # type: (bool) -> None

        if not isinstance(always_run, bool):
            raise ValueError(
                "Invalid value for always_run: %s (must be type boolean)"
                % (always_run,)
            )
        self.always_run = always_run

    @staticmethod
    def setup_once():
        # type: () -> None
        sys.excepthook = _make_excepthook(sys.excepthook)


def _make_excepthook(old_excepthook):
    # type: (Excepthook) -> Excepthook
    def sentry_sdk_excepthook(type_, value, traceback):
        # type: (Type[BaseException], BaseException, Optional[TracebackType]) -> None
        integration = sentry_sdk.get_client().get_integration(ExcepthookIntegration)

        # Note: If  we replace this with ensure_integration_enabled then
        # we break the exceptiongroup backport;
        # See: https://github.com/getsentry/sentry-python/issues/3097
        if integration is None:
            return old_excepthook(type_, value, traceback)

        if _should_send(integration.always_run):
            with capture_internal_exceptions():
                event, hint = event_from_exception(
                    (type_, value, traceback),
                    client_options=sentry_sdk.get_client().options,
                    mechanism={"type": "excepthook", "handled": False},
                )
                sentry_sdk.capture_event(event, hint=hint)

        return old_excepthook(type_, value, traceback)

    return sentry_sdk_excepthook


def _should_send(always_run=False):
    # type: (bool) -> bool
    if always_run:
        return True

    if hasattr(sys, "ps1"):
        # Disable the excepthook for interactive Python shells, otherwise
        # every typo gets sent to Sentry.
        return False

    return True
