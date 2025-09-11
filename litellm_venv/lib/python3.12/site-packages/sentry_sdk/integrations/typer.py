import sentry_sdk
from sentry_sdk.utils import (
    capture_internal_exceptions,
    event_from_exception,
)
from sentry_sdk.integrations import Integration, DidNotEnable

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

try:
    import typer
except ImportError:
    raise DidNotEnable("Typer not installed")


class TyperIntegration(Integration):
    identifier = "typer"

    @staticmethod
    def setup_once():
        # type: () -> None
        typer.main.except_hook = _make_excepthook(typer.main.except_hook)  # type: ignore


def _make_excepthook(old_excepthook):
    # type: (Excepthook) -> Excepthook
    def sentry_sdk_excepthook(type_, value, traceback):
        # type: (Type[BaseException], BaseException, Optional[TracebackType]) -> None
        integration = sentry_sdk.get_client().get_integration(TyperIntegration)

        # Note: If we replace this with ensure_integration_enabled then
        # we break the exceptiongroup backport;
        # See: https://github.com/getsentry/sentry-python/issues/3097
        if integration is None:
            return old_excepthook(type_, value, traceback)

        with capture_internal_exceptions():
            event, hint = event_from_exception(
                (type_, value, traceback),
                client_options=sentry_sdk.get_client().options,
                mechanism={"type": "typer", "handled": False},
            )
            sentry_sdk.capture_event(event, hint=hint)

        return old_excepthook(type_, value, traceback)

    return sentry_sdk_excepthook
