from functools import wraps

from django.dispatch import Signal

import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.integrations.django import DJANGO_VERSION

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Union


def _get_receiver_name(receiver):
    # type: (Callable[..., Any]) -> str
    name = ""

    if hasattr(receiver, "__qualname__"):
        name = receiver.__qualname__
    elif hasattr(receiver, "__name__"):  # Python 2.7 has no __qualname__
        name = receiver.__name__
    elif hasattr(
        receiver, "func"
    ):  # certain functions (like partials) dont have a name
        if hasattr(receiver, "func") and hasattr(receiver.func, "__name__"):
            name = "partial(<function " + receiver.func.__name__ + ">)"

    if (
        name == ""
    ):  # In case nothing was found, return the string representation (this is the slowest case)
        return str(receiver)

    if hasattr(receiver, "__module__"):  # prepend with module, if there is one
        name = receiver.__module__ + "." + name

    return name


def patch_signals():
    # type: () -> None
    """
    Patch django signal receivers to create a span.

    This only wraps sync receivers. Django>=5.0 introduced async receivers, but
    since we don't create transactions for ASGI Django, we don't wrap them.
    """
    from sentry_sdk.integrations.django import DjangoIntegration

    old_live_receivers = Signal._live_receivers

    def _sentry_live_receivers(self, sender):
        # type: (Signal, Any) -> Union[tuple[list[Callable[..., Any]], list[Callable[..., Any]]], list[Callable[..., Any]]]
        if DJANGO_VERSION >= (5, 0):
            sync_receivers, async_receivers = old_live_receivers(self, sender)
        else:
            sync_receivers = old_live_receivers(self, sender)
            async_receivers = []

        def sentry_sync_receiver_wrapper(receiver):
            # type: (Callable[..., Any]) -> Callable[..., Any]
            @wraps(receiver)
            def wrapper(*args, **kwargs):
                # type: (Any, Any) -> Any
                signal_name = _get_receiver_name(receiver)
                with sentry_sdk.start_span(
                    op=OP.EVENT_DJANGO,
                    name=signal_name,
                    origin=DjangoIntegration.origin,
                ) as span:
                    span.set_data("signal", signal_name)
                    return receiver(*args, **kwargs)

            return wrapper

        integration = sentry_sdk.get_client().get_integration(DjangoIntegration)
        if (
            integration
            and integration.signals_spans
            and self not in integration.signals_denylist
        ):
            for idx, receiver in enumerate(sync_receivers):
                sync_receivers[idx] = sentry_sync_receiver_wrapper(receiver)

        if DJANGO_VERSION >= (5, 0):
            return sync_receivers, async_receivers
        else:
            return sync_receivers

    Signal._live_receivers = _sentry_live_receivers
