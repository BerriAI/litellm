import asyncio
import inspect
from functools import wraps

import sentry_sdk
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Union

    from sentry_sdk._types import Event, EventProcessor

try:
    import quart_auth  # type: ignore
except ImportError:
    quart_auth = None

try:
    from quart import (  # type: ignore
        has_request_context,
        has_websocket_context,
        Request,
        Quart,
        request,
        websocket,
    )
    from quart.signals import (  # type: ignore
        got_background_exception,
        got_request_exception,
        got_websocket_exception,
        request_started,
        websocket_started,
    )
except ImportError:
    raise DidNotEnable("Quart is not installed")
else:
    # Quart 0.19 is based on Flask and hence no longer has a Scaffold
    try:
        from quart.scaffold import Scaffold  # type: ignore
    except ImportError:
        from flask.sansio.scaffold import Scaffold  # type: ignore

TRANSACTION_STYLE_VALUES = ("endpoint", "url")


class QuartIntegration(Integration):
    identifier = "quart"
    origin = f"auto.http.{identifier}"

    transaction_style = ""

    def __init__(self, transaction_style="endpoint"):
        # type: (str) -> None
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError(
                "Invalid value for transaction_style: %s (must be in %s)"
                % (transaction_style, TRANSACTION_STYLE_VALUES)
            )
        self.transaction_style = transaction_style

    @staticmethod
    def setup_once():
        # type: () -> None

        request_started.connect(_request_websocket_started)
        websocket_started.connect(_request_websocket_started)
        got_background_exception.connect(_capture_exception)
        got_request_exception.connect(_capture_exception)
        got_websocket_exception.connect(_capture_exception)

        patch_asgi_app()
        patch_scaffold_route()


def patch_asgi_app():
    # type: () -> None
    old_app = Quart.__call__

    async def sentry_patched_asgi_app(self, scope, receive, send):
        # type: (Any, Any, Any, Any) -> Any
        if sentry_sdk.get_client().get_integration(QuartIntegration) is None:
            return await old_app(self, scope, receive, send)

        middleware = SentryAsgiMiddleware(
            lambda *a, **kw: old_app(self, *a, **kw),
            span_origin=QuartIntegration.origin,
        )
        middleware.__call__ = middleware._run_asgi3
        return await middleware(scope, receive, send)

    Quart.__call__ = sentry_patched_asgi_app


def patch_scaffold_route():
    # type: () -> None
    old_route = Scaffold.route

    def _sentry_route(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        old_decorator = old_route(*args, **kwargs)

        def decorator(old_func):
            # type: (Any) -> Any

            if inspect.isfunction(old_func) and not asyncio.iscoroutinefunction(
                old_func
            ):

                @wraps(old_func)
                @ensure_integration_enabled(QuartIntegration, old_func)
                def _sentry_func(*args, **kwargs):
                    # type: (*Any, **Any) -> Any
                    current_scope = sentry_sdk.get_current_scope()
                    if current_scope.transaction is not None:
                        current_scope.transaction.update_active_thread()

                    sentry_scope = sentry_sdk.get_isolation_scope()
                    if sentry_scope.profile is not None:
                        sentry_scope.profile.update_active_thread_id()

                    return old_func(*args, **kwargs)

                return old_decorator(_sentry_func)

            return old_decorator(old_func)

        return decorator

    Scaffold.route = _sentry_route


def _set_transaction_name_and_source(scope, transaction_style, request):
    # type: (sentry_sdk.Scope, str, Request) -> None

    try:
        name_for_style = {
            "url": request.url_rule.rule,
            "endpoint": request.url_rule.endpoint,
        }
        scope.set_transaction_name(
            name_for_style[transaction_style],
            source=SOURCE_FOR_STYLE[transaction_style],
        )
    except Exception:
        pass


async def _request_websocket_started(app, **kwargs):
    # type: (Quart, **Any) -> None
    integration = sentry_sdk.get_client().get_integration(QuartIntegration)
    if integration is None:
        return

    if has_request_context():
        request_websocket = request._get_current_object()
    if has_websocket_context():
        request_websocket = websocket._get_current_object()

    # Set the transaction name here, but rely on ASGI middleware
    # to actually start the transaction
    _set_transaction_name_and_source(
        sentry_sdk.get_current_scope(), integration.transaction_style, request_websocket
    )

    scope = sentry_sdk.get_isolation_scope()
    evt_processor = _make_request_event_processor(app, request_websocket, integration)
    scope.add_event_processor(evt_processor)


def _make_request_event_processor(app, request, integration):
    # type: (Quart, Request, QuartIntegration) -> EventProcessor
    def inner(event, hint):
        # type: (Event, dict[str, Any]) -> Event
        # if the request is gone we are fine not logging the data from
        # it.  This might happen if the processor is pushed away to
        # another thread.
        if request is None:
            return event

        with capture_internal_exceptions():
            # TODO: Figure out what to do with request body. Methods on request
            # are async, but event processors are not.

            request_info = event.setdefault("request", {})
            request_info["url"] = request.url
            request_info["query_string"] = request.query_string
            request_info["method"] = request.method
            request_info["headers"] = _filter_headers(dict(request.headers))

            if should_send_default_pii():
                request_info["env"] = {"REMOTE_ADDR": request.access_route[0]}
                _add_user_to_event(event)

        return event

    return inner


async def _capture_exception(sender, exception, **kwargs):
    # type: (Quart, Union[ValueError, BaseException], **Any) -> None
    integration = sentry_sdk.get_client().get_integration(QuartIntegration)
    if integration is None:
        return

    event, hint = event_from_exception(
        exception,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": "quart", "handled": False},
    )

    sentry_sdk.capture_event(event, hint=hint)


def _add_user_to_event(event):
    # type: (Event) -> None
    if quart_auth is None:
        return

    user = quart_auth.current_user
    if user is None:
        return

    with capture_internal_exceptions():
        user_info = event.setdefault("user", {})

        user_info["id"] = quart_auth.current_user._auth_id
