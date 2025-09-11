import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import (
    ensure_integration_enabled,
    event_from_exception,
    transaction_from_function,
)

try:
    from starlite import Request, Starlite, State  # type: ignore
    from starlite.handlers.base import BaseRouteHandler  # type: ignore
    from starlite.middleware import DefineMiddleware  # type: ignore
    from starlite.plugins.base import get_plugin_for_value  # type: ignore
    from starlite.routes.http import HTTPRoute  # type: ignore
    from starlite.utils import ConnectionDataExtractor, is_async_callable, Ref  # type: ignore
    from pydantic import BaseModel  # type: ignore
except ImportError:
    raise DidNotEnable("Starlite is not installed")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from starlite.types import (  # type: ignore
        ASGIApp,
        Hint,
        HTTPReceiveMessage,
        HTTPScope,
        Message,
        Middleware,
        Receive,
        Scope as StarliteScope,
        Send,
        WebSocketReceiveMessage,
    )
    from starlite import MiddlewareProtocol
    from sentry_sdk._types import Event


_DEFAULT_TRANSACTION_NAME = "generic Starlite request"


class StarliteIntegration(Integration):
    identifier = "starlite"
    origin = f"auto.http.{identifier}"

    @staticmethod
    def setup_once():
        # type: () -> None
        patch_app_init()
        patch_middlewares()
        patch_http_route_handle()


class SentryStarliteASGIMiddleware(SentryAsgiMiddleware):
    def __init__(self, app, span_origin=StarliteIntegration.origin):
        # type: (ASGIApp, str) -> None
        super().__init__(
            app=app,
            unsafe_context_data=False,
            transaction_style="endpoint",
            mechanism_type="asgi",
            span_origin=span_origin,
        )


def patch_app_init():
    # type: () -> None
    """
    Replaces the Starlite class's `__init__` function in order to inject `after_exception` handlers and set the
    `SentryStarliteASGIMiddleware` as the outmost middleware in the stack.
    See:
    - https://starlite-api.github.io/starlite/usage/0-the-starlite-app/5-application-hooks/#after-exception
    - https://starlite-api.github.io/starlite/usage/7-middleware/0-middleware-intro/
    """
    old__init__ = Starlite.__init__

    @ensure_integration_enabled(StarliteIntegration, old__init__)
    def injection_wrapper(self, *args, **kwargs):
        # type: (Starlite, *Any, **Any) -> None
        after_exception = kwargs.pop("after_exception", [])
        kwargs.update(
            after_exception=[
                exception_handler,
                *(
                    after_exception
                    if isinstance(after_exception, list)
                    else [after_exception]
                ),
            ]
        )

        SentryStarliteASGIMiddleware.__call__ = SentryStarliteASGIMiddleware._run_asgi3  # type: ignore
        middleware = kwargs.get("middleware") or []
        kwargs["middleware"] = [SentryStarliteASGIMiddleware, *middleware]
        old__init__(self, *args, **kwargs)

    Starlite.__init__ = injection_wrapper


def patch_middlewares():
    # type: () -> None
    old_resolve_middleware_stack = BaseRouteHandler.resolve_middleware

    @ensure_integration_enabled(StarliteIntegration, old_resolve_middleware_stack)
    def resolve_middleware_wrapper(self):
        # type: (BaseRouteHandler) -> list[Middleware]
        return [
            enable_span_for_middleware(middleware)
            for middleware in old_resolve_middleware_stack(self)
        ]

    BaseRouteHandler.resolve_middleware = resolve_middleware_wrapper


def enable_span_for_middleware(middleware):
    # type: (Middleware) -> Middleware
    if (
        not hasattr(middleware, "__call__")  # noqa: B004
        or middleware is SentryStarliteASGIMiddleware
    ):
        return middleware

    if isinstance(middleware, DefineMiddleware):
        old_call = middleware.middleware.__call__  # type: ASGIApp
    else:
        old_call = middleware.__call__

    async def _create_span_call(self, scope, receive, send):
        # type: (MiddlewareProtocol, StarliteScope, Receive, Send) -> None
        if sentry_sdk.get_client().get_integration(StarliteIntegration) is None:
            return await old_call(self, scope, receive, send)

        middleware_name = self.__class__.__name__
        with sentry_sdk.start_span(
            op=OP.MIDDLEWARE_STARLITE,
            name=middleware_name,
            origin=StarliteIntegration.origin,
        ) as middleware_span:
            middleware_span.set_tag("starlite.middleware_name", middleware_name)

            # Creating spans for the "receive" callback
            async def _sentry_receive(*args, **kwargs):
                # type: (*Any, **Any) -> Union[HTTPReceiveMessage, WebSocketReceiveMessage]
                if sentry_sdk.get_client().get_integration(StarliteIntegration) is None:
                    return await receive(*args, **kwargs)
                with sentry_sdk.start_span(
                    op=OP.MIDDLEWARE_STARLITE_RECEIVE,
                    name=getattr(receive, "__qualname__", str(receive)),
                    origin=StarliteIntegration.origin,
                ) as span:
                    span.set_tag("starlite.middleware_name", middleware_name)
                    return await receive(*args, **kwargs)

            receive_name = getattr(receive, "__name__", str(receive))
            receive_patched = receive_name == "_sentry_receive"
            new_receive = _sentry_receive if not receive_patched else receive

            # Creating spans for the "send" callback
            async def _sentry_send(message):
                # type: (Message) -> None
                if sentry_sdk.get_client().get_integration(StarliteIntegration) is None:
                    return await send(message)
                with sentry_sdk.start_span(
                    op=OP.MIDDLEWARE_STARLITE_SEND,
                    name=getattr(send, "__qualname__", str(send)),
                    origin=StarliteIntegration.origin,
                ) as span:
                    span.set_tag("starlite.middleware_name", middleware_name)
                    return await send(message)

            send_name = getattr(send, "__name__", str(send))
            send_patched = send_name == "_sentry_send"
            new_send = _sentry_send if not send_patched else send

            return await old_call(self, scope, new_receive, new_send)

    not_yet_patched = old_call.__name__ not in ["_create_span_call"]

    if not_yet_patched:
        if isinstance(middleware, DefineMiddleware):
            middleware.middleware.__call__ = _create_span_call
        else:
            middleware.__call__ = _create_span_call

    return middleware


def patch_http_route_handle():
    # type: () -> None
    old_handle = HTTPRoute.handle

    async def handle_wrapper(self, scope, receive, send):
        # type: (HTTPRoute, HTTPScope, Receive, Send) -> None
        if sentry_sdk.get_client().get_integration(StarliteIntegration) is None:
            return await old_handle(self, scope, receive, send)

        sentry_scope = sentry_sdk.get_isolation_scope()
        request = scope["app"].request_class(
            scope=scope, receive=receive, send=send
        )  # type: Request[Any, Any]
        extracted_request_data = ConnectionDataExtractor(
            parse_body=True, parse_query=True
        )(request)
        body = extracted_request_data.pop("body")

        request_data = await body

        def event_processor(event, _):
            # type: (Event, Hint) -> Event
            route_handler = scope.get("route_handler")

            request_info = event.get("request", {})
            request_info["content_length"] = len(scope.get("_body", b""))
            if should_send_default_pii():
                request_info["cookies"] = extracted_request_data["cookies"]
            if request_data is not None:
                request_info["data"] = request_data

            func = None
            if route_handler.name is not None:
                tx_name = route_handler.name
            elif isinstance(route_handler.fn, Ref):
                func = route_handler.fn.value
            else:
                func = route_handler.fn
            if func is not None:
                tx_name = transaction_from_function(func)

            tx_info = {"source": SOURCE_FOR_STYLE["endpoint"]}

            if not tx_name:
                tx_name = _DEFAULT_TRANSACTION_NAME
                tx_info = {"source": TRANSACTION_SOURCE_ROUTE}

            event.update(
                {
                    "request": request_info,
                    "transaction": tx_name,
                    "transaction_info": tx_info,
                }
            )
            return event

        sentry_scope._name = StarliteIntegration.identifier
        sentry_scope.add_event_processor(event_processor)

        return await old_handle(self, scope, receive, send)

    HTTPRoute.handle = handle_wrapper


def retrieve_user_from_scope(scope):
    # type: (StarliteScope) -> Optional[dict[str, Any]]
    scope_user = scope.get("user")
    if not scope_user:
        return None
    if isinstance(scope_user, dict):
        return scope_user
    if isinstance(scope_user, BaseModel):
        return scope_user.dict()
    if hasattr(scope_user, "asdict"):  # dataclasses
        return scope_user.asdict()

    plugin = get_plugin_for_value(scope_user)
    if plugin and not is_async_callable(plugin.to_dict):
        return plugin.to_dict(scope_user)

    return None


@ensure_integration_enabled(StarliteIntegration)
def exception_handler(exc, scope, _):
    # type: (Exception, StarliteScope, State) -> None
    user_info = None  # type: Optional[dict[str, Any]]
    if should_send_default_pii():
        user_info = retrieve_user_from_scope(scope)
    if user_info and isinstance(user_info, dict):
        sentry_scope = sentry_sdk.get_isolation_scope()
        sentry_scope.set_user(user_info)

    event, hint = event_from_exception(
        exc,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": StarliteIntegration.identifier, "handled": False},
    )

    sentry_sdk.capture_event(event, hint=hint)
