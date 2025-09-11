from collections.abc import Set
import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.integrations import (
    _DEFAULT_FAILED_REQUEST_STATUS_CODES,
    DidNotEnable,
    Integration,
)
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import (
    ensure_integration_enabled,
    event_from_exception,
    transaction_from_function,
)

try:
    from litestar import Request, Litestar  # type: ignore
    from litestar.handlers.base import BaseRouteHandler  # type: ignore
    from litestar.middleware import DefineMiddleware  # type: ignore
    from litestar.routes.http import HTTPRoute  # type: ignore
    from litestar.data_extractors import ConnectionDataExtractor  # type: ignore
    from litestar.exceptions import HTTPException  # type: ignore
except ImportError:
    raise DidNotEnable("Litestar is not installed")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional, Union
    from litestar.types.asgi_types import ASGIApp  # type: ignore
    from litestar.types import (  # type: ignore
        HTTPReceiveMessage,
        HTTPScope,
        Message,
        Middleware,
        Receive,
        Scope as LitestarScope,
        Send,
        WebSocketReceiveMessage,
    )
    from litestar.middleware import MiddlewareProtocol
    from sentry_sdk._types import Event, Hint

_DEFAULT_TRANSACTION_NAME = "generic Litestar request"


class LitestarIntegration(Integration):
    identifier = "litestar"
    origin = f"auto.http.{identifier}"

    def __init__(
        self,
        failed_request_status_codes=_DEFAULT_FAILED_REQUEST_STATUS_CODES,  # type: Set[int]
    ) -> None:
        self.failed_request_status_codes = failed_request_status_codes

    @staticmethod
    def setup_once():
        # type: () -> None
        patch_app_init()
        patch_middlewares()
        patch_http_route_handle()

        # The following line follows the pattern found in other integrations such as `DjangoIntegration.setup_once`.
        # The Litestar `ExceptionHandlerMiddleware.__call__` catches exceptions and does the following
        # (among other things):
        #   1. Logs them, some at least (such as 500s) as errors
        #   2. Calls after_exception hooks
        # The `LitestarIntegration`` provides an after_exception hook (see `patch_app_init` below) to create a Sentry event
        # from an exception, which ends up being called during step 2 above. However, the Sentry `LoggingIntegration` will
        # by default create a Sentry event from error logs made in step 1 if we do not prevent it from doing so.
        ignore_logger("litestar")


class SentryLitestarASGIMiddleware(SentryAsgiMiddleware):
    def __init__(self, app, span_origin=LitestarIntegration.origin):
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
    Replaces the Litestar class's `__init__` function in order to inject `after_exception` handlers and set the
    `SentryLitestarASGIMiddleware` as the outmost middleware in the stack.
    See:
    - https://docs.litestar.dev/2/usage/applications.html#after-exception
    - https://docs.litestar.dev/2/usage/middleware/using-middleware.html
    """
    old__init__ = Litestar.__init__

    @ensure_integration_enabled(LitestarIntegration, old__init__)
    def injection_wrapper(self, *args, **kwargs):
        # type: (Litestar, *Any, **Any) -> None
        kwargs["after_exception"] = [
            exception_handler,
            *(kwargs.get("after_exception") or []),
        ]

        SentryLitestarASGIMiddleware.__call__ = SentryLitestarASGIMiddleware._run_asgi3  # type: ignore
        middleware = kwargs.get("middleware") or []
        kwargs["middleware"] = [SentryLitestarASGIMiddleware, *middleware]
        old__init__(self, *args, **kwargs)

    Litestar.__init__ = injection_wrapper


def patch_middlewares():
    # type: () -> None
    old_resolve_middleware_stack = BaseRouteHandler.resolve_middleware

    @ensure_integration_enabled(LitestarIntegration, old_resolve_middleware_stack)
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
        or middleware is SentryLitestarASGIMiddleware
    ):
        return middleware

    if isinstance(middleware, DefineMiddleware):
        old_call = middleware.middleware.__call__  # type: ASGIApp
    else:
        old_call = middleware.__call__

    async def _create_span_call(self, scope, receive, send):
        # type: (MiddlewareProtocol, LitestarScope, Receive, Send) -> None
        if sentry_sdk.get_client().get_integration(LitestarIntegration) is None:
            return await old_call(self, scope, receive, send)

        middleware_name = self.__class__.__name__
        with sentry_sdk.start_span(
            op=OP.MIDDLEWARE_LITESTAR,
            name=middleware_name,
            origin=LitestarIntegration.origin,
        ) as middleware_span:
            middleware_span.set_tag("litestar.middleware_name", middleware_name)

            # Creating spans for the "receive" callback
            async def _sentry_receive(*args, **kwargs):
                # type: (*Any, **Any) -> Union[HTTPReceiveMessage, WebSocketReceiveMessage]
                if sentry_sdk.get_client().get_integration(LitestarIntegration) is None:
                    return await receive(*args, **kwargs)
                with sentry_sdk.start_span(
                    op=OP.MIDDLEWARE_LITESTAR_RECEIVE,
                    name=getattr(receive, "__qualname__", str(receive)),
                    origin=LitestarIntegration.origin,
                ) as span:
                    span.set_tag("litestar.middleware_name", middleware_name)
                    return await receive(*args, **kwargs)

            receive_name = getattr(receive, "__name__", str(receive))
            receive_patched = receive_name == "_sentry_receive"
            new_receive = _sentry_receive if not receive_patched else receive

            # Creating spans for the "send" callback
            async def _sentry_send(message):
                # type: (Message) -> None
                if sentry_sdk.get_client().get_integration(LitestarIntegration) is None:
                    return await send(message)
                with sentry_sdk.start_span(
                    op=OP.MIDDLEWARE_LITESTAR_SEND,
                    name=getattr(send, "__qualname__", str(send)),
                    origin=LitestarIntegration.origin,
                ) as span:
                    span.set_tag("litestar.middleware_name", middleware_name)
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
        if sentry_sdk.get_client().get_integration(LitestarIntegration) is None:
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
            # Accounts for use of type `Ref` in earlier versions of litestar without the need to reference it as a type
            elif hasattr(route_handler.fn, "value"):
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

        sentry_scope._name = LitestarIntegration.identifier
        sentry_scope.add_event_processor(event_processor)

        return await old_handle(self, scope, receive, send)

    HTTPRoute.handle = handle_wrapper


def retrieve_user_from_scope(scope):
    # type: (LitestarScope) -> Optional[dict[str, Any]]
    scope_user = scope.get("user")
    if isinstance(scope_user, dict):
        return scope_user
    if hasattr(scope_user, "asdict"):  # dataclasses
        return scope_user.asdict()

    return None


@ensure_integration_enabled(LitestarIntegration)
def exception_handler(exc, scope):
    # type: (Exception, LitestarScope) -> None
    user_info = None  # type: Optional[dict[str, Any]]
    if should_send_default_pii():
        user_info = retrieve_user_from_scope(scope)
    if user_info and isinstance(user_info, dict):
        sentry_scope = sentry_sdk.get_isolation_scope()
        sentry_scope.set_user(user_info)

    if isinstance(exc, HTTPException):
        integration = sentry_sdk.get_client().get_integration(LitestarIntegration)
        if (
            integration is not None
            and exc.status_code not in integration.failed_request_status_codes
        ):
            return

    event, hint = event_from_exception(
        exc,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": LitestarIntegration.identifier, "handled": False},
    )

    sentry_sdk.capture_event(event, hint=hint)
