"""
An ASGI middleware.

Based on Tom Christie's `sentry-asgi <https://github.com/encode/sentry-asgi>`.
"""

import asyncio
import inspect
from copy import deepcopy
from functools import partial

import sentry_sdk
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP

from sentry_sdk.integrations._asgi_common import (
    _get_headers,
    _get_request_data,
    _get_url,
)
from sentry_sdk.integrations._wsgi_common import (
    DEFAULT_HTTP_METHODS_TO_CAPTURE,
    nullcontext,
)
from sentry_sdk.sessions import track_session
from sentry_sdk.tracing import (
    SOURCE_FOR_STYLE,
    TRANSACTION_SOURCE_ROUTE,
    TRANSACTION_SOURCE_URL,
    TRANSACTION_SOURCE_COMPONENT,
    TRANSACTION_SOURCE_CUSTOM,
)
from sentry_sdk.utils import (
    ContextVar,
    event_from_exception,
    HAS_REAL_CONTEXTVARS,
    CONTEXTVARS_ERROR_MESSAGE,
    logger,
    transaction_from_function,
    _get_installed_modules,
)
from sentry_sdk.tracing import Transaction

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Dict
    from typing import Optional
    from typing import Tuple

    from sentry_sdk._types import Event, Hint


_asgi_middleware_applied = ContextVar("sentry_asgi_middleware_applied")

_DEFAULT_TRANSACTION_NAME = "generic ASGI request"

TRANSACTION_STYLE_VALUES = ("endpoint", "url")


def _capture_exception(exc, mechanism_type="asgi"):
    # type: (Any, str) -> None

    event, hint = event_from_exception(
        exc,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": mechanism_type, "handled": False},
    )
    sentry_sdk.capture_event(event, hint=hint)


def _looks_like_asgi3(app):
    # type: (Any) -> bool
    """
    Try to figure out if an application object supports ASGI3.

    This is how uvicorn figures out the application version as well.
    """
    if inspect.isclass(app):
        return hasattr(app, "__await__")
    elif inspect.isfunction(app):
        return asyncio.iscoroutinefunction(app)
    else:
        call = getattr(app, "__call__", None)  # noqa
        return asyncio.iscoroutinefunction(call)


class SentryAsgiMiddleware:
    __slots__ = (
        "app",
        "__call__",
        "transaction_style",
        "mechanism_type",
        "span_origin",
        "http_methods_to_capture",
    )

    def __init__(
        self,
        app,  # type: Any
        unsafe_context_data=False,  # type: bool
        transaction_style="endpoint",  # type: str
        mechanism_type="asgi",  # type: str
        span_origin="manual",  # type: str
        http_methods_to_capture=DEFAULT_HTTP_METHODS_TO_CAPTURE,  # type: Tuple[str, ...]
    ):
        # type: (...) -> None
        """
        Instrument an ASGI application with Sentry. Provides HTTP/websocket
        data to sent events and basic handling for exceptions bubbling up
        through the middleware.

        :param unsafe_context_data: Disable errors when a proper contextvars installation could not be found. We do not recommend changing this from the default.
        """
        if not unsafe_context_data and not HAS_REAL_CONTEXTVARS:
            # We better have contextvars or we're going to leak state between
            # requests.
            raise RuntimeError(
                "The ASGI middleware for Sentry requires Python 3.7+ "
                "or the aiocontextvars package." + CONTEXTVARS_ERROR_MESSAGE
            )
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError(
                "Invalid value for transaction_style: %s (must be in %s)"
                % (transaction_style, TRANSACTION_STYLE_VALUES)
            )

        asgi_middleware_while_using_starlette_or_fastapi = (
            mechanism_type == "asgi" and "starlette" in _get_installed_modules()
        )
        if asgi_middleware_while_using_starlette_or_fastapi:
            logger.warning(
                "The Sentry Python SDK can now automatically support ASGI frameworks like Starlette and FastAPI. "
                "Please remove 'SentryAsgiMiddleware' from your project. "
                "See https://docs.sentry.io/platforms/python/guides/asgi/ for more information."
            )

        self.transaction_style = transaction_style
        self.mechanism_type = mechanism_type
        self.span_origin = span_origin
        self.app = app
        self.http_methods_to_capture = http_methods_to_capture

        if _looks_like_asgi3(app):
            self.__call__ = self._run_asgi3  # type: Callable[..., Any]
        else:
            self.__call__ = self._run_asgi2

    def _run_asgi2(self, scope):
        # type: (Any) -> Any
        async def inner(receive, send):
            # type: (Any, Any) -> Any
            return await self._run_app(scope, receive, send, asgi_version=2)

        return inner

    async def _run_asgi3(self, scope, receive, send):
        # type: (Any, Any, Any) -> Any
        return await self._run_app(scope, receive, send, asgi_version=3)

    async def _run_app(self, scope, receive, send, asgi_version):
        # type: (Any, Any, Any, Any, int) -> Any
        is_recursive_asgi_middleware = _asgi_middleware_applied.get(False)
        is_lifespan = scope["type"] == "lifespan"
        if is_recursive_asgi_middleware or is_lifespan:
            try:
                if asgi_version == 2:
                    return await self.app(scope)(receive, send)
                else:
                    return await self.app(scope, receive, send)

            except Exception as exc:
                _capture_exception(exc, mechanism_type=self.mechanism_type)
                raise exc from None

        _asgi_middleware_applied.set(True)
        try:
            with sentry_sdk.isolation_scope() as sentry_scope:
                with track_session(sentry_scope, session_mode="request"):
                    sentry_scope.clear_breadcrumbs()
                    sentry_scope._name = "asgi"
                    processor = partial(self.event_processor, asgi_scope=scope)
                    sentry_scope.add_event_processor(processor)

                    ty = scope["type"]
                    (
                        transaction_name,
                        transaction_source,
                    ) = self._get_transaction_name_and_source(
                        self.transaction_style,
                        scope,
                    )

                    method = scope.get("method", "").upper()
                    transaction = None
                    if method in self.http_methods_to_capture:
                        if ty in ("http", "websocket"):
                            transaction = continue_trace(
                                _get_headers(scope),
                                op="{}.server".format(ty),
                                name=transaction_name,
                                source=transaction_source,
                                origin=self.span_origin,
                            )
                            logger.debug(
                                "[ASGI] Created transaction (continuing trace): %s",
                                transaction,
                            )
                        else:
                            transaction = Transaction(
                                op=OP.HTTP_SERVER,
                                name=transaction_name,
                                source=transaction_source,
                                origin=self.span_origin,
                            )
                            logger.debug(
                                "[ASGI] Created transaction (new): %s", transaction
                            )

                        transaction.set_tag("asgi.type", ty)
                        logger.debug(
                            "[ASGI] Set transaction name and source on transaction: '%s' / '%s'",
                            transaction.name,
                            transaction.source,
                        )

                    with (
                        sentry_sdk.start_transaction(
                            transaction,
                            custom_sampling_context={"asgi_scope": scope},
                        )
                        if transaction is not None
                        else nullcontext()
                    ):
                        logger.debug("[ASGI] Started transaction: %s", transaction)
                        try:

                            async def _sentry_wrapped_send(event):
                                # type: (Dict[str, Any]) -> Any
                                if transaction is not None:
                                    is_http_response = (
                                        event.get("type") == "http.response.start"
                                        and "status" in event
                                    )
                                    if is_http_response:
                                        transaction.set_http_status(event["status"])

                                return await send(event)

                            if asgi_version == 2:
                                return await self.app(scope)(
                                    receive, _sentry_wrapped_send
                                )
                            else:
                                return await self.app(
                                    scope, receive, _sentry_wrapped_send
                                )
                        except Exception as exc:
                            _capture_exception(exc, mechanism_type=self.mechanism_type)
                            raise exc from None
        finally:
            _asgi_middleware_applied.set(False)

    def event_processor(self, event, hint, asgi_scope):
        # type: (Event, Hint, Any) -> Optional[Event]
        request_data = event.get("request", {})
        request_data.update(_get_request_data(asgi_scope))
        event["request"] = deepcopy(request_data)

        # Only set transaction name if not already set by Starlette or FastAPI (or other frameworks)
        already_set = event["transaction"] != _DEFAULT_TRANSACTION_NAME and event[
            "transaction_info"
        ].get("source") in [
            TRANSACTION_SOURCE_COMPONENT,
            TRANSACTION_SOURCE_ROUTE,
            TRANSACTION_SOURCE_CUSTOM,
        ]
        if not already_set:
            name, source = self._get_transaction_name_and_source(
                self.transaction_style, asgi_scope
            )
            event["transaction"] = name
            event["transaction_info"] = {"source": source}

            logger.debug(
                "[ASGI] Set transaction name and source in event_processor: '%s' / '%s'",
                event["transaction"],
                event["transaction_info"]["source"],
            )

        return event

    # Helper functions.
    #
    # Note: Those functions are not public API. If you want to mutate request
    # data to your liking it's recommended to use the `before_send` callback
    # for that.

    def _get_transaction_name_and_source(self, transaction_style, asgi_scope):
        # type: (SentryAsgiMiddleware, str, Any) -> Tuple[str, str]
        name = None
        source = SOURCE_FOR_STYLE[transaction_style]
        ty = asgi_scope.get("type")

        if transaction_style == "endpoint":
            endpoint = asgi_scope.get("endpoint")
            # Webframeworks like Starlette mutate the ASGI env once routing is
            # done, which is sometime after the request has started. If we have
            # an endpoint, overwrite our generic transaction name.
            if endpoint:
                name = transaction_from_function(endpoint) or ""
            else:
                name = _get_url(asgi_scope, "http" if ty == "http" else "ws", host=None)
                source = TRANSACTION_SOURCE_URL

        elif transaction_style == "url":
            # FastAPI includes the route object in the scope to let Sentry extract the
            # path from it for the transaction name
            route = asgi_scope.get("route")
            if route:
                path = getattr(route, "path", None)
                if path is not None:
                    name = path
            else:
                name = _get_url(asgi_scope, "http" if ty == "http" else "ws", host=None)
                source = TRANSACTION_SOURCE_URL

        if name is None:
            name = _DEFAULT_TRANSACTION_NAME
            source = TRANSACTION_SOURCE_ROUTE
            return name, source

        return name, source
