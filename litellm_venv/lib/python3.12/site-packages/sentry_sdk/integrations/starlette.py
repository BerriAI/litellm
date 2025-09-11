import asyncio
import functools
import warnings
from collections.abc import Set
from copy import deepcopy

import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.integrations import (
    DidNotEnable,
    Integration,
    _DEFAULT_FAILED_REQUEST_STATUS_CODES,
)
from sentry_sdk.integrations._wsgi_common import (
    DEFAULT_HTTP_METHODS_TO_CAPTURE,
    HttpCodeRangeContainer,
    _is_json_content_type,
    request_body_within_bounds,
)
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import (
    SOURCE_FOR_STYLE,
    TRANSACTION_SOURCE_COMPONENT,
    TRANSACTION_SOURCE_ROUTE,
)
from sentry_sdk.utils import (
    AnnotatedValue,
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
    logger,
    parse_version,
    transaction_from_function,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Awaitable, Callable, Container, Dict, Optional, Tuple, Union

    from sentry_sdk._types import Event, HttpStatusCodeRange

try:
    import starlette  # type: ignore
    from starlette import __version__ as STARLETTE_VERSION
    from starlette.applications import Starlette  # type: ignore
    from starlette.datastructures import UploadFile  # type: ignore
    from starlette.middleware import Middleware  # type: ignore
    from starlette.middleware.authentication import (  # type: ignore
        AuthenticationMiddleware,
    )
    from starlette.requests import Request  # type: ignore
    from starlette.routing import Match  # type: ignore
    from starlette.types import ASGIApp, Receive, Scope as StarletteScope, Send  # type: ignore
except ImportError:
    raise DidNotEnable("Starlette is not installed")

try:
    # Starlette 0.20
    from starlette.middleware.exceptions import ExceptionMiddleware  # type: ignore
except ImportError:
    # Startlette 0.19.1
    from starlette.exceptions import ExceptionMiddleware  # type: ignore

try:
    # Optional dependency of Starlette to parse form data.
    try:
        # python-multipart 0.0.13 and later
        import python_multipart as multipart  # type: ignore
    except ImportError:
        # python-multipart 0.0.12 and earlier
        import multipart  # type: ignore
except ImportError:
    multipart = None


_DEFAULT_TRANSACTION_NAME = "generic Starlette request"

TRANSACTION_STYLE_VALUES = ("endpoint", "url")


class StarletteIntegration(Integration):
    identifier = "starlette"
    origin = f"auto.http.{identifier}"

    transaction_style = ""

    def __init__(
        self,
        transaction_style="url",  # type: str
        failed_request_status_codes=_DEFAULT_FAILED_REQUEST_STATUS_CODES,  # type: Union[Set[int], list[HttpStatusCodeRange], None]
        middleware_spans=True,  # type: bool
        http_methods_to_capture=DEFAULT_HTTP_METHODS_TO_CAPTURE,  # type: tuple[str, ...]
    ):
        # type: (...) -> None
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError(
                "Invalid value for transaction_style: %s (must be in %s)"
                % (transaction_style, TRANSACTION_STYLE_VALUES)
            )
        self.transaction_style = transaction_style
        self.middleware_spans = middleware_spans
        self.http_methods_to_capture = tuple(map(str.upper, http_methods_to_capture))

        if isinstance(failed_request_status_codes, Set):
            self.failed_request_status_codes = (
                failed_request_status_codes
            )  # type: Container[int]
        else:
            warnings.warn(
                "Passing a list or None for failed_request_status_codes is deprecated. "
                "Please pass a set of int instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            if failed_request_status_codes is None:
                self.failed_request_status_codes = _DEFAULT_FAILED_REQUEST_STATUS_CODES
            else:
                self.failed_request_status_codes = HttpCodeRangeContainer(
                    failed_request_status_codes
                )

    @staticmethod
    def setup_once():
        # type: () -> None
        version = parse_version(STARLETTE_VERSION)

        if version is None:
            raise DidNotEnable(
                "Unparsable Starlette version: {}".format(STARLETTE_VERSION)
            )

        patch_middlewares()
        patch_asgi_app()
        patch_request_response()

        if version >= (0, 24):
            patch_templates()


def _enable_span_for_middleware(middleware_class):
    # type: (Any) -> type
    old_call = middleware_class.__call__

    async def _create_span_call(app, scope, receive, send, **kwargs):
        # type: (Any, Dict[str, Any], Callable[[], Awaitable[Dict[str, Any]]], Callable[[Dict[str, Any]], Awaitable[None]], Any) -> None
        integration = sentry_sdk.get_client().get_integration(StarletteIntegration)
        if integration is None or not integration.middleware_spans:
            return await old_call(app, scope, receive, send, **kwargs)

        middleware_name = app.__class__.__name__

        # Update transaction name with middleware name
        name, source = _get_transaction_from_middleware(app, scope, integration)
        if name is not None:
            sentry_sdk.get_current_scope().set_transaction_name(
                name,
                source=source,
            )

        with sentry_sdk.start_span(
            op=OP.MIDDLEWARE_STARLETTE,
            name=middleware_name,
            origin=StarletteIntegration.origin,
        ) as middleware_span:
            middleware_span.set_tag("starlette.middleware_name", middleware_name)

            # Creating spans for the "receive" callback
            async def _sentry_receive(*args, **kwargs):
                # type: (*Any, **Any) -> Any
                with sentry_sdk.start_span(
                    op=OP.MIDDLEWARE_STARLETTE_RECEIVE,
                    name=getattr(receive, "__qualname__", str(receive)),
                    origin=StarletteIntegration.origin,
                ) as span:
                    span.set_tag("starlette.middleware_name", middleware_name)
                    return await receive(*args, **kwargs)

            receive_name = getattr(receive, "__name__", str(receive))
            receive_patched = receive_name == "_sentry_receive"
            new_receive = _sentry_receive if not receive_patched else receive

            # Creating spans for the "send" callback
            async def _sentry_send(*args, **kwargs):
                # type: (*Any, **Any) -> Any
                with sentry_sdk.start_span(
                    op=OP.MIDDLEWARE_STARLETTE_SEND,
                    name=getattr(send, "__qualname__", str(send)),
                    origin=StarletteIntegration.origin,
                ) as span:
                    span.set_tag("starlette.middleware_name", middleware_name)
                    return await send(*args, **kwargs)

            send_name = getattr(send, "__name__", str(send))
            send_patched = send_name == "_sentry_send"
            new_send = _sentry_send if not send_patched else send

            return await old_call(app, scope, new_receive, new_send, **kwargs)

    not_yet_patched = old_call.__name__ not in [
        "_create_span_call",
        "_sentry_authenticationmiddleware_call",
        "_sentry_exceptionmiddleware_call",
    ]

    if not_yet_patched:
        middleware_class.__call__ = _create_span_call

    return middleware_class


@ensure_integration_enabled(StarletteIntegration)
def _capture_exception(exception, handled=False):
    # type: (BaseException, **Any) -> None
    event, hint = event_from_exception(
        exception,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": StarletteIntegration.identifier, "handled": handled},
    )

    sentry_sdk.capture_event(event, hint=hint)


def patch_exception_middleware(middleware_class):
    # type: (Any) -> None
    """
    Capture all exceptions in Starlette app and
    also extract user information.
    """
    old_middleware_init = middleware_class.__init__

    not_yet_patched = "_sentry_middleware_init" not in str(old_middleware_init)

    if not_yet_patched:

        def _sentry_middleware_init(self, *args, **kwargs):
            # type: (Any, Any, Any) -> None
            old_middleware_init(self, *args, **kwargs)

            # Patch existing exception handlers
            old_handlers = self._exception_handlers.copy()

            async def _sentry_patched_exception_handler(self, *args, **kwargs):
                # type: (Any, Any, Any) -> None
                integration = sentry_sdk.get_client().get_integration(
                    StarletteIntegration
                )

                exp = args[0]

                if integration is not None:
                    is_http_server_error = (
                        hasattr(exp, "status_code")
                        and isinstance(exp.status_code, int)
                        and exp.status_code in integration.failed_request_status_codes
                    )
                    if is_http_server_error:
                        _capture_exception(exp, handled=True)

                # Find a matching handler
                old_handler = None
                for cls in type(exp).__mro__:
                    if cls in old_handlers:
                        old_handler = old_handlers[cls]
                        break

                if old_handler is None:
                    return

                if _is_async_callable(old_handler):
                    return await old_handler(self, *args, **kwargs)
                else:
                    return old_handler(self, *args, **kwargs)

            for key in self._exception_handlers.keys():
                self._exception_handlers[key] = _sentry_patched_exception_handler

        middleware_class.__init__ = _sentry_middleware_init

        old_call = middleware_class.__call__

        async def _sentry_exceptionmiddleware_call(self, scope, receive, send):
            # type: (Dict[str, Any], Dict[str, Any], Callable[[], Awaitable[Dict[str, Any]]], Callable[[Dict[str, Any]], Awaitable[None]]) -> None
            # Also add the user (that was eventually set by be Authentication middle
            # that was called before this middleware). This is done because the authentication
            # middleware sets the user in the scope and then (in the same function)
            # calls this exception middelware. In case there is no exception (or no handler
            # for the type of exception occuring) then the exception bubbles up and setting the
            # user information into the sentry scope is done in auth middleware and the
            # ASGI middleware will then send everything to Sentry and this is fine.
            # But if there is an exception happening that the exception middleware here
            # has a handler for, it will send the exception directly to Sentry, so we need
            # the user information right now.
            # This is why we do it here.
            _add_user_to_sentry_scope(scope)
            await old_call(self, scope, receive, send)

        middleware_class.__call__ = _sentry_exceptionmiddleware_call


@ensure_integration_enabled(StarletteIntegration)
def _add_user_to_sentry_scope(scope):
    # type: (Dict[str, Any]) -> None
    """
    Extracts user information from the ASGI scope and
    adds it to Sentry's scope.
    """
    if "user" not in scope:
        return

    if not should_send_default_pii():
        return

    user_info = {}  # type: Dict[str, Any]
    starlette_user = scope["user"]

    username = getattr(starlette_user, "username", None)
    if username:
        user_info.setdefault("username", starlette_user.username)

    user_id = getattr(starlette_user, "id", None)
    if user_id:
        user_info.setdefault("id", starlette_user.id)

    email = getattr(starlette_user, "email", None)
    if email:
        user_info.setdefault("email", starlette_user.email)

    sentry_scope = sentry_sdk.get_isolation_scope()
    sentry_scope.user = user_info


def patch_authentication_middleware(middleware_class):
    # type: (Any) -> None
    """
    Add user information to Sentry scope.
    """
    old_call = middleware_class.__call__

    not_yet_patched = "_sentry_authenticationmiddleware_call" not in str(old_call)

    if not_yet_patched:

        async def _sentry_authenticationmiddleware_call(self, scope, receive, send):
            # type: (Dict[str, Any], Dict[str, Any], Callable[[], Awaitable[Dict[str, Any]]], Callable[[Dict[str, Any]], Awaitable[None]]) -> None
            await old_call(self, scope, receive, send)
            _add_user_to_sentry_scope(scope)

        middleware_class.__call__ = _sentry_authenticationmiddleware_call


def patch_middlewares():
    # type: () -> None
    """
    Patches Starlettes `Middleware` class to record
    spans for every middleware invoked.
    """
    old_middleware_init = Middleware.__init__

    not_yet_patched = "_sentry_middleware_init" not in str(old_middleware_init)

    if not_yet_patched:

        def _sentry_middleware_init(self, cls, **options):
            # type: (Any, Any, Any) -> None
            if cls == SentryAsgiMiddleware:
                return old_middleware_init(self, cls, **options)

            span_enabled_cls = _enable_span_for_middleware(cls)
            old_middleware_init(self, span_enabled_cls, **options)

            if cls == AuthenticationMiddleware:
                patch_authentication_middleware(cls)

            if cls == ExceptionMiddleware:
                patch_exception_middleware(cls)

        Middleware.__init__ = _sentry_middleware_init


def patch_asgi_app():
    # type: () -> None
    """
    Instrument Starlette ASGI app using the SentryAsgiMiddleware.
    """
    old_app = Starlette.__call__

    async def _sentry_patched_asgi_app(self, scope, receive, send):
        # type: (Starlette, StarletteScope, Receive, Send) -> None
        integration = sentry_sdk.get_client().get_integration(StarletteIntegration)
        if integration is None:
            return await old_app(self, scope, receive, send)

        middleware = SentryAsgiMiddleware(
            lambda *a, **kw: old_app(self, *a, **kw),
            mechanism_type=StarletteIntegration.identifier,
            transaction_style=integration.transaction_style,
            span_origin=StarletteIntegration.origin,
            http_methods_to_capture=(
                integration.http_methods_to_capture
                if integration
                else DEFAULT_HTTP_METHODS_TO_CAPTURE
            ),
        )

        middleware.__call__ = middleware._run_asgi3
        return await middleware(scope, receive, send)

    Starlette.__call__ = _sentry_patched_asgi_app


# This was vendored in from Starlette to support Starlette 0.19.1 because
# this function was only introduced in 0.20.x
def _is_async_callable(obj):
    # type: (Any) -> bool
    while isinstance(obj, functools.partial):
        obj = obj.func

    return asyncio.iscoroutinefunction(obj) or (
        callable(obj) and asyncio.iscoroutinefunction(obj.__call__)
    )


def patch_request_response():
    # type: () -> None
    old_request_response = starlette.routing.request_response

    def _sentry_request_response(func):
        # type: (Callable[[Any], Any]) -> ASGIApp
        old_func = func

        is_coroutine = _is_async_callable(old_func)
        if is_coroutine:

            async def _sentry_async_func(*args, **kwargs):
                # type: (*Any, **Any) -> Any
                integration = sentry_sdk.get_client().get_integration(
                    StarletteIntegration
                )
                if integration is None:
                    return await old_func(*args, **kwargs)

                request = args[0]

                _set_transaction_name_and_source(
                    sentry_sdk.get_current_scope(),
                    integration.transaction_style,
                    request,
                )

                sentry_scope = sentry_sdk.get_isolation_scope()
                extractor = StarletteRequestExtractor(request)
                info = await extractor.extract_request_info()

                def _make_request_event_processor(req, integration):
                    # type: (Any, Any) -> Callable[[Event, dict[str, Any]], Event]
                    def event_processor(event, hint):
                        # type: (Event, Dict[str, Any]) -> Event

                        # Add info from request to event
                        request_info = event.get("request", {})
                        if info:
                            if "cookies" in info:
                                request_info["cookies"] = info["cookies"]
                            if "data" in info:
                                request_info["data"] = info["data"]
                        event["request"] = deepcopy(request_info)

                        return event

                    return event_processor

                sentry_scope._name = StarletteIntegration.identifier
                sentry_scope.add_event_processor(
                    _make_request_event_processor(request, integration)
                )

                return await old_func(*args, **kwargs)

            func = _sentry_async_func

        else:

            @functools.wraps(old_func)
            def _sentry_sync_func(*args, **kwargs):
                # type: (*Any, **Any) -> Any
                integration = sentry_sdk.get_client().get_integration(
                    StarletteIntegration
                )
                if integration is None:
                    return old_func(*args, **kwargs)

                current_scope = sentry_sdk.get_current_scope()
                if current_scope.transaction is not None:
                    current_scope.transaction.update_active_thread()

                sentry_scope = sentry_sdk.get_isolation_scope()
                if sentry_scope.profile is not None:
                    sentry_scope.profile.update_active_thread_id()

                request = args[0]

                _set_transaction_name_and_source(
                    sentry_scope, integration.transaction_style, request
                )

                extractor = StarletteRequestExtractor(request)
                cookies = extractor.extract_cookies_from_request()

                def _make_request_event_processor(req, integration):
                    # type: (Any, Any) -> Callable[[Event, dict[str, Any]], Event]
                    def event_processor(event, hint):
                        # type: (Event, dict[str, Any]) -> Event

                        # Extract information from request
                        request_info = event.get("request", {})
                        if cookies:
                            request_info["cookies"] = cookies

                        event["request"] = deepcopy(request_info)

                        return event

                    return event_processor

                sentry_scope._name = StarletteIntegration.identifier
                sentry_scope.add_event_processor(
                    _make_request_event_processor(request, integration)
                )

                return old_func(*args, **kwargs)

            func = _sentry_sync_func

        return old_request_response(func)

    starlette.routing.request_response = _sentry_request_response


def patch_templates():
    # type: () -> None

    # If markupsafe is not installed, then Jinja2 is not installed
    # (markupsafe is a dependency of Jinja2)
    # In this case we do not need to patch the Jinja2Templates class
    try:
        from markupsafe import Markup
    except ImportError:
        return  # Nothing to do

    from starlette.templating import Jinja2Templates  # type: ignore

    old_jinja2templates_init = Jinja2Templates.__init__

    not_yet_patched = "_sentry_jinja2templates_init" not in str(
        old_jinja2templates_init
    )

    if not_yet_patched:

        def _sentry_jinja2templates_init(self, *args, **kwargs):
            # type: (Jinja2Templates, *Any, **Any) -> None
            def add_sentry_trace_meta(request):
                # type: (Request) -> Dict[str, Any]
                trace_meta = Markup(
                    sentry_sdk.get_current_scope().trace_propagation_meta()
                )
                return {
                    "sentry_trace_meta": trace_meta,
                }

            kwargs.setdefault("context_processors", [])

            if add_sentry_trace_meta not in kwargs["context_processors"]:
                kwargs["context_processors"].append(add_sentry_trace_meta)

            return old_jinja2templates_init(self, *args, **kwargs)

        Jinja2Templates.__init__ = _sentry_jinja2templates_init


class StarletteRequestExtractor:
    """
    Extracts useful information from the Starlette request
    (like form data or cookies) and adds it to the Sentry event.
    """

    request = None  # type: Request

    def __init__(self, request):
        # type: (StarletteRequestExtractor, Request) -> None
        self.request = request

    def extract_cookies_from_request(self):
        # type: (StarletteRequestExtractor) -> Optional[Dict[str, Any]]
        cookies = None  # type: Optional[Dict[str, Any]]
        if should_send_default_pii():
            cookies = self.cookies()

        return cookies

    async def extract_request_info(self):
        # type: (StarletteRequestExtractor) -> Optional[Dict[str, Any]]
        client = sentry_sdk.get_client()

        request_info = {}  # type: Dict[str, Any]

        with capture_internal_exceptions():
            # Add cookies
            if should_send_default_pii():
                request_info["cookies"] = self.cookies()

            # If there is no body, just return the cookies
            content_length = await self.content_length()
            if not content_length:
                return request_info

            # Add annotation if body is too big
            if content_length and not request_body_within_bounds(
                client, content_length
            ):
                request_info["data"] = AnnotatedValue.removed_because_over_size_limit()
                return request_info

            # Add JSON body, if it is a JSON request
            json = await self.json()
            if json:
                request_info["data"] = json
                return request_info

            # Add form as key/value pairs, if request has form data
            form = await self.form()
            if form:
                form_data = {}
                for key, val in form.items():
                    is_file = isinstance(val, UploadFile)
                    form_data[key] = (
                        val
                        if not is_file
                        else AnnotatedValue.removed_because_raw_data()
                    )

                request_info["data"] = form_data
                return request_info

            # Raw data, do not add body just an annotation
            request_info["data"] = AnnotatedValue.removed_because_raw_data()
            return request_info

    async def content_length(self):
        # type: (StarletteRequestExtractor) -> Optional[int]
        if "content-length" in self.request.headers:
            return int(self.request.headers["content-length"])

        return None

    def cookies(self):
        # type: (StarletteRequestExtractor) -> Dict[str, Any]
        return self.request.cookies

    async def form(self):
        # type: (StarletteRequestExtractor) -> Any
        if multipart is None:
            return None

        # Parse the body first to get it cached, as Starlette does not cache form() as it
        # does with body() and json() https://github.com/encode/starlette/discussions/1933
        # Calling `.form()` without calling `.body()` first will
        # potentially break the users project.
        await self.request.body()

        return await self.request.form()

    def is_json(self):
        # type: (StarletteRequestExtractor) -> bool
        return _is_json_content_type(self.request.headers.get("content-type"))

    async def json(self):
        # type: (StarletteRequestExtractor) -> Optional[Dict[str, Any]]
        if not self.is_json():
            return None

        return await self.request.json()


def _transaction_name_from_router(scope):
    # type: (StarletteScope) -> Optional[str]
    router = scope.get("router")
    if not router:
        return None

    for route in router.routes:
        match = route.matches(scope)
        if match[0] == Match.FULL:
            return route.path

    return None


def _set_transaction_name_and_source(scope, transaction_style, request):
    # type: (sentry_sdk.Scope, str, Any) -> None
    name = None
    source = SOURCE_FOR_STYLE[transaction_style]

    if transaction_style == "endpoint":
        endpoint = request.scope.get("endpoint")
        if endpoint:
            name = transaction_from_function(endpoint) or None

    elif transaction_style == "url":
        name = _transaction_name_from_router(request.scope)

    if name is None:
        name = _DEFAULT_TRANSACTION_NAME
        source = TRANSACTION_SOURCE_ROUTE

    scope.set_transaction_name(name, source=source)
    logger.debug(
        "[Starlette] Set transaction name and source on scope: %s / %s", name, source
    )


def _get_transaction_from_middleware(app, asgi_scope, integration):
    # type: (Any, Dict[str, Any], StarletteIntegration) -> Tuple[Optional[str], Optional[str]]
    name = None
    source = None

    if integration.transaction_style == "endpoint":
        name = transaction_from_function(app.__class__)
        source = TRANSACTION_SOURCE_COMPONENT
    elif integration.transaction_style == "url":
        name = _transaction_name_from_router(asgi_scope)
        source = TRANSACTION_SOURCE_ROUTE

    return name, source
