import functools

import sentry_sdk
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
    parse_version,
    transaction_from_function,
)
from sentry_sdk.integrations import (
    Integration,
    DidNotEnable,
    _DEFAULT_FAILED_REQUEST_STATUS_CODES,
    _check_minimum_version,
)
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Set

    from sentry_sdk.integrations.wsgi import _ScopedResponse
    from typing import Any
    from typing import Dict
    from typing import Callable
    from typing import Optional
    from bottle import FileUpload, FormsDict, LocalRequest  # type: ignore

    from sentry_sdk._types import EventProcessor, Event

try:
    from bottle import (
        Bottle,
        HTTPResponse,
        Route,
        request as bottle_request,
        __version__ as BOTTLE_VERSION,
    )
except ImportError:
    raise DidNotEnable("Bottle not installed")


TRANSACTION_STYLE_VALUES = ("endpoint", "url")


class BottleIntegration(Integration):
    identifier = "bottle"
    origin = f"auto.http.{identifier}"

    transaction_style = ""

    def __init__(
        self,
        transaction_style="endpoint",  # type: str
        *,
        failed_request_status_codes=_DEFAULT_FAILED_REQUEST_STATUS_CODES,  # type: Set[int]
    ):
        # type: (...) -> None

        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError(
                "Invalid value for transaction_style: %s (must be in %s)"
                % (transaction_style, TRANSACTION_STYLE_VALUES)
            )
        self.transaction_style = transaction_style
        self.failed_request_status_codes = failed_request_status_codes

    @staticmethod
    def setup_once():
        # type: () -> None
        version = parse_version(BOTTLE_VERSION)
        _check_minimum_version(BottleIntegration, version)

        old_app = Bottle.__call__

        @ensure_integration_enabled(BottleIntegration, old_app)
        def sentry_patched_wsgi_app(self, environ, start_response):
            # type: (Any, Dict[str, str], Callable[..., Any]) -> _ScopedResponse
            middleware = SentryWsgiMiddleware(
                lambda *a, **kw: old_app(self, *a, **kw),
                span_origin=BottleIntegration.origin,
            )

            return middleware(environ, start_response)

        Bottle.__call__ = sentry_patched_wsgi_app

        old_handle = Bottle._handle

        @functools.wraps(old_handle)
        def _patched_handle(self, environ):
            # type: (Bottle, Dict[str, Any]) -> Any
            integration = sentry_sdk.get_client().get_integration(BottleIntegration)
            if integration is None:
                return old_handle(self, environ)

            scope = sentry_sdk.get_isolation_scope()
            scope._name = "bottle"
            scope.add_event_processor(
                _make_request_event_processor(self, bottle_request, integration)
            )
            res = old_handle(self, environ)

            return res

        Bottle._handle = _patched_handle

        old_make_callback = Route._make_callback

        @functools.wraps(old_make_callback)
        def patched_make_callback(self, *args, **kwargs):
            # type: (Route, *object, **object) -> Any
            prepared_callback = old_make_callback(self, *args, **kwargs)

            integration = sentry_sdk.get_client().get_integration(BottleIntegration)
            if integration is None:
                return prepared_callback

            def wrapped_callback(*args, **kwargs):
                # type: (*object, **object) -> Any
                try:
                    res = prepared_callback(*args, **kwargs)
                except Exception as exception:
                    _capture_exception(exception, handled=False)
                    raise exception

                if (
                    isinstance(res, HTTPResponse)
                    and res.status_code in integration.failed_request_status_codes
                ):
                    _capture_exception(res, handled=True)

                return res

            return wrapped_callback

        Route._make_callback = patched_make_callback


class BottleRequestExtractor(RequestExtractor):
    def env(self):
        # type: () -> Dict[str, str]
        return self.request.environ

    def cookies(self):
        # type: () -> Dict[str, str]
        return self.request.cookies

    def raw_data(self):
        # type: () -> bytes
        return self.request.body.read()

    def form(self):
        # type: () -> FormsDict
        if self.is_json():
            return None
        return self.request.forms.decode()

    def files(self):
        # type: () -> Optional[Dict[str, str]]
        if self.is_json():
            return None

        return self.request.files

    def size_of_file(self, file):
        # type: (FileUpload) -> int
        return file.content_length


def _set_transaction_name_and_source(event, transaction_style, request):
    # type: (Event, str, Any) -> None
    name = ""

    if transaction_style == "url":
        name = request.route.rule or ""

    elif transaction_style == "endpoint":
        name = (
            request.route.name
            or transaction_from_function(request.route.callback)
            or ""
        )

    event["transaction"] = name
    event["transaction_info"] = {"source": SOURCE_FOR_STYLE[transaction_style]}


def _make_request_event_processor(app, request, integration):
    # type: (Bottle, LocalRequest, BottleIntegration) -> EventProcessor

    def event_processor(event, hint):
        # type: (Event, dict[str, Any]) -> Event
        _set_transaction_name_and_source(event, integration.transaction_style, request)

        with capture_internal_exceptions():
            BottleRequestExtractor(request).extract_into_event(event)

        return event

    return event_processor


def _capture_exception(exception, handled):
    # type: (BaseException, bool) -> None
    event, hint = event_from_exception(
        exception,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": "bottle", "handled": handled},
    )
    sentry_sdk.capture_event(event, hint=hint)
