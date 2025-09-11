import functools
import os
import sys
import weakref

import sentry_sdk
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
    reraise,
)

try:
    from pyramid.httpexceptions import HTTPException
    from pyramid.request import Request
except ImportError:
    raise DidNotEnable("Pyramid not installed")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyramid.response import Response
    from typing import Any
    from sentry_sdk.integrations.wsgi import _ScopedResponse
    from typing import Callable
    from typing import Dict
    from typing import Optional
    from webob.cookies import RequestCookies
    from webob.request import _FieldStorageWithFile

    from sentry_sdk.utils import ExcInfo
    from sentry_sdk._types import Event, EventProcessor


if getattr(Request, "authenticated_userid", None):

    def authenticated_userid(request):
        # type: (Request) -> Optional[Any]
        return request.authenticated_userid

else:
    # bw-compat for pyramid < 1.5
    from pyramid.security import authenticated_userid  # type: ignore


TRANSACTION_STYLE_VALUES = ("route_name", "route_pattern")


class PyramidIntegration(Integration):
    identifier = "pyramid"
    origin = f"auto.http.{identifier}"

    transaction_style = ""

    def __init__(self, transaction_style="route_name"):
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
        from pyramid import router

        old_call_view = router._call_view

        @functools.wraps(old_call_view)
        def sentry_patched_call_view(registry, request, *args, **kwargs):
            # type: (Any, Request, *Any, **Any) -> Response
            integration = sentry_sdk.get_client().get_integration(PyramidIntegration)
            if integration is None:
                return old_call_view(registry, request, *args, **kwargs)

            _set_transaction_name_and_source(
                sentry_sdk.get_current_scope(), integration.transaction_style, request
            )
            scope = sentry_sdk.get_isolation_scope()
            scope.add_event_processor(
                _make_event_processor(weakref.ref(request), integration)
            )

            return old_call_view(registry, request, *args, **kwargs)

        router._call_view = sentry_patched_call_view

        if hasattr(Request, "invoke_exception_view"):
            old_invoke_exception_view = Request.invoke_exception_view

            def sentry_patched_invoke_exception_view(self, *args, **kwargs):
                # type: (Request, *Any, **Any) -> Any
                rv = old_invoke_exception_view(self, *args, **kwargs)

                if (
                    self.exc_info
                    and all(self.exc_info)
                    and rv.status_int == 500
                    and sentry_sdk.get_client().get_integration(PyramidIntegration)
                    is not None
                ):
                    _capture_exception(self.exc_info)

                return rv

            Request.invoke_exception_view = sentry_patched_invoke_exception_view

        old_wsgi_call = router.Router.__call__

        @ensure_integration_enabled(PyramidIntegration, old_wsgi_call)
        def sentry_patched_wsgi_call(self, environ, start_response):
            # type: (Any, Dict[str, str], Callable[..., Any]) -> _ScopedResponse
            def sentry_patched_inner_wsgi_call(environ, start_response):
                # type: (Dict[str, Any], Callable[..., Any]) -> Any
                try:
                    return old_wsgi_call(self, environ, start_response)
                except Exception:
                    einfo = sys.exc_info()
                    _capture_exception(einfo)
                    reraise(*einfo)

            middleware = SentryWsgiMiddleware(
                sentry_patched_inner_wsgi_call,
                span_origin=PyramidIntegration.origin,
            )
            return middleware(environ, start_response)

        router.Router.__call__ = sentry_patched_wsgi_call


@ensure_integration_enabled(PyramidIntegration)
def _capture_exception(exc_info):
    # type: (ExcInfo) -> None
    if exc_info[0] is None or issubclass(exc_info[0], HTTPException):
        return

    event, hint = event_from_exception(
        exc_info,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": "pyramid", "handled": False},
    )

    sentry_sdk.capture_event(event, hint=hint)


def _set_transaction_name_and_source(scope, transaction_style, request):
    # type: (sentry_sdk.Scope, str, Request) -> None
    try:
        name_for_style = {
            "route_name": request.matched_route.name,
            "route_pattern": request.matched_route.pattern,
        }
        scope.set_transaction_name(
            name_for_style[transaction_style],
            source=SOURCE_FOR_STYLE[transaction_style],
        )
    except Exception:
        pass


class PyramidRequestExtractor(RequestExtractor):
    def url(self):
        # type: () -> str
        return self.request.path_url

    def env(self):
        # type: () -> Dict[str, str]
        return self.request.environ

    def cookies(self):
        # type: () -> RequestCookies
        return self.request.cookies

    def raw_data(self):
        # type: () -> str
        return self.request.text

    def form(self):
        # type: () -> Dict[str, str]
        return {
            key: value
            for key, value in self.request.POST.items()
            if not getattr(value, "filename", None)
        }

    def files(self):
        # type: () -> Dict[str, _FieldStorageWithFile]
        return {
            key: value
            for key, value in self.request.POST.items()
            if getattr(value, "filename", None)
        }

    def size_of_file(self, postdata):
        # type: (_FieldStorageWithFile) -> int
        file = postdata.file
        try:
            return os.fstat(file.fileno()).st_size
        except Exception:
            return 0


def _make_event_processor(weak_request, integration):
    # type: (Callable[[], Request], PyramidIntegration) -> EventProcessor
    def pyramid_event_processor(event, hint):
        # type: (Event, Dict[str, Any]) -> Event
        request = weak_request()
        if request is None:
            return event

        with capture_internal_exceptions():
            PyramidRequestExtractor(request).extract_into_event(event)

        if should_send_default_pii():
            with capture_internal_exceptions():
                user_info = event.setdefault("user", {})
                user_info.setdefault("id", authenticated_userid(request))

        return event

    return pyramid_event_processor
