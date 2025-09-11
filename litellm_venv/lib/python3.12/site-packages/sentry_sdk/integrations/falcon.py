import sentry_sdk
from sentry_sdk.integrations import _check_minimum_version, Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    event_from_exception,
    parse_version,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from typing import Optional

    from sentry_sdk._types import Event, EventProcessor

# In Falcon 3.0 `falcon.api_helpers` is renamed to `falcon.app_helpers`
# and `falcon.API` to `falcon.App`

try:
    import falcon  # type: ignore

    from falcon import __version__ as FALCON_VERSION
except ImportError:
    raise DidNotEnable("Falcon not installed")

try:
    import falcon.app_helpers  # type: ignore

    falcon_helpers = falcon.app_helpers
    falcon_app_class = falcon.App
    FALCON3 = True
except ImportError:
    import falcon.api_helpers  # type: ignore

    falcon_helpers = falcon.api_helpers
    falcon_app_class = falcon.API
    FALCON3 = False


_FALCON_UNSET = None  # type: Optional[object]
if FALCON3:  # falcon.request._UNSET is only available in Falcon 3.0+
    with capture_internal_exceptions():
        from falcon.request import _UNSET as _FALCON_UNSET  # type: ignore[import-not-found, no-redef]


class FalconRequestExtractor(RequestExtractor):
    def env(self):
        # type: () -> Dict[str, Any]
        return self.request.env

    def cookies(self):
        # type: () -> Dict[str, Any]
        return self.request.cookies

    def form(self):
        # type: () -> None
        return None  # No such concept in Falcon

    def files(self):
        # type: () -> None
        return None  # No such concept in Falcon

    def raw_data(self):
        # type: () -> Optional[str]

        # As request data can only be read once we won't make this available
        # to Sentry. Just send back a dummy string in case there was a
        # content length.
        # TODO(jmagnusson): Figure out if there's a way to support this
        content_length = self.content_length()
        if content_length > 0:
            return "[REQUEST_CONTAINING_RAW_DATA]"
        else:
            return None

    def json(self):
        # type: () -> Optional[Dict[str, Any]]
        # fallback to cached_media = None if self.request._media is not available
        cached_media = None
        with capture_internal_exceptions():
            # self.request._media is the cached self.request.media
            # value. It is only available if self.request.media
            # has already been accessed. Therefore, reading
            # self.request._media will not exhaust the raw request
            # stream (self.request.bounded_stream) because it has
            # already been read if self.request._media is set.
            cached_media = self.request._media

        if cached_media is not _FALCON_UNSET:
            return cached_media

        return None


class SentryFalconMiddleware:
    """Captures exceptions in Falcon requests and send to Sentry"""

    def process_request(self, req, resp, *args, **kwargs):
        # type: (Any, Any, *Any, **Any) -> None
        integration = sentry_sdk.get_client().get_integration(FalconIntegration)
        if integration is None:
            return

        scope = sentry_sdk.get_isolation_scope()
        scope._name = "falcon"
        scope.add_event_processor(_make_request_event_processor(req, integration))


TRANSACTION_STYLE_VALUES = ("uri_template", "path")


class FalconIntegration(Integration):
    identifier = "falcon"
    origin = f"auto.http.{identifier}"

    transaction_style = ""

    def __init__(self, transaction_style="uri_template"):
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

        version = parse_version(FALCON_VERSION)
        _check_minimum_version(FalconIntegration, version)

        _patch_wsgi_app()
        _patch_handle_exception()
        _patch_prepare_middleware()


def _patch_wsgi_app():
    # type: () -> None
    original_wsgi_app = falcon_app_class.__call__

    def sentry_patched_wsgi_app(self, env, start_response):
        # type: (falcon.API, Any, Any) -> Any
        integration = sentry_sdk.get_client().get_integration(FalconIntegration)
        if integration is None:
            return original_wsgi_app(self, env, start_response)

        sentry_wrapped = SentryWsgiMiddleware(
            lambda envi, start_resp: original_wsgi_app(self, envi, start_resp),
            span_origin=FalconIntegration.origin,
        )

        return sentry_wrapped(env, start_response)

    falcon_app_class.__call__ = sentry_patched_wsgi_app


def _patch_handle_exception():
    # type: () -> None
    original_handle_exception = falcon_app_class._handle_exception

    @ensure_integration_enabled(FalconIntegration, original_handle_exception)
    def sentry_patched_handle_exception(self, *args):
        # type: (falcon.API, *Any) -> Any
        # NOTE(jmagnusson): falcon 2.0 changed falcon.API._handle_exception
        # method signature from `(ex, req, resp, params)` to
        # `(req, resp, ex, params)`
        ex = response = None
        with capture_internal_exceptions():
            ex = next(argument for argument in args if isinstance(argument, Exception))
            response = next(
                argument for argument in args if isinstance(argument, falcon.Response)
            )

        was_handled = original_handle_exception(self, *args)

        if ex is None or response is None:
            # Both ex and response should have a non-None value at this point; otherwise,
            # there is an error with the SDK that will have been captured in the
            # capture_internal_exceptions block above.
            return was_handled

        if _exception_leads_to_http_5xx(ex, response):
            event, hint = event_from_exception(
                ex,
                client_options=sentry_sdk.get_client().options,
                mechanism={"type": "falcon", "handled": False},
            )
            sentry_sdk.capture_event(event, hint=hint)

        return was_handled

    falcon_app_class._handle_exception = sentry_patched_handle_exception


def _patch_prepare_middleware():
    # type: () -> None
    original_prepare_middleware = falcon_helpers.prepare_middleware

    def sentry_patched_prepare_middleware(
        middleware=None, independent_middleware=False, asgi=False
    ):
        # type: (Any, Any, bool) -> Any
        if asgi:
            # We don't support ASGI Falcon apps, so we don't patch anything here
            return original_prepare_middleware(middleware, independent_middleware, asgi)

        integration = sentry_sdk.get_client().get_integration(FalconIntegration)
        if integration is not None:
            middleware = [SentryFalconMiddleware()] + (middleware or [])

        # We intentionally omit the asgi argument here, since the default is False anyways,
        # and this way, we remain backwards-compatible with pre-3.0.0 Falcon versions.
        return original_prepare_middleware(middleware, independent_middleware)

    falcon_helpers.prepare_middleware = sentry_patched_prepare_middleware


def _exception_leads_to_http_5xx(ex, response):
    # type: (Exception, falcon.Response) -> bool
    is_server_error = isinstance(ex, falcon.HTTPError) and (ex.status or "").startswith(
        "5"
    )
    is_unhandled_error = not isinstance(
        ex, (falcon.HTTPError, falcon.http_status.HTTPStatus)
    )

    # We only check the HTTP status on Falcon 3 because in Falcon 2, the status on the response
    # at the stage where we capture it is listed as 200, even though we would expect to see a 500
    # status. Since at the time of this change, Falcon 2 is ca. 4 years old, we have decided to
    # only perform this check on Falcon 3+, despite the risk that some handled errors might be
    # reported to Sentry as unhandled on Falcon 2.
    return (is_server_error or is_unhandled_error) and (
        not FALCON3 or _has_http_5xx_status(response)
    )


def _has_http_5xx_status(response):
    # type: (falcon.Response) -> bool
    return response.status.startswith("5")


def _set_transaction_name_and_source(event, transaction_style, request):
    # type: (Event, str, falcon.Request) -> None
    name_for_style = {
        "uri_template": request.uri_template,
        "path": request.path,
    }
    event["transaction"] = name_for_style[transaction_style]
    event["transaction_info"] = {"source": SOURCE_FOR_STYLE[transaction_style]}


def _make_request_event_processor(req, integration):
    # type: (falcon.Request, FalconIntegration) -> EventProcessor

    def event_processor(event, hint):
        # type: (Event, dict[str, Any]) -> Event
        _set_transaction_name_and_source(event, integration.transaction_style, req)

        with capture_internal_exceptions():
            FalconRequestExtractor(req).extract_into_event(event)

        return event

    return event_processor
