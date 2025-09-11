import io
import logging
import os
import urllib.parse
import urllib.request
import urllib.error
import urllib3
import sys

from itertools import chain, product

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Dict
    from typing import Optional
    from typing import Self

from sentry_sdk.utils import (
    logger as sentry_logger,
    env_to_bool,
    capture_internal_exceptions,
)
from sentry_sdk.envelope import Envelope


logger = logging.getLogger("spotlight")


DEFAULT_SPOTLIGHT_URL = "http://localhost:8969/stream"
DJANGO_SPOTLIGHT_MIDDLEWARE_PATH = "sentry_sdk.spotlight.SpotlightMiddleware"


class SpotlightClient:
    def __init__(self, url):
        # type: (str) -> None
        self.url = url
        self.http = urllib3.PoolManager()
        self.tries = 0

    def capture_envelope(self, envelope):
        # type: (Envelope) -> None
        body = io.BytesIO()
        envelope.serialize_into(body)
        try:
            req = self.http.request(
                url=self.url,
                body=body.getvalue(),
                method="POST",
                headers={
                    "Content-Type": "application/x-sentry-envelope",
                },
            )
            req.close()
        except Exception as e:
            # TODO: Implement buffering and retrying with exponential backoff
            sentry_logger.warning(str(e))


try:
    from django.utils.deprecation import MiddlewareMixin
    from django.http import HttpResponseServerError, HttpResponse, HttpRequest
    from django.conf import settings

    SPOTLIGHT_JS_ENTRY_PATH = "/assets/main.js"
    SPOTLIGHT_JS_SNIPPET_PATTERN = (
        "<script>window.__spotlight = {{ initOptions: {{ sidecarUrl: '{spotlight_url}', fullPage: false }} }};</script>\n"
        '<script type="module" crossorigin src="{spotlight_js_url}"></script>\n'
    )
    SPOTLIGHT_ERROR_PAGE_SNIPPET = (
        '<html><base href="{spotlight_url}">\n'
        '<script>window.__spotlight = {{ initOptions: {{ fullPage: true, startFrom: "/errors/{event_id}" }}}};</script>\n'
    )
    CHARSET_PREFIX = "charset="
    BODY_TAG_NAME = "body"
    BODY_CLOSE_TAG_POSSIBILITIES = tuple(
        "</{}>".format("".join(chars))
        for chars in product(*zip(BODY_TAG_NAME.upper(), BODY_TAG_NAME.lower()))
    )

    class SpotlightMiddleware(MiddlewareMixin):  # type: ignore[misc]
        _spotlight_script = None  # type: Optional[str]

        def __init__(self, get_response):
            # type: (Self, Callable[..., HttpResponse]) -> None
            super().__init__(get_response)

            import sentry_sdk.api

            self.sentry_sdk = sentry_sdk.api

            spotlight_client = self.sentry_sdk.get_client().spotlight
            if spotlight_client is None:
                sentry_logger.warning(
                    "Cannot find Spotlight client from SpotlightMiddleware, disabling the middleware."
                )
                return None
            # Spotlight URL has a trailing `/stream` part at the end so split it off
            self._spotlight_url = urllib.parse.urljoin(spotlight_client.url, "../")

        @property
        def spotlight_script(self):
            # type: (Self) -> Optional[str]
            if self._spotlight_script is None:
                try:
                    spotlight_js_url = urllib.parse.urljoin(
                        self._spotlight_url, SPOTLIGHT_JS_ENTRY_PATH
                    )
                    req = urllib.request.Request(
                        spotlight_js_url,
                        method="HEAD",
                    )
                    urllib.request.urlopen(req)
                    self._spotlight_script = SPOTLIGHT_JS_SNIPPET_PATTERN.format(
                        spotlight_url=self._spotlight_url,
                        spotlight_js_url=spotlight_js_url,
                    )
                except urllib.error.URLError as err:
                    sentry_logger.debug(
                        "Cannot get Spotlight JS to inject at %s. SpotlightMiddleware will not be very useful.",
                        spotlight_js_url,
                        exc_info=err,
                    )

            return self._spotlight_script

        def process_response(self, _request, response):
            # type: (Self, HttpRequest, HttpResponse) -> Optional[HttpResponse]
            content_type_header = tuple(
                p.strip()
                for p in response.headers.get("Content-Type", "").lower().split(";")
            )
            content_type = content_type_header[0]
            if len(content_type_header) > 1 and content_type_header[1].startswith(
                CHARSET_PREFIX
            ):
                encoding = content_type_header[1][len(CHARSET_PREFIX) :]
            else:
                encoding = "utf-8"

            if (
                self.spotlight_script is not None
                and not response.streaming
                and content_type == "text/html"
            ):
                content_length = len(response.content)
                injection = self.spotlight_script.encode(encoding)
                injection_site = next(
                    (
                        idx
                        for idx in (
                            response.content.rfind(body_variant.encode(encoding))
                            for body_variant in BODY_CLOSE_TAG_POSSIBILITIES
                        )
                        if idx > -1
                    ),
                    content_length,
                )

                # This approach works even when we don't have a `</body>` tag
                response.content = (
                    response.content[:injection_site]
                    + injection
                    + response.content[injection_site:]
                )

                if response.has_header("Content-Length"):
                    response.headers["Content-Length"] = content_length + len(injection)

            return response

        def process_exception(self, _request, exception):
            # type: (Self, HttpRequest, Exception) -> Optional[HttpResponseServerError]
            if not settings.DEBUG:
                return None

            try:
                spotlight = (
                    urllib.request.urlopen(self._spotlight_url).read().decode("utf-8")
                )
            except urllib.error.URLError:
                return None
            else:
                event_id = self.sentry_sdk.capture_exception(exception)
                return HttpResponseServerError(
                    spotlight.replace(
                        "<html>",
                        SPOTLIGHT_ERROR_PAGE_SNIPPET.format(
                            spotlight_url=self._spotlight_url, event_id=event_id
                        ),
                    )
                )

except ImportError:
    settings = None


def setup_spotlight(options):
    # type: (Dict[str, Any]) -> Optional[SpotlightClient]
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(" [spotlight] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

    url = options.get("spotlight")

    if url is True:
        url = DEFAULT_SPOTLIGHT_URL

    if not isinstance(url, str):
        return None

    with capture_internal_exceptions():
        if (
            settings is not None
            and settings.DEBUG
            and env_to_bool(os.environ.get("SENTRY_SPOTLIGHT_ON_ERROR", "1"))
            and env_to_bool(os.environ.get("SENTRY_SPOTLIGHT_MIDDLEWARE", "1"))
        ):
            middleware = settings.MIDDLEWARE
            if DJANGO_SPOTLIGHT_MIDDLEWARE_PATH not in middleware:
                settings.MIDDLEWARE = type(middleware)(
                    chain(middleware, (DJANGO_SPOTLIGHT_MIDDLEWARE_PATH,))
                )
                logger.info("Enabled Spotlight integration for Django")

    client = SpotlightClient(url)
    logger.info("Enabled Spotlight using sidecar at %s", url)

    return client
