from contextlib import contextmanager
import json
from copy import deepcopy

import sentry_sdk
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.utils import AnnotatedValue, logger

try:
    from django.http.request import RawPostDataException
except ImportError:
    RawPostDataException = None

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from typing import Iterator
    from typing import Mapping
    from typing import MutableMapping
    from typing import Optional
    from typing import Union
    from sentry_sdk._types import Event, HttpStatusCodeRange


SENSITIVE_ENV_KEYS = (
    "REMOTE_ADDR",
    "HTTP_X_FORWARDED_FOR",
    "HTTP_SET_COOKIE",
    "HTTP_COOKIE",
    "HTTP_AUTHORIZATION",
    "HTTP_X_API_KEY",
    "HTTP_X_FORWARDED_FOR",
    "HTTP_X_REAL_IP",
)

SENSITIVE_HEADERS = tuple(
    x[len("HTTP_") :] for x in SENSITIVE_ENV_KEYS if x.startswith("HTTP_")
)

DEFAULT_HTTP_METHODS_TO_CAPTURE = (
    "CONNECT",
    "DELETE",
    "GET",
    # "HEAD",  # do not capture HEAD requests by default
    # "OPTIONS",  # do not capture OPTIONS requests by default
    "PATCH",
    "POST",
    "PUT",
    "TRACE",
)


# This noop context manager can be replaced with "from contextlib import nullcontext" when we drop Python 3.6 support
@contextmanager
def nullcontext():
    # type: () -> Iterator[None]
    yield


def request_body_within_bounds(client, content_length):
    # type: (Optional[sentry_sdk.client.BaseClient], int) -> bool
    if client is None:
        return False

    bodies = client.options["max_request_body_size"]
    return not (
        bodies == "never"
        or (bodies == "small" and content_length > 10**3)
        or (bodies == "medium" and content_length > 10**4)
    )


class RequestExtractor:
    """
    Base class for request extraction.
    """

    # It does not make sense to make this class an ABC because it is not used
    # for typing, only so that child classes can inherit common methods from
    # it. Only some child classes implement all methods that raise
    # NotImplementedError in this class.

    def __init__(self, request):
        # type: (Any) -> None
        self.request = request

    def extract_into_event(self, event):
        # type: (Event) -> None
        client = sentry_sdk.get_client()
        if not client.is_active():
            return

        data = None  # type: Optional[Union[AnnotatedValue, Dict[str, Any]]]

        content_length = self.content_length()
        request_info = event.get("request", {})

        if should_send_default_pii():
            request_info["cookies"] = dict(self.cookies())

        if not request_body_within_bounds(client, content_length):
            data = AnnotatedValue.removed_because_over_size_limit()
        else:
            # First read the raw body data
            # It is important to read this first because if it is Django
            # it will cache the body and then we can read the cached version
            # again in parsed_body() (or json() or wherever).
            raw_data = None
            try:
                raw_data = self.raw_data()
            except (RawPostDataException, ValueError):
                # If DjangoRestFramework is used it already read the body for us
                # so reading it here will fail. We can ignore this.
                pass

            parsed_body = self.parsed_body()
            if parsed_body is not None:
                data = parsed_body
            elif raw_data:
                data = AnnotatedValue.removed_because_raw_data()
            else:
                data = None

        if data is not None:
            request_info["data"] = data

        event["request"] = deepcopy(request_info)

    def content_length(self):
        # type: () -> int
        try:
            return int(self.env().get("CONTENT_LENGTH", 0))
        except ValueError:
            return 0

    def cookies(self):
        # type: () -> MutableMapping[str, Any]
        raise NotImplementedError()

    def raw_data(self):
        # type: () -> Optional[Union[str, bytes]]
        raise NotImplementedError()

    def form(self):
        # type: () -> Optional[Dict[str, Any]]
        raise NotImplementedError()

    def parsed_body(self):
        # type: () -> Optional[Dict[str, Any]]
        try:
            form = self.form()
        except Exception:
            form = None
        try:
            files = self.files()
        except Exception:
            files = None

        if form or files:
            data = {}
            if form:
                data = dict(form.items())
            if files:
                for key in files.keys():
                    data[key] = AnnotatedValue.removed_because_raw_data()

            return data

        return self.json()

    def is_json(self):
        # type: () -> bool
        return _is_json_content_type(self.env().get("CONTENT_TYPE"))

    def json(self):
        # type: () -> Optional[Any]
        try:
            if not self.is_json():
                return None

            try:
                raw_data = self.raw_data()
            except (RawPostDataException, ValueError):
                # The body might have already been read, in which case this will
                # fail
                raw_data = None

            if raw_data is None:
                return None

            if isinstance(raw_data, str):
                return json.loads(raw_data)
            else:
                return json.loads(raw_data.decode("utf-8"))
        except ValueError:
            pass

        return None

    def files(self):
        # type: () -> Optional[Dict[str, Any]]
        raise NotImplementedError()

    def size_of_file(self, file):
        # type: (Any) -> int
        raise NotImplementedError()

    def env(self):
        # type: () -> Dict[str, Any]
        raise NotImplementedError()


def _is_json_content_type(ct):
    # type: (Optional[str]) -> bool
    mt = (ct or "").split(";", 1)[0]
    return (
        mt == "application/json"
        or (mt.startswith("application/"))
        and mt.endswith("+json")
    )


def _filter_headers(headers):
    # type: (Mapping[str, str]) -> Mapping[str, Union[AnnotatedValue, str]]
    if should_send_default_pii():
        return headers

    return {
        k: (
            v
            if k.upper().replace("-", "_") not in SENSITIVE_HEADERS
            else AnnotatedValue.removed_because_over_size_limit()
        )
        for k, v in headers.items()
    }


def _in_http_status_code_range(code, code_ranges):
    # type: (object, list[HttpStatusCodeRange]) -> bool
    for target in code_ranges:
        if isinstance(target, int):
            if code == target:
                return True
            continue

        try:
            if code in target:
                return True
        except TypeError:
            logger.warning(
                "failed_request_status_codes has to be a list of integers or containers"
            )

    return False


class HttpCodeRangeContainer:
    """
    Wrapper to make it possible to use list[HttpStatusCodeRange] as a Container[int].
    Used for backwards compatibility with the old `failed_request_status_codes` option.
    """

    def __init__(self, code_ranges):
        # type: (list[HttpStatusCodeRange]) -> None
        self._code_ranges = code_ranges

    def __contains__(self, item):
        # type: (object) -> bool
        return _in_http_status_code_range(item, self._code_ranges)
