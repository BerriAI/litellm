import urllib

from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.integrations._wsgi_common import _filter_headers

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from typing import Optional
    from typing import Union
    from typing_extensions import Literal

    from sentry_sdk.utils import AnnotatedValue


def _get_headers(asgi_scope):
    # type: (Any) -> Dict[str, str]
    """
    Extract headers from the ASGI scope, in the format that the Sentry protocol expects.
    """
    headers = {}  # type: Dict[str, str]
    for raw_key, raw_value in asgi_scope["headers"]:
        key = raw_key.decode("latin-1")
        value = raw_value.decode("latin-1")
        if key in headers:
            headers[key] = headers[key] + ", " + value
        else:
            headers[key] = value

    return headers


def _get_url(asgi_scope, default_scheme, host):
    # type: (Dict[str, Any], Literal["ws", "http"], Optional[Union[AnnotatedValue, str]]) -> str
    """
    Extract URL from the ASGI scope, without also including the querystring.
    """
    scheme = asgi_scope.get("scheme", default_scheme)

    server = asgi_scope.get("server", None)
    path = asgi_scope.get("root_path", "") + asgi_scope.get("path", "")

    if host:
        return "%s://%s%s" % (scheme, host, path)

    if server is not None:
        host, port = server
        default_port = {"http": 80, "https": 443, "ws": 80, "wss": 443}.get(scheme)
        if port != default_port:
            return "%s://%s:%s%s" % (scheme, host, port, path)
        return "%s://%s%s" % (scheme, host, path)
    return path


def _get_query(asgi_scope):
    # type: (Any) -> Any
    """
    Extract querystring from the ASGI scope, in the format that the Sentry protocol expects.
    """
    qs = asgi_scope.get("query_string")
    if not qs:
        return None
    return urllib.parse.unquote(qs.decode("latin-1"))


def _get_ip(asgi_scope):
    # type: (Any) -> str
    """
    Extract IP Address from the ASGI scope based on request headers with fallback to scope client.
    """
    headers = _get_headers(asgi_scope)
    try:
        return headers["x-forwarded-for"].split(",")[0].strip()
    except (KeyError, IndexError):
        pass

    try:
        return headers["x-real-ip"]
    except KeyError:
        pass

    return asgi_scope.get("client")[0]


def _get_request_data(asgi_scope):
    # type: (Any) -> Dict[str, Any]
    """
    Returns data related to the HTTP request from the ASGI scope.
    """
    request_data = {}  # type: Dict[str, Any]
    ty = asgi_scope["type"]
    if ty in ("http", "websocket"):
        request_data["method"] = asgi_scope.get("method")

        request_data["headers"] = headers = _filter_headers(_get_headers(asgi_scope))
        request_data["query_string"] = _get_query(asgi_scope)

        request_data["url"] = _get_url(
            asgi_scope, "http" if ty == "http" else "ws", headers.get("host")
        )

    client = asgi_scope.get("client")
    if client and should_send_default_pii():
        request_data["env"] = {"REMOTE_ADDR": _get_ip(asgi_scope)}

    return request_data
