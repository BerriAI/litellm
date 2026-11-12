import json
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Final, Literal, Protocol

from fastapi import HTTPException
from starlette._utils import get_route_path
from starlette.routing import compile_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.common_utils.anthropic_error_payload import route_error_content

MaxRequestSizeGetter = Callable[[], int | float | None]
RequestSizeLimitEnabledGetter = Callable[[], bool]


class _RequestHeadersFields(Protocol):
    def get(self, key: Literal["headers"], /) -> Iterable[tuple[bytes, bytes]] | None: ...


class _RequestBodyFields(Protocol):
    def get(self, key: Literal["body"], default: bytes, /) -> bytes: ...


def _request_body_fields(message: _RequestBodyFields) -> _RequestBodyFields:
    return message


class RequestEntityTooLarge(HTTPException):
    def __init__(self, max_request_size_mb: float) -> None:
        super().__init__(status_code=413, detail=f"Request size is too large. Max size is {max_request_size_mb} MB")


def effective_max_request_size_mb(
    scope: Scope,
    configured: float | None,
    route_limits: Mapping[tuple[str, str], float],
) -> float | None:
    global_limit: Final = configured if configured is not None and configured > 0 else None
    method: Final = scope.get("method")
    path: Final = get_route_path(scope)
    exact_limit: Final = route_limits.get((method, path)) if isinstance(method, str) else None
    route_limit: Final = (
        exact_limit
        if exact_limit is not None
        else next(
            (
                limit
                for (route_method, route), limit in route_limits.items()
                if route_method == method and "{" in route and compile_path(route)[0].fullmatch(path) is not None
            ),
            None,
        )
    )
    if route_limit is None:
        return global_limit
    return min(route_limit, global_limit) if global_limit is not None else route_limit


class RequestSizeLimitMiddleware:
    """
    Reject oversized requests before downstream auth/routes parse the body.

    Content-Length can be rejected without reading any body bytes. Requests
    without Content-Length are counted as the ASGI stream is consumed, limiting
    memory exposure to the configured threshold plus the current chunk.
    """

    def __init__(
        self,
        app: ASGIApp,
        get_max_request_size_mb: MaxRequestSizeGetter,
        is_request_size_limit_enabled: RequestSizeLimitEnabledGetter,
        route_max_request_size_mb: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        self.app = app
        self.get_max_request_size_mb = get_max_request_size_mb
        self.is_request_size_limit_enabled = is_request_size_limit_enabled
        self.route_max_request_size_mb = MappingProxyType(dict(route_max_request_size_mb or {}))

    def _max_request_size_mb(self, scope: Scope) -> float | None:
        configured: Final = self.get_max_request_size_mb()
        return effective_max_request_size_mb(
            scope,
            configured if self.is_request_size_limit_enabled() else None,
            self.route_max_request_size_mb,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_request_size_mb: Final = self._max_request_size_mb(scope)
        max_request_size_bytes: Final = _mb_to_bytes(max_request_size_mb)
        if max_request_size_bytes is None or max_request_size_mb is None:
            await self.app(scope, receive, send)
            return

        content_length: Final = _get_content_length(scope=scope)
        if content_length is not None and content_length > max_request_size_bytes:
            await send_request_too_large(scope=scope, send=send, max_request_size_mb=max_request_size_mb)
            return

        received_body_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_body_bytes

            message: Final = await receive()
            if message["type"] != "http.request":
                return message

            received_body_bytes += len(_request_body_fields(message).get("body", b""))
            if received_body_bytes > max_request_size_bytes:
                raise RequestEntityTooLarge(max_request_size_mb)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started

            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestEntityTooLarge:
            if response_started:
                raise
            await send_request_too_large(scope=scope, send=send, max_request_size_mb=max_request_size_mb)


def _mb_to_bytes(max_request_size_mb: float | None) -> int | None:
    if max_request_size_mb is None:
        return None
    if max_request_size_mb <= 0:
        return None
    return int(max_request_size_mb * 1024 * 1024)


def _get_content_length(scope: _RequestHeadersFields) -> int | None:
    headers: Final = dict(scope.get("headers") or [])
    raw_content_length: Final = headers.get(b"content-length")
    if raw_content_length is None:
        return None

    try:
        return int(raw_content_length)
    except ValueError:
        return None


async def send_request_too_large(
    scope: Scope,
    send: Send,
    max_request_size_mb: float | None,
) -> None:
    message: Final = f"Request size is too large. Max size is {max_request_size_mb} MB"
    body: Final = json.dumps(
        route_error_content(scope, "request_too_large", message, {"error": message}),
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
