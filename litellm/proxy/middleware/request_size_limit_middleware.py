import json
from typing import Callable, Optional, Union

from starlette.types import ASGIApp, Message, Receive, Scope, Send

MaxRequestSizeGetter = Callable[[], Optional[Union[int, float]]]
RequestSizeLimitEnabledGetter = Callable[[], bool]


class RequestEntityTooLarge(Exception):
    pass


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
    ) -> None:
        self.app = app
        self.get_max_request_size_mb = get_max_request_size_mb
        self.is_request_size_limit_enabled = is_request_size_limit_enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_request_size_mb = self.get_max_request_size_mb()
        max_request_size_bytes = _mb_to_bytes(max_request_size_mb)
        if max_request_size_bytes is None or not self.is_request_size_limit_enabled():
            await self.app(scope, receive, send)
            return

        content_length = _get_content_length(scope=scope)
        if content_length is not None and content_length > max_request_size_bytes:
            await _send_request_too_large(
                send=send, max_request_size_mb=max_request_size_mb
            )
            return

        received_body_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_body_bytes

            message = await receive()
            if message["type"] != "http.request":
                return message

            received_body_bytes += len(message.get("body", b""))
            if received_body_bytes > max_request_size_bytes:
                raise RequestEntityTooLarge
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
            await _send_request_too_large(
                send=send, max_request_size_mb=max_request_size_mb
            )


def _mb_to_bytes(max_request_size_mb: Optional[Union[int, float]]) -> Optional[int]:
    if max_request_size_mb is None:
        return None
    if max_request_size_mb <= 0:
        return None
    return int(max_request_size_mb * 1024 * 1024)


def _get_content_length(scope: Scope) -> Optional[int]:
    headers = dict(scope.get("headers") or [])
    raw_content_length = headers.get(b"content-length")
    if raw_content_length is None:
        return None

    try:
        return int(raw_content_length)
    except ValueError:
        return None


async def _send_request_too_large(
    send: Send,
    max_request_size_mb: Optional[Union[int, float]],
) -> None:
    body = json.dumps(
        {"error": f"Request size is too large. Max size is {max_request_size_mb} MB"},
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
