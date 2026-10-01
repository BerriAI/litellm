from dataclasses import dataclass
from threading import Lock
from typing import Final

import httpx
from httpx._client import USE_CLIENT_DEFAULT, UseClientDefault
from httpx._types import AuthTypes
from typing_extensions import ReadOnly, TypedDict


class UpstreamResponseMetadata(TypedDict):
    attempt_id: ReadOnly[str]
    status_code: ReadOnly[int | None]
    headers: ReadOnly[tuple[tuple[str, str], ...]]
    truncated: ReadOnly[bool]


_MAX_RESPONSES: Final = 8
_MAX_HEADERS: Final = 64
_MAX_NAME: Final = 128
_MAX_VALUE: Final = 512
_SECRET_HEADERS: Final = frozenset(
    (
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "set-cookie2",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "x-amz-security-token",
        "x-goog-api-key",
    )
)


def _safe_header(name: str, value: str) -> tuple[str, str]:
    normalized: Final = name.lower()
    secret: Final = normalized in _SECRET_HEADERS or normalized.endswith(
        ("-api-key", "-secret", "-token", "-authorization", "-cookie")
    )
    return normalized[:_MAX_NAME], "[REDACTED]" if secret else value[:_MAX_VALUE]


def response_metadata(
    attempt_id: str, status_code: int | None, headers: tuple[tuple[str, str], ...]
) -> UpstreamResponseMetadata:
    return UpstreamResponseMetadata(
        attempt_id=attempt_id,
        status_code=status_code,
        headers=tuple(_safe_header(name, value) for name, value in headers[:_MAX_HEADERS]),
        truncated=len(headers) > _MAX_HEADERS
        or any(len(name) > _MAX_NAME or len(value) > _MAX_VALUE for name, value in headers[:_MAX_HEADERS]),
    )


@dataclass(frozen=True, slots=True)
class CapturedUpstreamResponse:
    attempt_id: str
    status_code: int
    headers: tuple[tuple[str, str], ...]
    truncated: bool = False


class UpstreamResponseCapture:
    def __init__(self, responses: tuple[CapturedUpstreamResponse, ...] = (), dropped_responses: bool = False) -> None:
        self._lock: Final = Lock()
        self._responses = responses
        self._dropped_responses = dropped_responses

    def __deepcopy__(self, memo: dict[int, object]) -> "UpstreamResponseCapture":
        with self._lock:
            return UpstreamResponseCapture(self._responses, self._dropped_responses)

    @property
    def responses(self) -> tuple[CapturedUpstreamResponse, ...]:
        with self._lock:
            return self._responses

    def record(self, attempt_id: str, response: httpx.Response) -> None:
        metadata: Final = response_metadata(attempt_id, response.status_code, tuple(response.headers.multi_items()))
        captured: Final = CapturedUpstreamResponse(
            attempt_id=attempt_id,
            status_code=response.status_code,
            headers=metadata["headers"],
            truncated=metadata["truncated"],
        )
        with self._lock:
            self._dropped_responses = self._dropped_responses or len(self._responses) >= _MAX_RESPONSES
            self._responses = (*self._responses[-(_MAX_RESPONSES - 1) :], captured)

    def snapshot(self) -> tuple[UpstreamResponseMetadata, ...]:
        with self._lock:
            return tuple(
                UpstreamResponseMetadata(
                    attempt_id=item.attempt_id,
                    status_code=item.status_code,
                    headers=item.headers,
                    truncated=item.truncated or self._dropped_responses,
                )
                for item in self._responses
            )


def send_with_capture(
    client: httpx.Client,
    request: httpx.Request,
    capture: UpstreamResponseCapture | None,
    attempt_id: str,
    stream: bool,
    *,
    auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
    follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
) -> httpx.Response:
    if capture is None:
        return client.send(request, stream=stream, auth=auth, follow_redirects=follow_redirects)
    response: Final = client.send(request, stream=True, auth=auth, follow_redirects=follow_redirects)
    for item in (*response.history, response):
        capture.record(attempt_id, item)
    try:
        if not stream:
            response.read()
        return response
    except BaseException:
        response.close()
        raise


async def async_send_with_capture(
    client: httpx.AsyncClient,
    request: httpx.Request,
    capture: UpstreamResponseCapture | None,
    attempt_id: str,
    stream: bool,
    *,
    auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
    follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
) -> httpx.Response:
    if capture is None:
        return await client.send(request, stream=stream, auth=auth, follow_redirects=follow_redirects)
    response: Final = await client.send(request, stream=True, auth=auth, follow_redirects=follow_redirects)
    for item in (*response.history, response):
        capture.record(attempt_id, item)
    try:
        if not stream:
            await response.aread()
        return response
    except BaseException:
        await response.aclose()
        raise
