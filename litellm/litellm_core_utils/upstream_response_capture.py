from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Final

import httpx
from typing_extensions import ReadOnly, TypedDict

from litellm.litellm_core_utils.secret_redaction import redact_string


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
        "ocp-apim-subscription-key",
    )
)


def _safe_header(name: str, value: str) -> tuple[str, str]:
    normalized: Final = name.lower()
    secret: Final = normalized in _SECRET_HEADERS or normalized.endswith(
        ("-api-key", "-secret", "-token", "-authorization", "-cookie")
    )
    return normalized[:_MAX_NAME], "[REDACTED]" if secret else redact_string(value)[:_MAX_VALUE]


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
        self._attempt_counts: Final[dict[str, int]] = {}

    def __deepcopy__(self, memo: dict[int, object]) -> "UpstreamResponseCapture":
        with self._lock:
            copied: Final = UpstreamResponseCapture(self._responses, self._dropped_responses)
            copied._attempt_counts.update(self._attempt_counts)
            return copied

    def allocate_attempt_id(self, call_id: str) -> str:
        with self._lock:
            attempt_count: Final = self._attempt_counts.get(call_id, 0)
            self._attempt_counts[call_id] = attempt_count + 1
        return call_id if attempt_count == 0 else f"{call_id}.{attempt_count}"

    @property
    def responses(self) -> tuple[CapturedUpstreamResponse, ...]:
        with self._lock:
            return self._responses

    def record(self, attempt_id: str, response: httpx.Response) -> None:
        self.record_metadata(
            attempt_id,
            response.status_code,
            tuple(response.headers.multi_items()),
        )

    def record_metadata(
        self,
        attempt_id: str,
        status_code: int | None,
        headers: tuple[tuple[str, str], ...],
    ) -> None:
        if status_code is None:
            return
        metadata: Final = response_metadata(
            attempt_id,
            status_code,
            headers,
        )
        captured: Final = CapturedUpstreamResponse(
            attempt_id=attempt_id,
            status_code=status_code,
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


@dataclass(frozen=True, slots=True)
class UpstreamAttempt:
    capture: UpstreamResponseCapture
    attempt_id: str


class _Scope:
    __slots__ = ("attempt", "open")

    def __init__(self, attempt: UpstreamAttempt) -> None:
        self.attempt = attempt
        self.open = True


_CURRENT_UPSTREAM_SCOPE: Final[ContextVar[_Scope | None]] = ContextVar(
    "litellm_current_upstream_scope",
    default=None,
)


def current_upstream_attempt() -> UpstreamAttempt | None:
    scope: Final = _CURRENT_UPSTREAM_SCOPE.get()
    return scope.attempt if scope is not None and scope.open else None


@contextmanager
def _bind_upstream_attempt(
    attempt: UpstreamAttempt,
) -> Generator[UpstreamAttempt, None, None]:
    scope: Final = _Scope(attempt)
    token: Final = _CURRENT_UPSTREAM_SCOPE.set(scope)
    try:
        yield attempt
    finally:
        scope.open = False
        _CURRENT_UPSTREAM_SCOPE.reset(token)


@contextmanager
def upstream_attempt(
    capture: UpstreamResponseCapture,
    call_id: str,
) -> Generator[UpstreamAttempt, None, None]:
    current: Final = current_upstream_attempt()
    if current is not None and current.capture is capture:
        yield current
        return
    attempt: Final = UpstreamAttempt(capture, capture.allocate_attempt_id(call_id))
    with _bind_upstream_attempt(attempt):
        yield attempt


@contextmanager
def resume_upstream_attempt(
    attempt: UpstreamAttempt | None,
) -> Generator[None, None, None]:
    if attempt is None:
        yield
        return
    with _bind_upstream_attempt(attempt):
        yield


@contextmanager
def suppress_upstream_capture() -> Generator[None, None, None]:
    token: Final = _CURRENT_UPSTREAM_SCOPE.set(None)
    try:
        yield
    finally:
        _CURRENT_UPSTREAM_SCOPE.reset(token)


def record_current_upstream(
    status_code: int | None,
    headers: tuple[tuple[str, str], ...],
) -> None:
    attempt: Final = current_upstream_attempt()
    if attempt is None:
        return
    attempt.capture.record_metadata(attempt.attempt_id, status_code, headers)
