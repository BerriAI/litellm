from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Final

import httpx


@dataclass(frozen=True, slots=True)
class CapturedUpstreamResponse:
    attempt_id: str
    status_code: int
    headers: tuple[tuple[str, str], ...]


class UpstreamResponseCapture:
    def __init__(self) -> None:
        self._lock: Final = Lock()
        self._responses: tuple[CapturedUpstreamResponse, ...] = ()

    @property
    def responses(self) -> tuple[CapturedUpstreamResponse, ...]:
        with self._lock:
            return self._responses

    @contextmanager
    def bind(self, attempt_id: str) -> Generator[None, None, None]:
        token: Final = _active_capture.set(_CaptureBinding(self, attempt_id))
        try:
            yield
        finally:
            _active_capture.reset(token)

    def record(self, attempt_id: str, response: httpx.Response) -> None:
        captured: Final = CapturedUpstreamResponse(
            attempt_id=attempt_id,
            status_code=response.status_code,
            headers=tuple(response.headers.multi_items()),
        )
        with self._lock:
            self._responses = (*self._responses, captured)

    def request_extensions(self, attempt_id: str) -> Mapping[str, object]:
        return MappingProxyType({_CAPTURE_EXTENSION: _CaptureBinding(self, attempt_id)})


@dataclass(frozen=True, slots=True)
class _CaptureBinding:
    recorder: UpstreamResponseCapture
    attempt_id: str


_active_capture: Final[ContextVar[_CaptureBinding | None]] = ContextVar("upstream_response_capture", default=None)
_CAPTURE_EXTENSION: Final = "litellm.upstream_response_capture"


def capture_response_headers(response: httpx.Response) -> None:
    binding: Final = _active_capture.get()
    if binding is not None:
        binding.recorder.record(binding.attempt_id, response)


async def async_capture_response_headers(response: httpx.Response) -> None:
    capture_response_headers(response)


def capture_explicit_response_headers(response: httpx.Response) -> None:
    binding: Final[object] = response.request.extensions.get(_CAPTURE_EXTENSION)
    if isinstance(binding, _CaptureBinding):
        binding.recorder.record(binding.attempt_id, response)


async def async_capture_explicit_response_headers(response: httpx.Response) -> None:
    capture_explicit_response_headers(response)
