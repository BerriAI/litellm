from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from .models import CapturedRequest
from .recorded_http import RecordedResponse
from .replay import ReplayServer

ResponseT = TypeVar("ResponseT")


@dataclass(frozen=True, slots=True)
class InProcessExecution(Generic[ResponseT]):
    requests: tuple[CapturedRequest, ...]
    response: ResponseT


def run_in_process(
    provider: ReplayServer,
    recorded_responses: tuple[RecordedResponse, ...],
    call: Callable[[str], ResponseT],
) -> InProcessExecution[ResponseT]:
    for recorded_response in recorded_responses:
        provider.enqueue_response(recorded_response)
    try:
        response: Final = call(provider.url)
        return InProcessExecution(requests=provider.take_requests(len(recorded_responses)), response=response)
    except Exception:
        provider.reset()
        raise


async def run_in_process_async(
    provider: ReplayServer,
    recorded_responses: tuple[RecordedResponse, ...],
    call: Callable[[str], Awaitable[ResponseT]],
) -> InProcessExecution[ResponseT]:
    for recorded_response in recorded_responses:
        provider.enqueue_response(recorded_response)
    try:
        response: Final = await call(provider.url)
        return InProcessExecution(requests=provider.take_requests(len(recorded_responses)), response=response)
    except Exception:
        provider.reset()
        raise
