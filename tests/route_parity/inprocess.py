from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from tests.route_parity.models import CapturedRequest
from tests.route_parity.recorded_http import RecordedResponse
from tests.route_parity.replay import ReplayServer

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
        response = call(provider.url)
        return InProcessExecution(requests=provider.take_requests(len(recorded_responses)), response=response)
    except Exception:
        provider.reset()
        raise
