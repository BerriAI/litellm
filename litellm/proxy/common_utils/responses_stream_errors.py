import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict

from litellm._logging import redact_internal_details_from_client_message
from litellm._uuid import uuid
from litellm.exceptions import MidStreamFallbackError
from litellm.types.llms.openai import ResponseFailedEvent, ResponsesAPIResponse, ResponsesAPIStreamEvents


class _ResponseIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: str | None = None
    model: str | None = None
    created_at: int | None = None


class _StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    type: str | None = None
    sequence_number: int | None = None
    response: _ResponseIdentity | None = None


class _FailureDetails(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    message: str | None = None
    code: str | int | None = None
    type: str | None = None
    status_code: int | None = None


def _original_failure(exception: Exception) -> Exception:
    if isinstance(exception, MidStreamFallbackError) and exception.original_exception is not None:
        return _original_failure(exception.original_exception)
    return exception


def _response_error_code(details: _FailureDetails) -> str:
    for value in (details.code, details.type):
        if value == "insufficient_quota":
            return "insufficient_quota"
        if value in (429, "429") or isinstance(value, str) and value.startswith("rate_limit"):
            return "rate_limit_exceeded"
    if isinstance(details.code, str) and details.code and not details.code.isdecimal():
        return details.code
    if details.status_code == 429:
        return "rate_limit_exceeded"
    return "server_error"


class ResponsesStreamErrorState:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.model: str | None = None
        self.created_at: int | None = None
        self.sequence_number = -1
        self.terminal_emitted = False

    @staticmethod
    def observe_chunk(chunk: object) -> _StreamEvent | None:
        if not isinstance(chunk, (BaseModel, Mapping)):
            return None
        return _StreamEvent.model_validate(chunk)

    def mark_emitted(self, event: _StreamEvent | None) -> None:
        if event is None:
            return
        if event.sequence_number is not None:
            self.sequence_number = max(self.sequence_number, event.sequence_number)
        if event.response is not None:
            self.response_id = event.response.id or self.response_id
            self.model = event.response.model or self.model
            if event.response.created_at is not None:
                self.created_at = event.response.created_at
        if event.type in ("response.completed", "response.failed", "response.incomplete"):
            self.terminal_emitted = True

    def format_failure(self, exception: Exception) -> str | None:
        if self.terminal_emitted:
            return None
        original: Final = _original_failure(exception)
        details: Final = _FailureDetails.model_validate(original)
        response: Final = ResponsesAPIResponse.model_validate(
            MappingProxyType(
                {
                    "id": self.response_id or f"resp_{uuid.uuid4().hex}",
                    "object": "response",
                    "created_at": self.created_at if self.created_at is not None else int(time.time()),
                    "model": self.model,
                    "status": "failed",
                    "output": (),
                    "error": MappingProxyType(
                        {
                            "code": _response_error_code(details),
                            "message": redact_internal_details_from_client_message(details.message or str(original)),
                        }
                    ),
                }
            )
        )
        event: Final = ResponseFailedEvent.model_validate(
            MappingProxyType(
                {
                    "type": ResponsesAPIStreamEvents.RESPONSE_FAILED,
                    "response": response,
                    "sequence_number": self.sequence_number + 1,
                }
            )
        )
        payload: Final = event.model_dump_json(exclude_none=True)
        self.terminal_emitted = True
        return f"event: response.failed\ndata: {payload}\n\n"
