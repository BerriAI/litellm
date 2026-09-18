"""The SSE error frame for a ``/v1/messages`` stream that fails once the response is already open.

Anthropic clients pick stream events by the ``event:`` name, so a frame with only a ``data:`` line
is skipped and the failure never reaches the caller"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import status

from litellm.llms.anthropic.common_utils import ANTHROPIC_ERROR_STATUS_CODE_MAP

_ANTHROPIC_ERROR_TYPE_BY_STATUS: Final[Mapping[int, str]] = MappingProxyType(
    {status_code: error_type for error_type, status_code in ANTHROPIC_ERROR_STATUS_CODE_MAP.items()}
)


def anthropic_error_type(status_code: int) -> str:
    """Anthropic only uses the types in the map above, so a status outside it is grouped the way
    Anthropic groups them: client statuses are request errors, everything else is an API error"""
    mapped: Final = _ANTHROPIC_ERROR_TYPE_BY_STATUS.get(status_code)
    if mapped is not None:
        return mapped
    if status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "invalid_request_error"
    return "api_error"


def anthropic_error_sse_frame(message: str, status_code: int) -> str:
    """One error frame, ready to write into an open stream"""
    error_type: Final = json.dumps(anthropic_error_type(status_code))
    body: Final = json.dumps(message)
    return f'event: error\ndata: {{"type": "error", "error": {{"type": {error_type}, "message": {body}}}}}\n\n'
