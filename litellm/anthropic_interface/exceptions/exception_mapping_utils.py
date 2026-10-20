"""
Utilities for mapping exceptions to Anthropic error format.

Similar to litellm/litellm_core_utils/exception_mapping_utils.py but for Anthropic response format.
"""

import json
from typing import Final

from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

from .exceptions import AnthropicErrorDetail, AnthropicErrorResponse, AnthropicErrorType

# HTTP status code -> Anthropic error type
# Source: https://docs.anthropic.com/en/api/errors
ANTHROPIC_ERROR_TYPE_MAP: Final[dict[int, AnthropicErrorType]] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


class AnthropicExceptionMapping:
    """
    Helper class for mapping exceptions to Anthropic error format.

    Similar pattern to ExceptionCheckers in litellm_core_utils/exception_mapping_utils.py
    """

    @staticmethod
    def get_error_type(status_code: int) -> AnthropicErrorType:
        """Map HTTP status code to Anthropic error type."""
        return ANTHROPIC_ERROR_TYPE_MAP.get(status_code, "api_error")

    @staticmethod
    def create_error_response(
        status_code: int,
        message: str,
        request_id: str | None = None,
    ) -> AnthropicErrorResponse:
        """
        Create an Anthropic-formatted error response dict.

        Anthropic error format:
        {
            "type": "error",
            "error": {"type": "...", "message": "..."},
            "request_id": "req_..."
        }
        """
        error_type: Final = AnthropicExceptionMapping.get_error_type(status_code)

        response: Final[AnthropicErrorResponse] = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }

        if request_id:
            response["request_id"] = request_id

        return response

    @staticmethod
    def extract_error_message(raw_message: str) -> str:
        """
        Extract error message from various provider response formats.

        Handles:
        - Bedrock: {"detail": {"message": "..."}}
        - AWS: {"Message": "..."}
        - Generic: {"message": "..."}
        - Plain strings
        """
        parsed: Final = safe_json_loads(raw_message)
        if isinstance(parsed, dict):
            # Bedrock format
            if "detail" in parsed and isinstance(parsed["detail"], dict):
                return parsed["detail"].get("message", raw_message)
            # AWS/generic format
            return parsed.get("Message") or parsed.get("message") or raw_message
        return raw_message

    @staticmethod
    def _is_anthropic_error_dict(parsed: dict) -> bool:
        """
        Check if a parsed dict is in Anthropic error format.

        Anthropic error format:
        {
            "type": "error",
            "error": {"type": "...", "message": "..."}
        }
        """
        return (
            parsed.get("type") == "error"
            and isinstance(parsed.get("error"), dict)
            and "type" in parsed["error"]
            and "message" in parsed["error"]
        )

    @staticmethod
    def _extract_message_from_dict(parsed: dict, raw_message: str) -> str:
        """
        Extract error message from a parsed provider-specific dict.

        Handles:
        - Bedrock: {"detail": {"message": "..."}}
        - AWS: {"Message": "..."}
        - Generic: {"message": "..."}
        """
        # Bedrock format
        if "detail" in parsed and isinstance(parsed["detail"], dict):
            return parsed["detail"].get("message", raw_message)
        # AWS/generic format
        return parsed.get("Message") or parsed.get("message") or raw_message

    @staticmethod
    def transform_to_anthropic_error(
        status_code: int,
        raw_message: str,
        request_id: str | None = None,
    ) -> AnthropicErrorResponse:
        """
        Transform an error message to Anthropic format.

        - If already in Anthropic format: passthrough unchanged
        - Otherwise: extract message and create Anthropic error

        Parses JSON only once for efficiency.

        Args:
            status_code: HTTP status code
            raw_message: Raw error message (may be JSON string or plain text)
            request_id: Optional request ID to include

        Returns:
            AnthropicErrorResponse dict
        """
        # Try to parse as JSON once
        parsed: dict | None = safe_json_loads(raw_message)
        if not isinstance(parsed, dict):
            parsed = None

        # If parsed and already in Anthropic format - passthrough
        if parsed is not None and AnthropicExceptionMapping._is_anthropic_error_dict(parsed):
            # Optionally add request_id if provided and not present
            if request_id and "request_id" not in parsed:
                parsed["request_id"] = request_id
            return parsed

        # Extract message - use parsed dict if available, otherwise raw string
        if parsed is not None:
            message = AnthropicExceptionMapping._extract_message_from_dict(parsed, raw_message)
        else:
            message = raw_message

        return AnthropicExceptionMapping.create_error_response(
            status_code=status_code,
            message=message,
            request_id=request_id,
        )


class AnthropicErrorSseFrame(str):
    """One `event: error` frame, for a stream that fails once the response headers are out.

    Anthropic clients pick stream events by the `event:` name, so a frame carrying only a `data:`
    line is skipped and the failure never reaches the caller. The frame remembers the status and
    body it was built from, so a stream that fails before its first byte can still answer as a
    JSON error with that exact status instead of a 200 that only says `api_error`
    """

    status_code: int
    error_response: AnthropicErrorResponse

    def __new__(cls, status_code: int, error_response: AnthropicErrorResponse) -> "AnthropicErrorSseFrame":
        frame: Final = super().__new__(cls, f"event: error\ndata: {json.dumps(error_response)}\n\n")
        frame.status_code = status_code
        frame.error_response = error_response
        return frame

    def json_body(self, call_id: str | None) -> AnthropicErrorResponse:
        if call_id is None:
            return self.error_response
        detail: Final[AnthropicErrorDetail] = {**self.error_response["error"], "litellm_call_id": call_id}
        body: Final[AnthropicErrorResponse] = {**self.error_response, "error": detail}
        return body


def anthropic_error_sse_frame(status_code: int, raw_message: str) -> AnthropicErrorSseFrame:
    return AnthropicErrorSseFrame(
        status_code,
        AnthropicExceptionMapping.transform_to_anthropic_error(status_code=status_code, raw_message=raw_message),
    )
