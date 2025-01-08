from typing import Any, Dict

from pydantic import BaseModel, Field

from litellm.exceptions import LITELLM_EXCEPTION_TYPES


class ErrorResponse(BaseModel):
    detail: Dict[str, Any] = Field(
        ...,
        example={  # type: ignore
            "error": {
                "message": "Error message",
                "type": "error_type",
                "param": "error_param",
                "code": "error_code",
            }
        },
    )


# Define a function to get the status code
def get_status_code(exception):
    if hasattr(exception, "status_code"):
        return exception.status_code
    # Default status codes for exceptions without a status_code attribute
    if exception.__name__ == "Timeout":
        return 408  # Request Timeout
    if exception.__name__ == "APIConnectionError":
        return 503  # Service Unavailable
    return 500  # Internal Server Error as default


# Create error responses
ERROR_RESPONSES = {
    get_status_code(exception): {
        "model": ErrorResponse,
        "description": exception.__doc__ or exception.__name__,
    }
    for exception in LITELLM_EXCEPTION_TYPES
}

# Ensure we have a 500 error response
if 500 not in ERROR_RESPONSES:
    ERROR_RESPONSES[500] = {
        "model": ErrorResponse,
        "description": "Internal Server Error",
    }
