import inspect
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


def _exception_description(exception):
    """Return a normalized description for OpenAPI / JSDoc consumers.

    Uses the class's own docstring (not an inherited one) so the rendered Swagger
    description matches the historical short-form (the class name) when only an
    upstream library defined a docstring. ``cleandoc`` strips the source-code
    indentation that would otherwise leak into the generated JSDoc comment in
    ``ui/litellm-dashboard/src/lib/http/schema.d.ts``.
    """
    doc = exception.__doc__
    if not doc:
        return exception.__name__
    return inspect.cleandoc(doc)


ERROR_RESPONSES = {
    get_status_code(exception): {
        "model": ErrorResponse,
        "description": _exception_description(exception),
    }
    for exception in LITELLM_EXCEPTION_TYPES
}

# Ensure we have a 500 error response
if 500 not in ERROR_RESPONSES:
    ERROR_RESPONSES[500] = {
        "model": ErrorResponse,
        "description": "Internal Server Error",
    }
