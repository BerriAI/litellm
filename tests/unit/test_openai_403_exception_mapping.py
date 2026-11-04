"""
Unit test verifying that OpenAI HTTP 403 responses map to litellm.PermissionDeniedError.

Closes #49936: [Bug]: OpenAI HTTP 403 raises APIError or BadRequestError instead of PermissionDeniedError
"""

import pytest
import litellm
from litellm.exceptions import PermissionDeniedError
from litellm.litellm_core_utils.exception_mapping_utils import exception_type


class MockOpenAIHTTP403Exception(Exception):
    def __init__(self, status_code: int = 403, body: dict = None, message: str = "Forbidden"):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.message = message


def test_openai_403_request_forbidden_raises_permission_denied_error():
    """Verify HTTP 403 with request_forbidden body raises PermissionDeniedError."""
    exc = MockOpenAIHTTP403Exception(
        status_code=403,
        body={
            "error": {
                "message": "Country, region, or territory not supported",
                "type": "request_forbidden",
                "param": None,
                "code": "unsupported_country_region_territory",
            }
        },
        message="Country, region, or territory not supported",
    )

    with pytest.raises(PermissionDeniedError) as exc_info:
        exception_type(
            model="gpt-4o",
            original_exception=exc,
            custom_llm_provider="openai",
        )

    assert exc_info.value.status_code == 403
    assert isinstance(exc_info.value, PermissionDeniedError)


def test_openai_403_invalid_request_error_raises_permission_denied_error():
    """Verify HTTP 403 with invalid_request_error body raises PermissionDeniedError."""
    exc = MockOpenAIHTTP403Exception(
        status_code=403,
        body={
            "error": {
                "message": "Forbidden",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
        message="Forbidden: invalid_request_error",
    )

    with pytest.raises(PermissionDeniedError) as exc_info:
        exception_type(
            model="gpt-4o",
            original_exception=exc,
            custom_llm_provider="openai",
        )

    assert exc_info.value.status_code == 403
    assert isinstance(exc_info.value, PermissionDeniedError)
