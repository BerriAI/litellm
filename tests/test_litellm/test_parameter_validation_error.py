"""
Test parameter validation error handling.

This test verifies that TypeError exceptions from missing required parameters
are properly converted to BadRequestError with clear error messages.
"""
import sys
import os
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.exceptions import BadRequestError


def test_missing_required_parameter_sync():
    """
    Test that missing required parameters raise BadRequestError (sync).

    When a required parameter is missing from a completion call,
    it should raise BadRequestError instead of TypeError.
    """
    # Try calling completion without the required 'model' parameter
    # This should raise BadRequestError, not TypeError
    with pytest.raises(BadRequestError) as exc_info:
        litellm.completion(
            messages=[{"role": "user", "content": "test"}],
            # model parameter is intentionally omitted
        )

    # Verify the error message mentions the missing parameter
    error_message = str(exc_info.value)
    assert "missing" in error_message.lower() or "required" in error_message.lower()


@pytest.mark.asyncio
async def test_missing_required_parameter_async():
    """
    Test that missing required parameters raise BadRequestError (async).

    When a required parameter is missing from an async completion call,
    it should raise BadRequestError instead of TypeError.
    """
    # Try calling acompletion without the required 'model' parameter
    # This should raise BadRequestError, not TypeError
    with pytest.raises(BadRequestError) as exc_info:
        await litellm.acompletion(
            messages=[{"role": "user", "content": "test"}],
            # model parameter is intentionally omitted
        )

    # Verify the error message mentions the missing parameter
    error_message = str(exc_info.value)
    assert "missing" in error_message.lower() or "required" in error_message.lower()


@pytest.mark.asyncio
async def test_missing_required_parameter_aresponses():
    """
    Test that missing required parameters raise BadRequestError for aresponses.

    When a required parameter is missing from an aresponses call,
    it should raise BadRequestError instead of TypeError.
    """
    # Try calling aresponses without the required 'model' parameter
    # This should raise BadRequestError, not TypeError
    with pytest.raises(BadRequestError) as exc_info:
        await litellm.aresponses(
            input=[{"role": "user", "content": [{"type": "input_text", "text": "test"}], "type": "message"}],
            # model parameter is intentionally omitted
        )

    # Verify the error message mentions the missing parameter
    error_message = str(exc_info.value)
    assert "missing" in error_message.lower() or "required" in error_message.lower()


def test_bad_request_error_has_400_status_code():
    """
    Test that BadRequestError has HTTP 400 status code.

    This verifies that the error will be properly returned as
    HTTP 400 Bad Request when used in an API endpoint.
    """
    try:
        litellm.completion(
            messages=[{"role": "user", "content": "test"}],
            # model parameter is intentionally omitted
        )
    except BadRequestError as e:
        # Verify status code is 400
        assert e.status_code == 400, f"Expected status code 400, got {e.status_code}"
        assert hasattr(e, "status_code"), "BadRequestError should have status_code attribute"
        return

    # If we get here, the test failed
    assert False, "Expected BadRequestError to be raised"


@pytest.mark.asyncio
async def test_responses_api_wrong_parameter():
    """
    Test that using 'messages' instead of 'input' for responses API raises BadRequestError.

    Responses API expects 'input' parameter, not 'messages'.
    This test verifies that using the wrong parameter raises a clear BadRequestError.
    """
    # Try calling responses with 'messages' instead of 'input'
    # This should raise BadRequestError with a clear message
    with pytest.raises((BadRequestError, TypeError)) as exc_info:
        await litellm.aresponses(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],  # Wrong: should be 'input'
        )

    # If it's a BadRequestError, verify it has proper status code
    assert exc_info.value.status_code == 400


