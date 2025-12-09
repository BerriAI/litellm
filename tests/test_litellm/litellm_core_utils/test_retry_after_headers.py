"""
Test that retry-after headers are properly preserved for status codes 502, 503, and 504.

This test module verifies that:
1. BadGatewayError exception exists and preserves retry-after headers
2. ServiceUnavailableError preserves retry-after headers
3. InternalServerError preserves retry-after headers
4. All these exceptions are in LITELLM_EXCEPTION_TYPES
"""

import httpx

import litellm


def test_bad_gateway_error_preserves_retry_after_header():
    """Test that BadGatewayError preserves retry-after headers."""
    response = httpx.Response(
        status_code=502,
        headers=httpx.Headers({"retry-after": "30"}),
        request=httpx.Request("POST", "http://test.com"),
    )

    error = litellm.BadGatewayError(
        message="Test bad gateway error",
        llm_provider="test",
        model="test-model",
        response=response,
    )

    assert error.status_code == 502
    assert "BadGatewayError" in str(error)
    assert error.response is not None
    assert error.response.headers is not None
    assert "retry-after" in error.response.headers
    assert error.response.headers["retry-after"] == "30"


def test_service_unavailable_error_preserves_retry_after_header():
    """Test that ServiceUnavailableError preserves retry-after headers."""
    response = httpx.Response(
        status_code=503,
        headers=httpx.Headers({"retry-after": "60"}),
        request=httpx.Request("POST", "http://test.com"),
    )

    error = litellm.ServiceUnavailableError(
        message="Service unavailable",
        llm_provider="test",
        model="test-model",
        response=response,
    )

    assert error.status_code == 503
    assert error.response.headers is not None
    assert "retry-after" in error.response.headers
    assert error.response.headers["retry-after"] == "60"


def test_internal_server_error_preserves_retry_after_header():
    """Test that InternalServerError preserves retry-after headers."""
    response = httpx.Response(
        status_code=500,
        headers=httpx.Headers({"retry-after": "45"}),
        request=httpx.Request("POST", "http://test.com"),
    )

    error = litellm.InternalServerError(
        message="Internal server error",
        llm_provider="test",
        model="test-model",
        response=response,
    )

    assert error.status_code == 500
    assert error.response.headers is not None
    assert "retry-after" in error.response.headers
    assert error.response.headers["retry-after"] == "45"


def test_bad_gateway_error_in_exception_types():
    """Test that BadGatewayError is in LITELLM_EXCEPTION_TYPES."""
    assert litellm.BadGatewayError in litellm.LITELLM_EXCEPTION_TYPES


def test_bad_gateway_error_attributes():
    """Test that BadGatewayError has expected attributes."""
    response = httpx.Response(
        status_code=502,
        headers=httpx.Headers({"retry-after": "20"}),
        request=httpx.Request("POST", "http://test.com"),
    )

    error = litellm.BadGatewayError(
        message="Test error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
        response=response,
        max_retries=3,
        num_retries=1,
    )

    assert error.status_code == 502
    assert error.llm_provider == "openai"
    assert error.model == "gpt-3.5-turbo"
    assert error.max_retries == 3
    assert error.num_retries == 1
    assert "LiteLLM Retried: 1 times" in str(error)
    assert "LiteLLM Max Retries: 3" in str(error)


def test_exception_without_response_headers():
    """Test that exceptions work even without response headers."""
    # Test with None response
    error = litellm.BadGatewayError(
        message="Test error",
        llm_provider="test",
        model="test-model",
        response=None,
    )

    assert error.status_code == 502
    assert error.response is not None  # Should create a default response
    assert (
        error.response.headers is not None
    )  # Should have headers object (may be empty)


def test_exception_with_empty_headers():
    """Test that exceptions work with empty headers."""
    response = httpx.Response(
        status_code=502,
        headers=httpx.Headers({}),  # Empty headers
        request=httpx.Request("POST", "http://test.com"),
    )

    error = litellm.BadGatewayError(
        message="Test error",
        llm_provider="test",
        model="test-model",
        response=response,
    )

    assert error.status_code == 502
    assert error.response.headers is not None
    # Should not have retry-after since we didn't provide it
    assert "retry-after" not in error.response.headers
