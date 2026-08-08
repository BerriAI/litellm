"""Tests for litellm.error_categories — protocol-level error normalization."""

import pytest
from litellm.error_categories import (
    ErrorCategory,
    ParsedError,
    categorize_exception,
    default_parse_error,
    google_parse_error,
)


class TestDefaultParseError:
    """OpenAI-compatible & Anthropic error parsing (HTTP-status-based)."""

    def test_auth_401(self):
        result = default_parse_error({}, 401)
        assert result.category == ErrorCategory.AUTH
        assert result.status_code == 401

    def test_auth_403(self):
        result = default_parse_error({}, 403)
        assert result.category == ErrorCategory.AUTH

    def test_rate_limit_429(self):
        result = default_parse_error({}, 429)
        assert result.category == ErrorCategory.RATE_LIMIT

    def test_server_500(self):
        result = default_parse_error({}, 500)
        assert result.category == ErrorCategory.SERVER
        assert result.message is not None

    def test_server_502(self):
        result = default_parse_error({}, 502)
        assert result.category == ErrorCategory.SERVER

    def test_server_503(self):
        result = default_parse_error({}, 503)
        assert result.category == ErrorCategory.SERVER

    def test_client_400(self):
        result = default_parse_error({}, 400)
        assert result.category == ErrorCategory.CLIENT

    def test_client_404(self):
        result = default_parse_error({}, 404)
        assert result.category == ErrorCategory.CLIENT

    def test_extracts_message_from_body(self):
        result = default_parse_error({"error": {"message": "Bad request"}}, 400)
        assert result.message == "Bad request"

    def test_no_crash_on_empty_body(self):
        result = default_parse_error({}, 500)
        assert result.category == ErrorCategory.SERVER

    def test_no_crash_on_invalid_body(self):
        """Should not crash if error field is not a dict."""
        result = default_parse_error({"error": "string error"}, 500)
        assert result.category == ErrorCategory.SERVER


class TestGoogleParseError:
    """Google Gemini / Vertex AI error parsing (status-string-aware)."""

    def test_http_auth(self):
        result = google_parse_error({}, 401)
        assert result.category == ErrorCategory.AUTH

    def test_body_unauthenticated(self):
        result = google_parse_error({"error": {"status": "UNAUTHENTICATED"}}, 200)
        assert result.category == ErrorCategory.AUTH

    def test_rate_limit(self):
        result = google_parse_error({}, 429)
        assert result.category == ErrorCategory.RATE_LIMIT

    def test_body_resource_exhausted(self):
        result = google_parse_error({"error": {"status": "RESOURCE_EXHAUSTED"}}, 200)
        assert result.category == ErrorCategory.RATE_LIMIT

    def test_server_unavailable(self):
        result = google_parse_error({"error": {"status": "UNAVAILABLE"}}, 200)
        assert result.category == ErrorCategory.SERVER

    def test_server_internal(self):
        result = google_parse_error({"error": {"status": "INTERNAL"}}, 200)
        assert result.category == ErrorCategory.SERVER

    def test_client_fallback(self):
        result = google_parse_error({"error": {"message": "Invalid argument"}}, 400)
        assert result.category == ErrorCategory.CLIENT
        assert result.message == "Invalid argument"

    def test_body_permission_denied(self):
        """PERMISSION_DENIED should map to AUTH."""
        result = google_parse_error({"error": {"status": "PERMISSION_DENIED"}}, 200)
        assert result.category == ErrorCategory.AUTH

    def test_body_deadline_exceeded(self):
        """DEADLINE_EXCEEDED should map to SERVER (retryable)."""
        result = google_parse_error({"error": {"status": "DEADLINE_EXCEEDED"}}, 200)
        assert result.category == ErrorCategory.SERVER


class TestCategorizeException:
    """Integration: extract ErrorCategory from existing LiteLLM exceptions."""

    def test_exception_with_category_attr(self):
        exc = Exception()
        exc.error_category = ErrorCategory.RATE_LIMIT  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.RATE_LIMIT

    def test_exception_with_status_code(self):
        exc = Exception()
        exc.status_code = 429  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.RATE_LIMIT

    def test_exception_by_name_auth(self):
        class AuthenticationError(Exception):
            pass

        assert categorize_exception(AuthenticationError()) == ErrorCategory.AUTH

    def test_exception_by_name_rate(self):
        class RateLimitError(Exception):
            pass

        assert categorize_exception(RateLimitError()) == ErrorCategory.RATE_LIMIT

    def test_exception_by_name_server(self):
        class ServiceUnavailableError(Exception):
            pass

        assert categorize_exception(ServiceUnavailableError()) == ErrorCategory.SERVER

    def test_exception_by_name_throttle(self):
        class ThrottlingError(Exception):
            pass

        assert categorize_exception(ThrottlingError()) == ErrorCategory.RATE_LIMIT

    def test_exception_by_name_timeout(self):
        class TimeoutError(Exception):
            pass

        assert categorize_exception(TimeoutError()) == ErrorCategory.SERVER

    def test_exception_with_status_code_401(self):
        exc = Exception()
        exc.status_code = 401  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.AUTH

    def test_exception_with_status_code_403(self):
        exc = Exception()
        exc.status_code = 403  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.AUTH

    def test_exception_with_status_code_500(self):
        exc = Exception()
        exc.status_code = 500  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.SERVER

    def test_exception_with_status_code_503(self):
        exc = Exception()
        exc.status_code = 503  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.SERVER

    def test_exception_with_status_code_400(self):
        exc = Exception()
        exc.status_code = 400  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.CLIENT

    def test_exception_with_status_code_404(self):
        exc = Exception()
        exc.status_code = 404  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.CLIENT

    def test_exception_with_status_code_408_timeout(self):
        """408 Request Timeout should be SERVER (retryable), not CLIENT."""
        exc = Exception()
        exc.status_code = 408  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.SERVER

    def test_exception_with_string_status_code(self):
        """String status_code should be converted to int."""
        exc = Exception()
        exc.status_code = "503"  # type: ignore[attr-defined]
        assert categorize_exception(exc) == ErrorCategory.SERVER

    def test_exception_with_invalid_status_code(self):
        """Invalid status_code should fall through to name heuristics."""
        exc = Exception()
        exc.status_code = "invalid"  # type: ignore[attr-defined]
        # Falls through to None since no name match
        assert categorize_exception(exc) is None

    def test_unknown_returns_none(self):
        assert categorize_exception(ValueError("unexpected")) is None


class TestErrorCategoryEnum:
    """ErrorCategory is a string enum for easy serialization."""

    def test_values(self):
        assert ErrorCategory.AUTH.value == "auth"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"
        assert ErrorCategory.SERVER.value == "server"
        assert ErrorCategory.CLIENT.value == "client"

    def test_is_str(self):
        assert isinstance(ErrorCategory.AUTH, str)


class TestParsedError:
    """ParsedError is an immutable value object."""

    def test_frozen(self):
        err = ParsedError(category=ErrorCategory.AUTH, message="Unauthorized")
        with pytest.raises(Exception):
            err.category = ErrorCategory.CLIENT  # type: ignore[misc]

    def test_repr(self):
        err = ParsedError(category=ErrorCategory.SERVER, status_code=503)
        assert "server" in repr(err)
