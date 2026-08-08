"""
Tests for _get_openai_exception_status_code.

Regression test for https://github.com/BerriAI/litellm/issues/35860:
a missing API key raises openai.OpenAIError at client construction (with no
status_code). Defaulting that to 500 makes it a retryable InternalServerError,
so a permanent credential/config error is retried instead of failing fast.
It should map to 401 (AuthenticationError) instead.
"""

import httpx
import openai

from litellm.llms.openai.openai import _get_openai_exception_status_code


class TestGetOpenAIExceptionStatusCode:
    def test_openai_error_without_status_code_maps_to_401(self):
        """openai.OpenAIError with no status_code (pre-request, e.g. missing key) -> 401."""
        err = openai.OpenAIError("Missing credentials. Please pass an `api_key`.")
        assert not hasattr(err, "status_code") or getattr(err, "status_code") is None
        assert _get_openai_exception_status_code(err) == 401

    def test_transient_openai_subclasses_stay_retryable(self):
        """APIConnectionError / APITimeoutError subclass OpenAIError and carry no
        status_code, but they are genuinely transient and must stay retryable
        (500), not be turned into a non-retryable 401. Only the bare
        OpenAIError base class (missing credentials) maps to 401."""
        conn_err = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1"))
        timeout_err = openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1"))
        assert _get_openai_exception_status_code(conn_err) == 500
        assert _get_openai_exception_status_code(timeout_err) == 500


    def test_existing_status_code_is_preserved(self):
        """An exception that already carries a status_code keeps it (real HTTP response)."""

        class FakeHTTPError(Exception):
            status_code = 429

        assert _get_openai_exception_status_code(FakeHTTPError("rate limited")) == 429

    def test_status_code_zero_is_preserved(self):
        """A falsy-but-present status_code (0) is still honored, not defaulted."""

        class ZeroStatus(Exception):
            status_code = 0

        assert _get_openai_exception_status_code(ZeroStatus("weird")) == 0

    def test_generic_exception_without_status_code_defaults_to_500(self):
        """A non-OpenAI exception with no status_code keeps the 500 default."""
        assert _get_openai_exception_status_code(ValueError("boom")) == 500

    def test_openai_error_subclass_with_status_code_kept(self):
        """A real openai HTTP error (has status_code) is unchanged."""

        class FakeOpenAIHTTPError(openai.OpenAIError):
            status_code = 400

        assert _get_openai_exception_status_code(FakeOpenAIHTTPError("bad request")) == 400
