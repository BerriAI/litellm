import httpx
import openai
import pytest

from litellm.llms.openai.openai import _status_code_for_openai_sdk_error


def test_missing_credentials_is_not_a_server_error():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/35860

    The SDK raises a bare OpenAIError at client construction when no key is
    configured. That happens before any HTTP exchange, so calling it a 500 marks a
    permanently unfixable configuration error as a retryable server error.
    """
    with pytest.raises(openai.OpenAIError) as exc_info:
        openai.OpenAI(api_key=None)

    error = exc_info.value
    assert not hasattr(error, "status_code")
    assert _status_code_for_openai_sdk_error(error) == 400


def test_connection_error_stays_retryable():
    """Transient failures must keep the 500 default so retries still happen."""
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://example.com"))

    assert not hasattr(error, "status_code")
    assert _status_code_for_openai_sdk_error(error) == 500


def test_existing_status_code_is_preserved():
    class _WithStatus(Exception):
        status_code = 429

    assert _status_code_for_openai_sdk_error(_WithStatus()) == 429


def test_unknown_exception_keeps_the_server_error_default():
    """Anything that is not a bare OpenAIError keeps the previous behaviour."""
    assert _status_code_for_openai_sdk_error(ValueError("boom")) == 500


def test_status_code_zero_is_preserved():
    """A falsy-but-present status code must not fall through to the default."""

    class _ZeroStatus(Exception):
        status_code = 0

    assert _status_code_for_openai_sdk_error(_ZeroStatus()) == 0
