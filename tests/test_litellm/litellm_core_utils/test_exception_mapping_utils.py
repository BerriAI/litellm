import os
import sys

import pytest

import litellm

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.exception_mapping_utils import (
    ExceptionCheckers,
    exception_type,
    extract_and_raise_litellm_exception,
)
from litellm.llms.openai.common_utils import OpenAIError

# Test cases for is_error_str_context_window_exceeded
# Tuple format: (error_message, expected_result)
context_window_test_cases = [
    # Positive cases (should return True)
    (
        "An error occurred: The input exceeds the model's maximum context limit of 8192 tokens.",
        True,
    ),
    (
        "Some text before, this model's maximum context length is 4096 tokens. Some text after.",
        True,
    ),
    (
        "Validation Error: string too long. expected a string with maximum length 1000.",
        True,
    ),
    ("Your prompt is longer than the model's context length of 2048.", True),
    ("AWS Bedrock Error: The request payload size has exceed context limit.", True),
    (
        "Input tokens exceed the configured limit of 272000 tokens. Your messages resulted in 509178 tokens. Please reduce the length of the messages.",
        True,
    ),
    (
        "`inputs` tokens + `max_new_tokens` must be <= 4096",
        True,
    ),
    (
        "request (67311 tokens) exceeds the available context size (65536 tokens), try increasing it",
        True,
    ),
    # Gemini 2.5/3 format
    (
        "The input token count exceeds the maximum number of tokens allowed 1048576.",
        True,
    ),
    (
        'GeminiException BadRequestError - {\n  "error": {\n    "code": 400,\n    "message": "The input token count exceeds the maximum number of tokens allowed 1048576.",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n',
        True,
    ),
    # Gemini 2.0 Flash format (includes input token count in message)
    (
        "The input token count (2800010) exceeds the maximum number of tokens allowed (1048575).",
        True,
    ),
    (
        'GeminiException BadRequestError - {\n  "error": {\n    "code": 400,\n    "message": "The input token count (2800010) exceeds the maximum number of tokens allowed (1048575).",\n    "status": "INVALID_ARGUMENT"\n  }\n}\n',
        True,
    ),
    # Test case insensitivity
    ("ERROR: THIS MODEL'S MAXIMUM CONTEXT LENGTH IS 1024.", True),
    # Cerebras context window error format
    # See: https://github.com/BerriAI/litellm/issues/XXXX
    (
        "Current length is 132784 while limit is 131000",
        True,
    ),
    (
        "CerebrasException - Please reduce the length of the messages or completion. Current length is 50000 while limit is 40000",
        True,
    ),
    # Negative cases (should return False)
    ("A generic API error occurred.", False),
    ("Invalid API Key provided.", False),
    ("Rate limit reached for requests.", False),
    ("The context is large, but acceptable.", False),
    ("", False),  # Empty string
    # OpenAI user param length validation - not a context window error
    (
        "Invalid 'user': string too long. Expected a string with maximum length 64, but got a string with length 123 instead.",
        False,
    ),
    (
        '{"error": {"message": "Invalid \'user\': string too long.", "code": "string_above_max_length"}}',
        False,
    ),
]


@pytest.mark.parametrize("error_str, expected", context_window_test_cases)
def test_is_error_str_context_window_exceeded(error_str, expected):
    """
    Tests the is_error_str_context_window_exceeded function with various error strings.
    """
    assert ExceptionCheckers.is_error_str_context_window_exceeded(error_str) == expected


class TestExceptionCheckers:
    """Test the ExceptionCheckers utility methods"""

    def test_is_error_str_rate_limit_ignores_embedded_numbers(self):
        """An arbitrary 429 inside user-provided payload must not trigger rate-limit detection"""

        error_str = "Invalid user message={'role': 'user', 'content': [{'text': 'payload429snippet'}]}"
        result = ExceptionCheckers.is_error_str_rate_limit(error_str)
        assert result is False

    def test_is_error_str_rate_limit_detects_true_rate_limit(self):
        """A real rate-limit error string should still be detected"""

        error_str = "RateLimitError: OpenAIException - You exceeded your current quota. (status code 429)"
        result = ExceptionCheckers.is_error_str_rate_limit(error_str)
        assert result is True

    def test_is_error_str_rate_limit_ignores_field_name_substring(self):
        """
        Regression: an unrelated upstream 400 whose body happens to contain
        ``rate_limit`` as a substring of an identifier (e.g. a field name
        like ``_litellm_rate_limit_descriptors`` echoed back by OpenAI in an
        "Unknown parameter" / "Unrecognized request arguments supplied"
        error) must NOT be classified as a rate-limit signal.

        Without a word boundary the regex ``rate[\\s_\\-]*limit`` matched
        the embedded ``rate_limit`` and turned a 400 BadRequest into a 429
        RateLimitError, hiding real proxy bugs (PR #27001 metadata leak)
        behind a misleading ``throttling_error`` code.
        """
        for error_str in [
            "Unknown parameter: '_litellm_rate_limit_descriptors'.",
            "Unrecognized request arguments supplied: _litellm_rate_limit_descriptors, _litellm_tpm_reserved_tokens",
            "my_rate_limit_field is not allowed",
        ]:
            assert (
                ExceptionCheckers.is_error_str_rate_limit(error_str) is False
            ), f"Field-name substring should not trigger rate-limit signal: {error_str!r}"

    def test_is_error_str_rate_limit_still_matches_standalone_token(self):
        """Both standalone forms still match — the tightening only affects
        embedded substrings."""
        for error_str in [
            "Rate limit exceeded for tokens",
            "rate_limit_exceeded: try again later",
            "rate-limit-exceeded",
            "You hit a rate limit",
        ]:
            assert (
                ExceptionCheckers.is_error_str_rate_limit(error_str) is True
            ), f"Standalone rate-limit phrase should match: {error_str!r}"


class TestExceptionTypeStatusCodeGate:
    """
    Integration tests for the status-code gate on the OpenAI rate-limit
    heuristic. Even if a 400 message happens to contain a substring that
    trips the string heuristic, an explicit upstream 400 must map to
    ``BadRequestError`` — never ``RateLimitError`` — because confusing the
    two changes whether the client retries.
    """

    def _make_openai_400(self, message: str) -> Exception:
        """Build something shaped like an upstream OpenAI 400."""
        import httpx
        import openai

        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(status_code=400, request=request)
        err = openai.BadRequestError(
            message=message,
            response=response,
            body={"message": message},
        )
        # openai.BadRequestError sets status_code via response, but assert
        # the surface the mapper actually reads.
        assert getattr(err, "status_code", None) == 400
        return err

    def test_openai_400_with_unknown_parameter_maps_to_bad_request(self):
        """
        PR #27001 regression: the v3 limiter leaked ``_litellm_*`` keys
        into the upstream body and OpenAI rejected the request with::

            400 Unknown parameter: '_litellm_rate_limit_descriptors'.

        The exception mapper turned that 400 into a 429
        ``RateLimitError`` / ``throttling_error`` because the message
        substring matched ``rate[\\s_\\-]*limit``. That hid the real bug
        and signalled "throttle and retry" to clients, which would loop
        forever on a malformed request. Lock in the fix.
        """
        from litellm.exceptions import BadRequestError, RateLimitError

        upstream = self._make_openai_400(
            "Unknown parameter: '_litellm_rate_limit_descriptors'."
        )

        with pytest.raises(BadRequestError) as exc_info:
            exception_type(
                model="gpt-4o-mini",
                original_exception=upstream,
                custom_llm_provider="openai",
                completion_kwargs={},
                extra_kwargs={},
            )
        assert exc_info.value.status_code == 400
        assert not isinstance(exc_info.value, RateLimitError)

    def test_openai_400_with_unrecognized_arguments_maps_to_bad_request(self):
        """Same regression, second OpenAI error wording observed in the wild."""
        from litellm.exceptions import BadRequestError, RateLimitError

        upstream = self._make_openai_400(
            "Unrecognized request arguments supplied: "
            "_litellm_rate_limit_descriptors, _litellm_tpm_reserved_model, "
            "_litellm_tpm_reserved_scopes, _litellm_tpm_reserved_tokens."
        )

        with pytest.raises(BadRequestError) as exc_info:
            exception_type(
                model="gpt-4o-mini",
                original_exception=upstream,
                custom_llm_provider="openai",
                completion_kwargs={},
                extra_kwargs={},
            )
        assert exc_info.value.status_code == 400
        assert not isinstance(exc_info.value, RateLimitError)

    def test_openai_429_still_maps_to_rate_limit(self):
        """The status-code gate must not regress the genuine 429 path."""
        import httpx
        import openai

        from litellm.exceptions import RateLimitError

        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(status_code=429, request=request)
        upstream = openai.RateLimitError(
            message="Rate limit reached for requests",
            response=response,
            body={"message": "Rate limit reached for requests"},
        )

        with pytest.raises(RateLimitError) as exc_info:
            exception_type(
                model="gpt-4o-mini",
                original_exception=upstream,
                custom_llm_provider="openai",
                completion_kwargs={},
                extra_kwargs={},
            )
        assert exc_info.value.status_code == 429

    def test_is_azure_content_policy_violation_error_with_policy_violation_text(self):
        """Test detection of Azure content policy violation with explicit policy violation text"""

        error_strings = [
            "invalid_request_error content_policy_violation occurred",
            "The response was filtered due to the prompt triggering Azure OpenAI's content management policy",
            "Your task failed as a result of our safety system detecting harmful content",
            "The model produced invalid content that violates our policy",
            "Request blocked due to content_filter_policy restrictions",
        ]

        for error_str in error_strings:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(
                error_str
            )
            assert result is True, f"Should detect policy violation in: {error_str}"

    def test_is_azure_content_policy_violation_error_case_insensitive(self):
        """Test that content policy violation detection is case insensitive"""

        error_strings = [
            "INVALID_REQUEST_ERROR CONTENT_POLICY_VIOLATION",
            "The Response Was Filtered Due To The Prompt Triggering Azure OpenAI's Content Management",
            "YOUR TASK FAILED AS A RESULT OF OUR SAFETY SYSTEM",
            "Content_Filter_Policy restriction detected",
        ]

        for error_str in error_strings:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(
                error_str
            )
            assert (
                result is True
            ), f"Should detect policy violation in uppercase: {error_str}"

    def test_is_azure_content_policy_violation_error_with_non_policy_errors(self):
        """Test that non-policy violation errors are not detected as policy violations"""

        error_strings = [
            "Invalid API key provided",
            "Rate limit exceeded for current model",
            "Model not found: gpt-nonexistent",
            "Request timeout occurred",
            "Authentication failed",
            "Insufficient quota remaining",
            "Bad request format",
            "Internal server error occurred",
        ]

        for error_str in error_strings:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(
                error_str
            )
            assert (
                result is False
            ), f"Should NOT detect policy violation in: {error_str}"

    def test_is_azure_content_policy_violation_error_with_partial_matches(self):
        """Test that partial keyword matches work correctly"""

        # These should match because they contain the required substrings
        positive_cases = [
            "Error: content_policy_violation detected in request",
            "Safety content management, your task failed as a result of our safety system",
            "the model produced invalid content",
        ]

        for error_str in positive_cases:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(
                error_str
            )
            assert result is True, f"Should detect policy violation in: {error_str}"

        # These should not match even though they contain similar words
        negative_cases = [
            "Invalid content format in request",  # "invalid" but not "invalid content"
            "Policy configuration error",  # "policy" but not policy violation context
            "Content type not supported",  # "content" but not content filter context
            "Management API unavailable",  # "management" but not content management context
        ]

        for error_str in negative_cases:
            result = ExceptionCheckers.is_azure_content_policy_violation_error(
                error_str
            )
            assert (
                result is False
            ), f"Should NOT detect policy violation in: {error_str}"


gemini_context_window_test_cases = [
    # Gemini 2.0 Flash format (includes input token count in message)
    (
        "The input token count (2800010) exceeds the maximum number of tokens allowed (1048575).",
        True,
    ),
    # Gemini 2.5/3 format
    (
        "The input token count exceeds the maximum number of tokens allowed (1048576).",
        True,
    ),
    ("A generic error occurred.", False),
]


@pytest.mark.parametrize(
    "error_message, should_raise_context_window", gemini_context_window_test_cases
)
def test_gemini_context_window_error_mapping(
    error_message, should_raise_context_window
):
    """
    Tests that the exception_type function correctly maps Gemini's
    context window exceeded errors to litellm.ContextWindowExceededError.
    """
    model = "gemini/gemini-2.0-flash"
    custom_llm_provider = "gemini"

    # Create a generic exception with the specific error message
    original_exception = Exception(error_message)

    if should_raise_context_window:
        with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
            exception_type(
                model=model,
                original_exception=original_exception,
                custom_llm_provider=custom_llm_provider,
            )
        # Check if the raised exception is indeed a ContextWindowExceededError
        assert isinstance(excinfo.value, litellm.ContextWindowExceededError)
    else:
        # For the negative case, we expect it to raise a generic APIConnectionError
        with pytest.raises(litellm.APIConnectionError):
            exception_type(
                model=model,
                original_exception=original_exception,
                custom_llm_provider=custom_llm_provider,
            )


def test_lemonade_context_window_error_mapping():
    """Lemonade's llama.cpp backend should map context overflows to LiteLLM's standard error."""

    model = "lemonade/Qwen3.6-35B-A3B-GGUF"
    error_message = (
        '{"error":{"code":"context_length_exceeded","message":"request '
        "(80010 tokens) exceeds the available context size (65536 tokens), "
        'try increasing it","status_code":400,"type":"invalid_request_error"}}'
    )
    original_exception = OpenAIError(
        status_code=400,
        message=error_message,
        headers={},
    )

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model=model,
            original_exception=original_exception,
            custom_llm_provider="lemonade",
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.llm_provider == "lemonade"
    assert excinfo.value.model == model


@pytest.mark.parametrize(
    "error_message",
    [
        "AnthropicException - prompt is too long: 250000 tokens > 200000 maximum",
        "AnthropicException - input length and max_tokens exceed context limit: "
        "200000 + 8000 > 200000, decrease input length or max_tokens and try again",
    ],
)
def test_anthropic_context_window_error_mapping(error_message):
    """Anthropic context-window overflows (input too long, or input + max_tokens
    over the context limit) must map to ContextWindowExceededError (400) even when
    the upstream exception carries no ``status_code`` attribute. Previously only
    "prompt is too long" was special-cased, so the "exceed context limit" phrasing
    fell through to a generic APIConnectionError (500)."""
    original_exception = Exception(error_message)

    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        exception_type(
            model="claude-sonnet-4-5",
            original_exception=original_exception,
            custom_llm_provider="anthropic",
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.llm_provider == "anthropic"


# Test cases for Vertex AI RateLimitError mapping
# As per https://github.com/BerriAI/litellm/issues/16189
vertex_rate_limit_test_cases = [
    ("429 Quota exceeded for model", True),
    ("Resource exhausted. Please try again later.", True),
    (
        "429 Unable to submit request because the service is temporarily out of capacity.",
        True,
    ),
    ("A generic error occurred.", False),  # Negative case
]


@pytest.mark.parametrize(
    "error_message, should_raise_rate_limit", vertex_rate_limit_test_cases
)
def test_vertex_ai_rate_limit_error_mapping(error_message, should_raise_rate_limit):
    """
    Tests that the exception_type function correctly maps Vertex AI's
    "Resource exhausted" error to a litellm.RateLimitError.
    """
    model = "gemini/gemini-2.5-flash"
    custom_llm_provider = "vertex_ai"

    # Create a generic exception with the specific error message
    original_exception = Exception(error_message)

    if should_raise_rate_limit:
        with pytest.raises(litellm.RateLimitError) as excinfo:
            exception_type(
                model=model,
                original_exception=original_exception,
                custom_llm_provider=custom_llm_provider,
            )
        # Check if the raised exception is indeed a RateLimitError
        assert isinstance(excinfo.value, litellm.RateLimitError)
    else:
        # For the negative case, we expect it to raise a generic APIConnectionError
        with pytest.raises(litellm.APIConnectionError):
            exception_type(
                model=model,
                original_exception=original_exception,
                custom_llm_provider=custom_llm_provider,
            )


class TestExtractAndRaiseLitellmException:
    """Tests for extract_and_raise_litellm_exception function"""

    def test_extract_and_raise_api_connection_error_without_response(self):
        """
        Test that APIConnectionError can be raised without response parameter.

        This is a regression test for the bug where extract_and_raise_litellm_exception
        would fail with TypeError when trying to raise APIConnectionError with a
        response parameter, since APIConnectionError doesn't accept that parameter.

        Relevant Issue: https://github.com/BerriAI/litellm/issues/XXXXX
        """
        error_str = "litellm.APIConnectionError: GeminiException - some error message"

        with pytest.raises(litellm.APIConnectionError) as excinfo:
            extract_and_raise_litellm_exception(
                response=None,
                error_str=error_str,
                model="gemini/gemini-3-pro-preview",
                custom_llm_provider="gemini",
            )

        assert "APIConnectionError" in str(excinfo.value)

    def test_extract_and_raise_bad_request_error_with_response(self):
        """
        Test that BadRequestError can be raised with response parameter.

        BadRequestError does accept the response parameter, so this should work.
        """
        error_str = "litellm.BadRequestError: Invalid request format"

        with pytest.raises(litellm.BadRequestError) as excinfo:
            extract_and_raise_litellm_exception(
                response=None,
                error_str=error_str,
                model="gpt-4",
                custom_llm_provider="openai",
            )

        assert "BadRequestError" in str(excinfo.value)

    def test_extract_and_raise_context_window_exceeded_error(self):
        """
        Test that ContextWindowExceededError can be raised.
        """
        error_str = "litellm.ContextWindowExceededError: Token limit exceeded"

        with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
            extract_and_raise_litellm_exception(
                response=None,
                error_str=error_str,
                model="gpt-4",
                custom_llm_provider="openai",
            )

        assert "ContextWindowExceededError" in str(excinfo.value)

    def test_no_exception_raised_for_non_litellm_error(self):
        """
        Test that no exception is raised for non-litellm error strings.
        """
        error_str = "Some generic error that is not a litellm exception"

        # Should not raise any exception
        result = extract_and_raise_litellm_exception(
            response=None,
            error_str=error_str,
            model="gpt-4",
            custom_llm_provider="openai",
        )

        assert result is None
