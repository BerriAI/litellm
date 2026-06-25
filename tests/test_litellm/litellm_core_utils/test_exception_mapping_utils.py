import os
import sys

import httpx
import pytest

import litellm

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.exception_mapping_utils import (
    ExceptionCheckers,
    _get_body_error_code,
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


class TestGetBodyErrorCode:
    """Unit tests for _get_body_error_code helper."""

    def test_parses_int_code(self):
        body = (
            '{"error":{"message":"high demand","type":"upstream_error",'
            '"param":"","code":429}}'
        )
        assert _get_body_error_code(body) == 429

    def test_parses_string_code(self):
        # some gateways serialize code as a string
        body = '{"error":{"message":"x","code":"503"}}'
        assert _get_body_error_code(body) == 503

    def test_returns_none_on_non_json(self):
        assert _get_body_error_code("not json") is None

    def test_returns_none_when_no_error_key(self):
        assert _get_body_error_code('{"ok":true}') is None

    def test_returns_none_when_no_code_key(self):
        assert _get_body_error_code('{"error":{"message":"x"}}') is None


# Test cases for Gemini upstream-error body-code mapping.
#
# Body code 429 wrapped in a 5xx HTTP envelope (e.g. new-api gateways)
# must map to RateLimitError so Router retries kick in. A 4xx HTTP
# envelope with body code:429 must NOT — it falls through to whatever
# the HTTP status code maps to (BadRequestError, AuthenticationError,
# etc.), matching upstream's existing semantics.
gemini_body_code_429_test_cases = [
    # (status_code, error_body, expected_exception_type, description)
    (
        500,
        '{"error":{"message":" This model is currently experiencing high demand.'
        " Spikes in demand are usually temporary. Please try again later."
        ' (request id: x)","type":"upstream_error","param":"","code":429}}',
        litellm.RateLimitError,
        "HTTP 500 envelope with body code:429 -> RateLimitError",
    ),
    (
        503,
        '{"error":{"message":"upstream unavailable","type":"upstream_error",'
        '"param":"","code":429}}',
        litellm.RateLimitError,
        "HTTP 503 envelope with body code:429 -> RateLimitError",
    ),
    (
        502,
        '{"error":{"message":"bad gateway","code":429}}',
        litellm.RateLimitError,
        "HTTP 502 envelope with body code:429 -> RateLimitError",
    ),
    (
        500,
        '{"error":{"message":"server boom","code":500}}',
        litellm.InternalServerError,
        "HTTP 500 with body code:500 stays InternalServerError",
    ),
    (
        500,
        "plain text 500 error",
        litellm.InternalServerError,
        "HTTP 500 with non-JSON body falls through to status_code mapping",
    ),
    (
        400,
        '{"error":{"message":"malformed","code":429}}',
        litellm.BadRequestError,
        "HTTP 400 with body code:429 must NOT be promoted to RateLimitError",
    ),
    (
        401,
        '{"error":{"message":"bad key","code":429}}',
        litellm.AuthenticationError,
        "HTTP 401 with body code:429 must NOT be promoted to RateLimitError",
    ),
]


@pytest.mark.parametrize(
    "status_code, error_body, expected_exception, description",
    gemini_body_code_429_test_cases,
)
def test_gemini_upstream_error_body_code_429_maps_to_rate_limit(
    status_code, error_body, expected_exception, description
):
    """
    Body code 429 inside a 5xx envelope -> RateLimitError so Router
    retries kick in. Body code 429 inside a 4xx envelope must fall
    through to the HTTP-status-code branch (P1 from greptile review).
    """
    model = "gemini/gemini-2.5-flash"
    custom_llm_provider = "gemini"

    # Build an exception that looks like what _handle_error produces:
    # a BaseLLMException-style object with .status_code and .message
    class _FakeGeminiError(Exception):
        def __init__(self, status_code, message):
            self.status_code = status_code
            self.message = message
            super().__init__(message)

    original_exception = _FakeGeminiError(status_code=status_code, message=error_body)

    with pytest.raises(expected_exception) as excinfo:
        exception_type(
            model=model,
            original_exception=original_exception,
            custom_llm_provider=custom_llm_provider,
        )
    assert isinstance(excinfo.value, expected_exception), description


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


class ModelError(Exception):
    """Mimics replicate's SDK exception, whose mapping keys on the class name."""


class CohereConnectionError(Exception):
    """Mimics cohere's SDK exception, whose mapping keys on the class name."""


def test_replicate_model_error_maps_to_bad_request():
    """The replicate branch keys on ``type(original_exception).__name__ ==
    "ModelError"`` rather than on the error string. The dispatch now lives in a
    per-provider helper, so this class-name value has to be threaded into the
    helper; if it is not, the bare name ``exception_type`` resolves to the
    module-level function and the comparison is always False, silently mismapping
    to APIConnectionError."""
    original_exception = ModelError("the deployed model failed to return a prediction")

    with pytest.raises(litellm.BadRequestError) as excinfo:
        exception_type(
            model="replicate/meta/llama-2-70b-chat",
            original_exception=original_exception,
            custom_llm_provider="replicate",
        )

    assert excinfo.value.llm_provider == "replicate"


def test_cohere_connection_error_maps_to_rate_limit():
    """The cohere branch keys on ``"CohereConnectionError" in
    type(original_exception).__name__``. With the dispatch extracted into a helper
    the class-name value must be passed in; otherwise ``in`` runs against the
    module-level ``exception_type`` function object and raises TypeError, which the
    outer handler swallows into a generic APIConnectionError."""
    original_exception = CohereConnectionError("connection reset by peer")
    original_exception.message = "connection reset by peer"

    with pytest.raises(litellm.RateLimitError) as excinfo:
        exception_type(
            model="command-r",
            original_exception=original_exception,
            custom_llm_provider="cohere",
        )

    assert excinfo.value.llm_provider == "cohere"


class ReplicateError(Exception):
    """Mimics a replicate HTTP error carrying a status_code and response."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.replicate.com/v1/predictions"),
        )


def test_replicate_422_maps_to_unprocessable_entity():
    """The replicate status-code ladder carried two identical ``status_code == 422``
    branches; the second was unreachable dead code. After dropping the duplicate the
    surviving branch must still map 422 to UnprocessableEntityError."""
    original_exception = ReplicateError("validation failed for the input", 422)

    with pytest.raises(litellm.UnprocessableEntityError) as excinfo:
        exception_type(
            model="replicate/meta/llama-2-70b-chat",
            original_exception=original_exception,
            custom_llm_provider="replicate",
        )

    assert excinfo.value.llm_provider == "replicate"
