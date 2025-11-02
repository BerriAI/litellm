import pytest
import httpx

from litellm.litellm_core_utils.exception_mapping_utils import ExceptionCheckers, exception_type
from litellm import ContentPolicyViolationError
from openai import BadRequestError


# Test cases for is_error_str_context_window_exceeded
# Tuple format: (error_message, expected_result)
context_window_test_cases = [
    # Positive cases (should return True)
    ("An error occurred: The input exceeds the model's maximum context limit of 8192 tokens.", True),
    ("Some text before, this model's maximum context length is 4096 tokens. Some text after.", True),
    ("Validation Error: string too long. expected a string with maximum length 1000.", True),
    ("Your prompt is longer than the model's context length of 2048.", True),
    ("AWS Bedrock Error: The request payload size has exceed context limit.", True),
    ("Input tokens exceed the configured limit of 272000 tokens. Your messages resulted in 509178 tokens. Please reduce the length of the messages.", True),

    # Test case insensitivity
    ("ERROR: THIS MODEL'S MAXIMUM CONTEXT LENGTH IS 1024.", True),

    # Negative cases (should return False)
    ("A generic API error occurred.", False),
    ("Invalid API Key provided.", False),
    ("Rate limit reached for requests.", False),
    ("The context is large, but acceptable.", False),
    ("", False), # Empty string
]

@pytest.mark.parametrize("error_str, expected", context_window_test_cases)
def test_is_error_str_context_window_exceeded(error_str, expected):
    """
    Tests the is_error_str_context_window_exceeded function with various error strings.
    """
    assert ExceptionCheckers.is_error_str_context_window_exceeded(error_str) == expected
    
def test_mapping_azure_content_policy_exception_contains_inner_error():
    """
    Test mapping of Azure content policy violation error that contains 'innererror' and 'content_filter_result'
    to the correct LiteLLM exception type.
    """
    
    dummy_request = httpx.Request("POST", "https://example.com/v1/completions")

    response = httpx.Response(
        status_code=400,
        json={
            "error": {
                "message": "The response was filtered due to the prompt triggering Azure OpenAI's content management policy.",
                "type": None,
                "param": "prompt",
                "code": "content_filter",
                "status": 400,
                "innererror": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_result": {
                        "hate": {"filtered": True, "severity": "high"},
                        "jailbreak": {"filtered": False, "detected": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "medium"},
                    },
                },
            }
        },
        request=dummy_request,
    )

    bad_request_error = BadRequestError(
        message=f"Error code: 400 - {response.json()}",
        response=response,
        body=response.json(),
    )
    
    with pytest.raises(ContentPolicyViolationError) as excinfo:
        exception_type('gpt-4o', Exception(bad_request_error), 'azure')

    mapped_exception = excinfo.value
    
    assert mapped_exception.__class__.__name__ == "ContentPolicyViolationError"
    
    message = str(mapped_exception)
    assert message.startswith("litellm.BadRequestError:")
    assert "litellm.ContentPolicyViolationError" in message
    assert "AzureException" in message
    assert "The response was filtered due to the prompt triggering Azure OpenAI's content management policy." in message
    assert "innererror" in message
    assert "content_filter_result" in message
    assert "ResponsibleAIPolicyViolation" in message