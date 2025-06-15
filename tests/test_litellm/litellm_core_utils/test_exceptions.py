import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest

def test_fireworks_ai_exception_mapping():
    """
    Comprehensive test for Fireworks AI exception mapping, including:
    1. Standard 429 rate limit errors
    2. Text-based rate limit detection (the main issue fixed)
    3. Generic 400 errors that should NOT be rate limits
    4. ExceptionCheckers utility function
    
    Related to: https://github.com/BerriAI/litellm/pull/11455
    Based on Fireworks AI documentation: https://docs.fireworks.ai/tools-sdks/python-client/api-reference
    """
    import litellm
    from litellm.llms.fireworks_ai.common_utils import FireworksAIException
    from litellm.litellm_core_utils.exception_mapping_utils import ExceptionCheckers
    
    # Test scenarios covering all important cases
    test_scenarios = [
        {
            "name": "Standard 429 rate limit with proper status code",
            "status_code": 429,
            "message": "Rate limit exceeded. Please try again in 60 seconds.",
            "expected_exception": litellm.RateLimitError,
        },
        {
            "name": "Status 400 with rate limit text (the main issue fixed)",
            "status_code": 400,
            "message": '{"error":{"object":"error","type":"invalid_request_error","message":"rate limit exceeded, please try again later"}}',
            "expected_exception": litellm.RateLimitError,
        },
        {
            "name": "Status 400 with generic invalid request (should NOT be rate limit)",
            "status_code": 400,
            "message": '{"error":{"type":"invalid_request_error","message":"Invalid parameter value"}}',
            "expected_exception": litellm.BadRequestError,
        },
    ]
    
    # Test each scenario
    for scenario in test_scenarios:
        mock_exception = FireworksAIException(
            status_code=scenario["status_code"],
            message=scenario["message"],
            headers={}
        )
        
        try:
            response = litellm.completion(
                model="fireworks_ai/llama-v3p1-70b-instruct",
                messages=[{"role": "user", "content": "Hello"}],
                mock_response=mock_exception,
            )
            pytest.fail(f"Expected {scenario['expected_exception'].__name__} to be raised")
        except scenario["expected_exception"] as e:
            if scenario["expected_exception"] == litellm.RateLimitError:
                assert "rate limit" in str(e).lower() or "429" in str(e)
        except Exception as e:
            pytest.fail(f"Expected {scenario['expected_exception'].__name__} but got {type(e).__name__}: {e}")
    
    # Test ExceptionCheckers.is_error_str_rate_limit() method directly
    
    # Test cases that should return True (rate limit detected)
    rate_limit_strings = [
        "429 rate limit exceeded",
        "Rate limit exceeded, please try again later", 
        "RATE LIMIT ERROR",
        "Error 429: rate limit",
        '{"error":{"type":"invalid_request_error","message":"rate limit exceeded, please try again later"}}',
        "HTTP 429 Too Many Requests",
    ]
    
    for error_str in rate_limit_strings:
        assert ExceptionCheckers.is_error_str_rate_limit(error_str), f"Should detect rate limit in: {error_str}"
    
    # Test cases that should return False (not rate limit)
    non_rate_limit_strings = [
        "400 Bad Request",
        "Authentication failed", 
        "Invalid model specified",
        "Context window exceeded",
        "Internal server error",
        "",
        "Some other error message",
    ]
    
    for error_str in non_rate_limit_strings:
        assert not ExceptionCheckers.is_error_str_rate_limit(error_str), f"Should NOT detect rate limit in: {error_str}"
    
    # Test edge cases
    assert not ExceptionCheckers.is_error_str_rate_limit(None)  # type: ignore
    assert not ExceptionCheckers.is_error_str_rate_limit(42)  # type: ignore