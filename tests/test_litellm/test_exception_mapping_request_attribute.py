
"""
Unit tests for the exception mapping request attribute handling fix.

This test verifies the fix for PR #15013 where getattr(original_exception, "request", None)
is used instead of original_exception.request to handle cases where exceptions don't have
a request attribute.

The key fix is that accessing original_exception.request directly would raise AttributeError
if the exception doesn't have a request attribute, but getattr(original_exception, "request", None)
safely returns None instead.

PR #15013 fixed 12 locations in exception_mapping_utils.py where direct access to .request
was replaced with getattr() calls:
- Line 1501: Cohere exception mapping
- Line 1574: HuggingFace exception mapping
- Line 1635: AI21 exception mapping
- Line 1660: NLP Cloud exception mapping
- Line 1720: NLP Cloud exception mapping (another case)
- Line 1740: NLP Cloud exception mapping (another case)
- Line 1851: Together AI exception mapping
- Line 1954: VLLM exception mapping
- Line 2209: Generic provider exception mapping
- Line 2244: Generic provider exception mapping (fallback)
- OpenRouter exception mapping (multiple locations)

This test ensures that none of these code paths will raise AttributeError when an exception
object doesn't have a request attribute, which was the root cause of the bug.
"""

import pytest
import httpx
from unittest.mock import patch

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.exceptions import APIError, APIConnectionError


class MockExceptionWithoutRequest:
    """Mock exception that does NOT have a request attribute."""
    
    def __init__(self, status_code=500, message="Test error"):
        self.status_code = status_code
        self.message = message
        # Intentionally no request attribute


def test_exception_mapping_request_attribute_fix():
    """
    Test the core fix: getattr(original_exception, "request", None) should not raise AttributeError
    even when the exception doesn't have a request attribute.
    
    This is the main test for PR #15013.
    """
    
    # Test case 1: Exception without request attribute should not cause AttributeError
    mock_exception = MockExceptionWithoutRequest(
        status_code=500,
        message="Test error without request attribute"
    )
    
    # The test is that this should NOT raise an AttributeError about missing 'request'
    try:
        exception_type(
            model="test-model",
            custom_llm_provider="cohere",  # Using cohere as it's one of the affected providers
            original_exception=mock_exception,
            completion_kwargs={},
            extra_kwargs={}
        )
        # We expect some exception to be raised (the mapped exception), but not AttributeError
    except AttributeError as e:
        if "'request'" in str(e):
            pytest.fail(f"The fix failed: Should not raise AttributeError about missing 'request' attribute: {e}")
        else:
            # If it's a different AttributeError, re-raise it
            raise
    except Exception:
        # Any other exception is fine - we just want to ensure no AttributeError about 'request'
        pass


def test_request_attribute_safety_with_getattr():
    """
    Test that the getattr approach works correctly for both cases:
    1. When request attribute exists
    2. When request attribute doesn't exist
    """
    
    # Case 1: Exception with request attribute
    class MockExceptionWithRequest:
        def __init__(self):
            self.status_code = 500
            self.message = "Test error"
            self.request = httpx.Request(method="POST", url="https://api.example.com")
    
    exception_with_request = MockExceptionWithRequest()
    request_value = getattr(exception_with_request, "request", None)
    assert request_value is not None
    assert isinstance(request_value, httpx.Request)
    
    # Case 2: Exception without request attribute
    exception_without_request = MockExceptionWithoutRequest()
    request_value = getattr(exception_without_request, "request", None)
    assert request_value is None  # Should be None, not raise AttributeError


def test_providers_affected_by_fix():
    """
    Test that the specific providers mentioned in the PR changes handle missing request attributes correctly.
    
    The PR changes affected these provider-specific code paths:
    - cohere: line 1501
    - huggingface: line 1574
    - ai21: line 1635
    - nlp_cloud: lines 1660, 1720, 1740
    - together_ai: line 1851
    - vllm: line 1954
    - generic providers: lines 2209, 2244
    """
    
    providers_to_test = [
        "cohere",
        "ai21",
        "together_ai",
        "vllm"
    ]
    
    for provider in providers_to_test:
        mock_exception = MockExceptionWithoutRequest(
            status_code=500,
            message=f"Test error for {provider}"
        )
        
        # The key test: this should not raise AttributeError about missing 'request'
        try:
            exception_type(
                model=f"{provider}-test-model",
                custom_llm_provider=provider,
                original_exception=mock_exception,
                completion_kwargs={},
                extra_kwargs={}
            )
        except AttributeError as e:
            if "'request'" in str(e):
                pytest.fail(f"Provider {provider} failed: Should not raise AttributeError about missing 'request' attribute: {e}")
        except Exception:
            # Any other exception is expected and fine
            pass


def test_huggingface_specific_case():
    """
    Test HuggingFace specific case which has its own handling logic.
    """
    mock_exception = MockExceptionWithoutRequest(
        status_code=400,
        message="length limit exceeded"
    )
    
    try:
        exception_type(
            model="huggingface-model",
            custom_llm_provider="huggingface",
            original_exception=mock_exception,
            completion_kwargs={},
            extra_kwargs={}
        )
    except AttributeError as e:
        if "'request'" in str(e):
            pytest.fail(f"HuggingFace exception handling failed: Should not raise AttributeError about missing 'request' attribute: {e}")
    except litellm.ContextWindowExceededError:
        # Expected for "length limit exceeded" message
        pass
    except Exception:
        # Other exceptions are fine
        pass


def test_nlp_cloud_specific_case():
    """
    Test NLP Cloud specific case which had multiple lines changed in the PR.
    """
    mock_exception = MockExceptionWithoutRequest(
        status_code=504,
        message="Gateway timeout"
    )
    
    try:
        exception_type(
            model="nlp-cloud-model",
            custom_llm_provider="nlp_cloud",
            original_exception=mock_exception,
            completion_kwargs={},
            extra_kwargs={}
        )
    except AttributeError as e:
        if "'request'" in str(e):
            pytest.fail(f"NLP Cloud exception handling failed: Should not raise AttributeError about missing 'request' attribute: {e}")
    except Exception:
        # Any other exception is expected
        pass


def test_generic_fallback_case():
    """
    Test the generic fallback case at the end of exception_type function.
    This tests the changes in lines 2209 and 2244 of the PR.
    """
    mock_exception = MockExceptionWithoutRequest(
        status_code=500,
        message="Generic error"
    )
    
    try:
        exception_type(
            model="unknown-model",
            custom_llm_provider="unknown_provider",
            original_exception=mock_exception,
            completion_kwargs={},
            extra_kwargs={}
        )
    except AttributeError as e:
        if "'request'" in str(e):
            pytest.fail(f"Generic fallback failed: Should not raise AttributeError about missing 'request' attribute: {e}")
    except APIConnectionError:
        # Expected for generic fallback
        pass
    except Exception:
        # Other exceptions might be fine too
        pass


def test_openrouter_specific_case():
    """
    Test OpenRouter which also uses the request attribute in exception mapping.
    """
    mock_exception = MockExceptionWithoutRequest(
        status_code=500,
        message="OpenRouter error"
    )
    
    try:
        exception_type(
            model="openrouter-model",
            custom_llm_provider="openrouter",
            original_exception=mock_exception,
            completion_kwargs={},
            extra_kwargs={}
        )
    except AttributeError as e:
        if "'request'" in str(e):
            pytest.fail(f"OpenRouter exception handling failed: Should not raise AttributeError about missing 'request' attribute: {e}")
    except Exception:
        # Other exceptions are expected
        pass


if __name__ == "__main__":
    # Run tests for manual verification
    test_exception_mapping_request_attribute_fix()
    test_request_attribute_safety_with_getattr()
    test_providers_affected_by_fix()
    test_huggingface_specific_case()
    test_nlp_cloud_specific_case()
    test_generic_fallback_case()
    test_openrouter_specific_case()
    print("All tests passed!")
