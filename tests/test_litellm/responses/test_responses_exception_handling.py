import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)

import litellm
from litellm import InternalServerError, BadRequestError, AuthenticationError, APIConnectionError
from litellm.litellm_core_utils.exception_mapping_utils import exception_type


class TestResponsesAPIExceptionMappingFix:
    """
    Test suite for Responses API exception handling fixes.
    
    Addresses GitHub Issue: "[Bug]: Responses API bridge swallows all errors 
    into generic 'APIConnectionError: OpenAIException - argument of type NoneType 
    is not iterable'"
    
    These tests verify the core fix:
    1. Internal Python exceptions (TypeError, ValueError, etc.) are correctly 
       mapped to InternalServerError (not misleading APIConnectionError)
    2. Already-mapped LiteLLM exceptions are detected correctly
    3. Exception details include proper context
    """

    def test_typeerror_mapped_to_internal_server_error(self):
        """
        Test that TypeError (e.g., "NoneType is not iterable") is correctly 
        mapped to InternalServerError instead of misleading APIConnectionError.
        
        This simulates the original bug where `in` operator on None caused TypeError.
        """
        model = "gpt-4"
        custom_llm_provider = "openai"
        original_exception = TypeError("argument of type 'NoneType' is not iterable")
        
        with pytest.raises(InternalServerError) as excinfo:
            exception_type(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=original_exception,
            )
        
        # Verify it's InternalServerError, not APIConnectionError
        assert isinstance(excinfo.value, InternalServerError)
        
        # Verify the error message includes helpful information
        error_msg = str(excinfo.value)
        assert "TypeError" in error_msg
        assert "Internal error in LiteLLM" in error_msg
        assert "argument of type 'NoneType' is not iterable" in error_msg
        
        # Verify model and provider context is preserved
        assert excinfo.value.model == model
        assert excinfo.value.llm_provider == custom_llm_provider

litellm/litellm_core_utils/exception_mapping_utils.py    def test_valueerror_mapped_to_apiconnectionerror(self):
        """
        Test that ValueError is NOT caught as InternalServerError.
        
        ValueError is commonly used for user input validation (600+ instances)
        and should fall through to APIConnectionError or be caught by provider-specific
        error string matching to map to BadRequestError.
        
        Catching ValueError globally would cause a regression by misclassifying
        legitimate user input validation as 500 errors instead of 4xx errors.
        """
        model = "gpt-4"
        custom_llm_provider = "openai"
        original_exception = ValueError("Invalid transformation parameter")
        
        # ValueError should NOT be caught by our isinstance check
        # It will fall through to APIConnectionError
        with pytest.raises(APIConnectionError) as excinfo:
            exception_type(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=original_exception,
            )
        
        assert isinstance(excinfo.value, APIConnectionError)
        assert excinfo.value.model == model
        assert excinfo.value.llm_provider == custom_llm_provider

    def test_keyerror_mapped_to_internal_server_error(self):
        """
        Test that KeyError from missing required keys is correctly 
        mapped to InternalServerError.
        
        Tests with Anthropic provider to verify the fix applies to ALL providers,
        not just OpenAI (the isinstance check is in the common section).
        """
        model = "anthropic/claude-3-5-sonnet"
        custom_llm_provider = "anthropic"
        original_exception = KeyError("required_field_for_transformation")
        
        with pytest.raises(InternalServerError) as excinfo:
            exception_type(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=original_exception,
            )
        
        assert isinstance(excinfo.value, InternalServerError)
        error_msg = str(excinfo.value)
        assert "KeyError" in error_msg
        assert "Internal error in LiteLLM" in error_msg
        assert excinfo.value.model == model
        assert excinfo.value.llm_provider == custom_llm_provider

    def test_attributeerror_mapped_to_apiconnectionerror(self):
        """
        Test that AttributeError is NOT caught as InternalServerError.
        
        Similar to ValueError, AttributeError may be used intentionally in provider code
        and should not be globally classified as a 500 Internal Server Error.
        """
        model = "gpt-4"
        custom_llm_provider = "openai"
        original_exception = AttributeError("'NoneType' object has no attribute 'get'")
        
        # AttributeError should NOT be caught by our isinstance check
        with pytest.raises(APIConnectionError) as excinfo:
            exception_type(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=original_exception,
            )
        
        assert isinstance(excinfo.value, APIConnectionError)

    def test_http_error_still_mapped_to_apiconnectionerror(self):
        """
        Test that non-internal errors without status codes are still mapped 
        to APIConnectionError (expected behavior for actual API errors).
        """
        model = "gpt-4"
        custom_llm_provider = "openai"
        original_exception = Exception("Connection refused")
        
        with pytest.raises(APIConnectionError) as excinfo:
            exception_type(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=original_exception,
            )
        
        assert isinstance(excinfo.value, APIConnectionError)
        assert "Connection refused" in str(excinfo.value)

    def test_already_mapped_exceptions_detected(self):
        """
        Test that already-mapped LiteLLM exceptions are correctly identified.
        
        Before the fix, the responses API would re-map already-mapped exceptions,
        causing loss of context. The fix adds a check using LITELLM_EXCEPTION_TYPES.
        """
        # Create sample exceptions
        bad_request = BadRequestError(message="test", model="gpt-4", llm_provider="openai")
        auth_error = AuthenticationError(message="test", model="gpt-4", llm_provider="openai")
        internal_error = InternalServerError(message="test", model="gpt-4", llm_provider="openai")
        api_error = APIConnectionError(message="test", model="gpt-4", llm_provider="openai")
        
        # Test that various LiteLLM exceptions are in LITELLM_EXCEPTION_TYPES
        assert any(isinstance(bad_request, exc_type) for exc_type in litellm.LITELLM_EXCEPTION_TYPES)
        assert any(isinstance(auth_error, exc_type) for exc_type in litellm.LITELLM_EXCEPTION_TYPES)
        assert any(isinstance(internal_error, exc_type) for exc_type in litellm.LITELLM_EXCEPTION_TYPES)
        assert any(isinstance(api_error, exc_type) for exc_type in litellm.LITELLM_EXCEPTION_TYPES)

    def test_typeerror_includes_error_details(self):
        """
        Test that TypeError includes helpful error details in InternalServerError.
        
        This helps developers debug where exactly the internal error occurred.
        """
        model = "gpt-4"
        custom_llm_provider = "openai"
        
        # Create a real exception with a stack trace
        try:
            test_var = None
            if "test" in test_var:  # This will raise TypeError
                pass
        except TypeError as e:
            original_exception = e
        
        with pytest.raises(InternalServerError) as excinfo:
            exception_type(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=original_exception,
            )
        
        error_msg = str(excinfo.value)
        
        # Verify error type and details are included
        assert "TypeError" in error_msg
        assert "Internal error in LiteLLM" in error_msg
        assert "NoneType" in error_msg

    def test_internal_error_context_preservation(self):
        """
        Test that model and provider information is preserved in mapped exceptions.
        
        This ensures developers can identify which model/provider caused the issue.
        """
        model = "gpt-4"
        custom_llm_provider = "openai"
        original_exception = TypeError("Test internal error for context preservation")
        
        with pytest.raises(InternalServerError) as excinfo:
            exception_type(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=original_exception,
            )
        
        # Verify model and provider are preserved
        assert excinfo.value.model == model
        assert excinfo.value.llm_provider == custom_llm_provider

    def test_internal_error_includes_error_type_in_message(self):
        """
        Test that the original error type name is included in the error message
        for TypeError and KeyError (the only types we catch as InternalServerError).
        """
        test_cases = [
            (TypeError("test"), "TypeError"),
            (KeyError("test"), "KeyError"),
        ]
        
        for original_exception, expected_type_name in test_cases:
            with pytest.raises(InternalServerError) as excinfo:
                exception_type(
                    model="gpt-4",
                    custom_llm_provider="openai",
                    original_exception=original_exception,
                )
            
            error_msg = str(excinfo.value)
            assert expected_type_name in error_msg, f"Expected {expected_type_name} in error message: {error_msg}"

    def test_internal_errors_work_for_all_providers(self):
        """
        Test that internal Python exceptions are mapped to InternalServerError
        for ALL providers, not just OpenAI.
        
        This verifies the isinstance check is in the common section of exception_type(),
        not inside provider-specific branches.
        """
        test_providers = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-5-sonnet"),
            ("azure", "gpt-4"),
            ("bedrock", "anthropic.claude-v2"),
            ("vertex_ai", "gemini-pro"),
        ]
        
        original_exception = TypeError("test internal error")
        
        for provider, model in test_providers:
            with pytest.raises(InternalServerError) as excinfo:
                exception_type(
                    model=model,
                    custom_llm_provider=provider,
                    original_exception=original_exception,
                )
            
            # Verify it's InternalServerError for all providers
            assert isinstance(excinfo.value, InternalServerError), f"Failed for provider: {provider}"
            assert "TypeError" in str(excinfo.value), f"Missing TypeError in message for provider: {provider}"
            assert excinfo.value.llm_provider == provider, f"Provider context lost for: {provider}"
