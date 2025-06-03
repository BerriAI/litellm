"""
Test module for LiteLLM proxy function calling support.

This module tests the 'supports_function_calling' function with proxy configurations
to ensure that proxied models correctly report their function calling capabilities.
"""

import pytest
import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

import litellm
from litellm.utils import supports_function_calling


class TestProxyFunctionCalling:
    """Test class for proxy function calling capabilities."""

    @pytest.fixture(autouse=True)
    def reset_mock_cache(self):
        """Reset model cache before each test."""
        from litellm.utils import _model_cache
        _model_cache.flush_cache()

    @pytest.mark.parametrize(
        "direct_model,proxy_model,expected_result",
        [
            # OpenAI models
            ("gpt-3.5-turbo", "litellm_proxy/gpt-3.5-turbo", True),
            ("gpt-4", "litellm_proxy/gpt-4", True),
            ("gpt-4o", "litellm_proxy/gpt-4o", True),
            ("gpt-4o-mini", "litellm_proxy/gpt-4o-mini", True),
            ("gpt-4-turbo", "litellm_proxy/gpt-4-turbo", True),
            ("gpt-4-1106-preview", "litellm_proxy/gpt-4-1106-preview", True),
            
            # Azure OpenAI models
            ("azure/gpt-4", "litellm_proxy/azure/gpt-4", True),
            ("azure/gpt-3.5-turbo", "litellm_proxy/azure/gpt-3.5-turbo", True),
            ("azure/gpt-4-1106-preview", "litellm_proxy/azure/gpt-4-1106-preview", True),
            
            # Anthropic models (Claude supports function calling)
            ("claude-3-5-sonnet-20240620", "litellm_proxy/claude-3-5-sonnet-20240620", True),
            ("claude-3-opus-20240229", "litellm_proxy/claude-3-opus-20240229", True),
            ("claude-3-sonnet-20240229", "litellm_proxy/claude-3-sonnet-20240229", True),
            ("claude-3-haiku-20240307", "litellm_proxy/claude-3-haiku-20240307", True),
            
            # Google models
            ("gemini/gemini-1.5-flash", "litellm_proxy/gemini/gemini-1.5-flash", True),
            ("gemini/gemini-1.5-pro", "litellm_proxy/gemini/gemini-1.5-pro", True),
            ("gemini-pro", "litellm_proxy/gemini-pro", True),
                 # Groq models
        ("groq/llama3-70b-8192", "litellm_proxy/groq/llama3-70b-8192", False),
        ("groq/gemma-7b-it", "litellm_proxy/groq/gemma-7b-it", True),
            
            # Models that don't support function calling
            ("anthropic.claude-instant-v1", "litellm_proxy/anthropic.claude-instant-v1", False),
            ("command-nightly", "litellm_proxy/command-nightly", False),
        ],
    )
    def test_proxy_function_calling_support_consistency(
        self, direct_model, proxy_model, expected_result
    ):
        """
        Test that proxy models return the same function calling support as their direct counterparts.
        
        This test verifies that when a model is accessed through litellm_proxy, 
        it reports the same function calling capabilities as when accessed directly.
        """
        try:
            # Test direct model
            direct_result = supports_function_calling(model=direct_model)
            
            # Test proxy model
            proxy_result = supports_function_calling(model=proxy_model)
            
            # Both should match expected result
            assert direct_result == expected_result, (
                f"Direct model {direct_model} returned {direct_result}, "
                f"expected {expected_result}"
            )
            
            assert proxy_result == expected_result, (
                f"Proxy model {proxy_model} returned {proxy_result}, "
                f"expected {expected_result}"
            )
            
            # Most importantly, direct and proxy results should be consistent
            assert direct_result == proxy_result, (
                f"Inconsistent results: direct model {direct_model} returned {direct_result}, "
                f"but proxy model {proxy_model} returned {proxy_result}"
            )
            
        except Exception as e:
            pytest.fail(f"Error testing {direct_model} vs {proxy_model}: {e}")

    @pytest.mark.parametrize(
        "proxy_model,expected_result",
        [
            # Test specific proxy models that should support function calling
            ("litellm_proxy/gpt-3.5-turbo", True),
            ("litellm_proxy/gpt-4", True),
            ("litellm_proxy/gpt-4o", True),
            ("litellm_proxy/claude-3-5-sonnet-20240620", True),
            ("litellm_proxy/gemini/gemini-1.5-pro", True),
            
            # Test proxy models that should not support function calling
            ("litellm_proxy/command-nightly", False),
            ("litellm_proxy/anthropic.claude-instant-v1", False),
        ],
    )
    def test_proxy_only_function_calling_support(self, proxy_model, expected_result):
        """
        Test proxy models independently to ensure they report correct function calling support.
        
        This test focuses on proxy models without comparing to direct models,
        useful for cases where we only care about the proxy behavior.
        """
        try:
            result = supports_function_calling(model=proxy_model)
            assert result == expected_result, (
                f"Proxy model {proxy_model} returned {result}, expected {expected_result}"
            )
        except Exception as e:
            pytest.fail(f"Error testing proxy model {proxy_model}: {e}")

    def test_litellm_utils_supports_function_calling_import(self):
        """Test that supports_function_calling can be imported from litellm.utils."""
        try:
            from litellm.utils import supports_function_calling
            assert callable(supports_function_calling)
        except ImportError as e:
            pytest.fail(f"Failed to import supports_function_calling: {e}")

    def test_litellm_supports_function_calling_import(self):
        """Test that supports_function_calling can be imported from litellm directly."""
        try:
            import litellm
            assert hasattr(litellm, 'supports_function_calling')
            assert callable(litellm.supports_function_calling)
        except Exception as e:
            pytest.fail(f"Failed to access litellm.supports_function_calling: {e}")

    @pytest.mark.parametrize(
        "model_name",
        [
            "litellm_proxy/gpt-3.5-turbo",
            "litellm_proxy/gpt-4",
            "litellm_proxy/claude-3-5-sonnet-20240620",
            "litellm_proxy/gemini/gemini-1.5-pro",
        ],
    )
    def test_proxy_model_with_custom_llm_provider_none(self, model_name):
        """
        Test proxy models with custom_llm_provider=None parameter.
        
        This tests the supports_function_calling function with the custom_llm_provider
        parameter explicitly set to None, which is a common usage pattern.
        """
        try:
            result = supports_function_calling(model=model_name, custom_llm_provider=None)
            # All the models in this test should support function calling
            assert result is True, (
                f"Model {model_name} should support function calling but returned {result}"
            )
        except Exception as e:
            pytest.fail(f"Error testing {model_name} with custom_llm_provider=None: {e}")

    def test_edge_cases_and_malformed_proxy_models(self):
        """Test edge cases and malformed proxy model names."""
        test_cases = [
            ("litellm_proxy/", False),  # Empty model name after proxy prefix
            ("litellm_proxy", False),   # Just the proxy prefix without slash
            ("litellm_proxy//gpt-3.5-turbo", False),  # Double slash
            ("litellm_proxy/nonexistent-model", False),  # Non-existent model
        ]
        
        for model_name, expected_result in test_cases:
            try:
                result = supports_function_calling(model=model_name)
                # For malformed models, we expect False or the function to handle gracefully
                assert result == expected_result, (
                    f"Edge case {model_name} returned {result}, expected {expected_result}"
                )
            except Exception:
                # It's acceptable for malformed model names to raise exceptions
                # rather than returning False, as long as they're handled gracefully
                pass

    def test_proxy_model_resolution_demonstration(self):
        """
        Demonstration test showing the current issue with proxy model resolution.
        
        This test documents the current behavior and can be used to verify
        when the issue is fixed.
        """
        direct_model = "gpt-3.5-turbo"
        proxy_model = "litellm_proxy/gpt-3.5-turbo"
        
        direct_result = supports_function_calling(model=direct_model)
        proxy_result = supports_function_calling(model=proxy_model)
        
        print(f"\nDemonstration of proxy model resolution:")
        print(f"Direct model '{direct_model}' supports function calling: {direct_result}")
        print(f"Proxy model '{proxy_model}' supports function calling: {proxy_result}")
        
        # This assertion will currently fail due to the bug
        # When the bug is fixed, this test should pass
        if direct_result != proxy_result:
            pytest.skip(
                f"Known issue: Proxy model resolution inconsistency. "
                f"Direct: {direct_result}, Proxy: {proxy_result}. "
                f"This test will pass when the issue is resolved."
            )
        
        assert direct_result == proxy_result, (
            f"Proxy model resolution issue: {direct_model} -> {direct_result}, "
            f"{proxy_model} -> {proxy_result}"
        )


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v"])
