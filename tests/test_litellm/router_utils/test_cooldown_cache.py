"""
Unit tests for CooldownCache exception masking functionality
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.router_utils.cooldown_cache import CooldownCache, CooldownCacheValue


class TestCooldownCacheExceptionMasking:
    """Test suite for CooldownCache exception masking functionality"""

    @pytest.fixture
    def cooldown_cache(self):
        """Create a CooldownCache instance for testing"""
        mock_dual_cache = MagicMock(spec=DualCache)
        return CooldownCache(cache=mock_dual_cache, default_cooldown_time=60.0)

    def test_exception_masker_initialization(self, cooldown_cache):
        """Test that the exception masker is properly initialized"""
        assert isinstance(cooldown_cache.exception_masker, SensitiveDataMasker)
        assert cooldown_cache.exception_masker.visible_prefix == 50
        assert cooldown_cache.exception_masker.visible_suffix == 0
        assert cooldown_cache.exception_masker.mask_char == "*"

    def test_short_exception_string_not_masked(self, cooldown_cache):
        """Test that short exception strings are not masked"""
        short_exception = "Short error"
        model_id = "test-model"
        exception_status = 500
        cooldown_time = 30.0

        cooldown_key, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(short_exception),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        # Short exception should not be masked
        assert cooldown_data["exception_received"] == short_exception
        assert cooldown_key == f"deployment:{model_id}:cooldown"

    def test_long_exception_string_masked(self, cooldown_cache):
        """Test that long exception strings are properly masked"""
        # Create a long exception string that simulates prompt leakage
        long_exception = (
            "litellm.proxy.proxy_server._handle_llm_api_exception(): Exception occurred - "
            "No deployments available for selected model, Try again in 5 seconds. "
            "Passed model=anthropic_claude_sonnet_4_v1_0. pre-call-checks=False, "
            "cooldown_list=[('deepseek_r1-eastus', {'exception_received': "
            "'litellm.RateLimitError: RateLimitError: Azure_aiException - "
            '{"error":{"code":"Invalid input","status":422,"message":"invalid input error",'
            '"details":[{"type":"model_attributes_type","loc":["body"],'
            '"msg":"Tell me a story about a dragon and a princess in a magical kingdom '
            "where the dragon is actually protecting the princess from an evil wizard "
            'who wants to steal her magical powers and use them to conquer the world"}]}'
        )

        model_id = "test-model"
        exception_status = 429
        cooldown_time = 60.0

        cooldown_key, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(long_exception),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        masked_exception = cooldown_data["exception_received"]

        # Should start with first 50 characters
        assert masked_exception.startswith(long_exception[:50])

        # Should contain masking characters
        assert "*" in masked_exception

        # Should be same length (prefix + asterisks)
        assert len(masked_exception) == len(long_exception)

        # Should not contain the sensitive prompt content
        assert "Tell me a story about a dragon" not in masked_exception
        assert "magical kingdom" not in masked_exception

        # Should preserve the error type information at the beginning (first 50 chars)
        assert masked_exception.startswith(
            "litellm.proxy.proxy_server._handle_llm_api_excepti"
        )

    def test_exception_with_api_keys_masked(self, cooldown_cache):
        """Test that API keys in exceptions are properly masked"""
        exception_with_key = (
            "Authentication failed with api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz "
            "and token=bearer_token_123456789 for model gpt-4"
        )

        model_id = "test-model"
        exception_status = 401
        cooldown_time = 30.0

        cooldown_key, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(exception_with_key),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        masked_exception = cooldown_data["exception_received"]

        # Should mask the sensitive content while preserving structure
        assert masked_exception.startswith(
            "Authentication failed with api_key=sk-12345678"
        )
        assert "*" in masked_exception
        assert len(masked_exception) == len(exception_with_key)

    def test_cooldown_data_structure(self, cooldown_cache):
        """Test that the cooldown data structure is correctly formed"""
        exception_msg = "Test exception for structure validation"
        model_id = "test-model"
        exception_status = 500
        cooldown_time = 45.0

        cooldown_key, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(exception_msg),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        # Verify cooldown data structure
        assert isinstance(cooldown_data, dict)
        assert "exception_received" in cooldown_data
        assert "status_code" in cooldown_data
        assert "timestamp" in cooldown_data
        assert "cooldown_time" in cooldown_data

        # Verify data types
        assert isinstance(cooldown_data["exception_received"], str)
        assert isinstance(cooldown_data["status_code"], str)
        assert isinstance(cooldown_data["timestamp"], float)
        assert isinstance(cooldown_data["cooldown_time"], float)

        # Verify values
        assert cooldown_data["status_code"] == str(exception_status)
        assert cooldown_data["cooldown_time"] == cooldown_time
        assert cooldown_data["exception_received"] == exception_msg

    def test_exception_object_conversion(self, cooldown_cache):
        """Test that different exception types are properly converted to strings"""
        # Test with different exception types
        exceptions = [
            ValueError("Invalid value provided"),
            KeyError("Missing required key"),
            RuntimeError("Runtime error occurred"),
            Exception("Generic exception"),
        ]

        for exc in exceptions:
            model_id = f"test-model-{exc.__class__.__name__}"

            cooldown_key, cooldown_data = cooldown_cache._common_add_cooldown_logic(
                model_id=model_id,
                original_exception=exc,
                exception_status=500,
                cooldown_time=30.0,
            )

            # Should successfully convert exception to string
            assert isinstance(cooldown_data["exception_received"], str)
            assert (
                str(exc) == cooldown_data["exception_received"]
            )  # Short exceptions not masked

    def test_masking_preserves_error_debugging_info(self, cooldown_cache):
        """Test that masking preserves essential debugging information"""
        debugging_exception = (
            "RateLimitError: Rate limit exceeded for model gpt-4. "
            "Current usage: 1000 tokens/minute. Limit: 500 tokens/minute. "
            "Request details: model=gpt-4, user_id=user123, "
            "prompt='Write a comprehensive analysis of the economic implications "
            "of artificial intelligence adoption in the healthcare sector, including "
            "potential cost savings, job displacement, and regulatory challenges'"
        )

        model_id = "gpt-4-deployment"
        exception_status = 429
        cooldown_time = 120.0

        cooldown_key, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(debugging_exception),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        masked_exception = cooldown_data["exception_received"]

        # Should preserve error type and initial debugging info (first 50 chars)
        assert masked_exception.startswith(
            "RateLimitError: Rate limit exceeded for model gpt-"
        )

        # Should mask the prompt content
        assert "Write a comprehensive analysis" not in masked_exception
        assert "healthcare sector" not in masked_exception

        # Should contain masking indicator
        assert "*" in masked_exception

    def test_error_handling_in_common_add_cooldown_logic(self, cooldown_cache):
        """Test error handling in the _common_add_cooldown_logic method"""
        # This test ensures that edge cases are properly handled
        model_id = "test-model"

        # Test with None exception (edge case) - should be handled gracefully
        cooldown_key, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=None,
            exception_status=500,
            cooldown_time=30.0,
        )

        # Should handle None by converting to string
        assert cooldown_data["exception_received"] == "None"
        assert cooldown_key == f"deployment:{model_id}:cooldown"

    def test_custom_masker_settings(self):
        """Test that custom masker settings work correctly"""
        mock_dual_cache = MagicMock(spec=DualCache)

        # Create cooldown cache and verify default settings
        cache = CooldownCache(cache=mock_dual_cache, default_cooldown_time=60.0)

        # Test that we can access and verify the masker configuration
        assert cache.exception_masker.visible_prefix == 50
        assert cache.exception_masker.visible_suffix == 0
        assert cache.exception_masker.mask_char == "*"

        # Test masking behavior with these settings
        long_string = "A" * 100  # 100 character string
        masked = cache.exception_masker._mask_value(long_string)

        # Should show first 50 characters, then all asterisks
        expected = "A" * 50 + "*" * 50
        assert masked == expected
