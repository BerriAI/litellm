"""
Unit tests for CooldownCache functionality.
"""

from unittest.mock import MagicMock

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.router_utils.cooldown_cache import CooldownCache


class TestCooldownCacheExceptionMasking:
    """Test suite for CooldownCache exception masking functionality."""

    @pytest.fixture
    def cooldown_cache(self):
        """Create a CooldownCache instance for testing."""
        mock_dual_cache = MagicMock(spec=DualCache)
        return CooldownCache(cache=mock_dual_cache, default_cooldown_time=60.0)

    def test_exception_masker_initialization(self, cooldown_cache):
        """Test that the exception masker is properly initialized."""
        assert isinstance(cooldown_cache.exception_masker, SensitiveDataMasker)
        assert cooldown_cache.exception_masker.visible_prefix == 50
        assert cooldown_cache.exception_masker.visible_suffix == 0
        assert cooldown_cache.exception_masker.mask_char == "*"

    def test_short_exception_string_not_masked(self, cooldown_cache):
        """Test that short exception strings are not masked."""
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

        assert cooldown_data["exception_received"] == short_exception
        assert cooldown_key == f"deployment:{model_id}:cooldown"

    def test_long_exception_string_masked(self, cooldown_cache):
        """Test that long exception strings are properly masked."""
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

        _, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(long_exception),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        masked_exception = cooldown_data["exception_received"]

        assert masked_exception.startswith(long_exception[:50])
        assert "*" in masked_exception
        assert len(masked_exception) == len(long_exception)
        assert "Tell me a story about a dragon" not in masked_exception
        assert "magical kingdom" not in masked_exception
        assert masked_exception.startswith(
            "litellm.proxy.proxy_server._handle_llm_api_excepti"
        )

    def test_exception_with_api_keys_masked(self, cooldown_cache):
        """Test that API keys in exceptions are properly masked."""
        exception_with_key = (
            "Authentication failed with api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz "
            "and token=bearer_token_123456789 for model gpt-4"
        )

        model_id = "test-model"
        exception_status = 401
        cooldown_time = 30.0

        _, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(exception_with_key),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        masked_exception = cooldown_data["exception_received"]

        assert masked_exception.startswith(
            "Authentication failed with api_key=sk-12345678"
        )
        assert "*" in masked_exception
        assert len(masked_exception) == len(exception_with_key)

    def test_cooldown_data_structure(self, cooldown_cache):
        """Test that the cooldown data structure is correctly formed."""
        exception_msg = "Test exception for structure validation"
        model_id = "test-model"
        exception_status = 500
        cooldown_time = 45.0

        _, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(exception_msg),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        assert isinstance(cooldown_data, dict)
        assert "exception_received" in cooldown_data
        assert "status_code" in cooldown_data
        assert "timestamp" in cooldown_data
        assert "cooldown_time" in cooldown_data
        assert isinstance(cooldown_data["exception_received"], str)
        assert isinstance(cooldown_data["status_code"], str)
        assert isinstance(cooldown_data["timestamp"], float)
        assert isinstance(cooldown_data["cooldown_time"], float)
        assert cooldown_data["status_code"] == str(exception_status)
        assert cooldown_data["cooldown_time"] == cooldown_time
        assert cooldown_data["exception_received"] == exception_msg

    def test_exception_object_conversion(self, cooldown_cache):
        """Test that different exception types are properly converted to strings."""
        exceptions = [
            ValueError("Invalid value provided"),
            KeyError("Missing required key"),
            RuntimeError("Runtime error occurred"),
            Exception("Generic exception"),
        ]

        for exc in exceptions:
            model_id = f"test-model-{exc.__class__.__name__}"

            _, cooldown_data = cooldown_cache._common_add_cooldown_logic(
                model_id=model_id,
                original_exception=exc,
                exception_status=500,
                cooldown_time=30.0,
            )

            assert isinstance(cooldown_data["exception_received"], str)
            assert str(exc) == cooldown_data["exception_received"]

    def test_masking_preserves_error_debugging_info(self, cooldown_cache):
        """Test that masking preserves essential debugging information."""
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

        _, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=Exception(debugging_exception),
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        masked_exception = cooldown_data["exception_received"]

        assert masked_exception.startswith(
            "RateLimitError: Rate limit exceeded for model gpt-"
        )
        assert "Write a comprehensive analysis" not in masked_exception
        assert "healthcare sector" not in masked_exception
        assert "*" in masked_exception

    def test_error_handling_in_common_add_cooldown_logic(self, cooldown_cache):
        """Test error handling in the _common_add_cooldown_logic method."""
        model_id = "test-model"

        cooldown_key, cooldown_data = cooldown_cache._common_add_cooldown_logic(
            model_id=model_id,
            original_exception=None,
            exception_status=500,
            cooldown_time=30.0,
        )

        assert cooldown_data["exception_received"] == "None"
        assert cooldown_key == f"deployment:{model_id}:cooldown"

    def test_custom_masker_settings(self):
        """Test that custom masker settings work correctly."""
        mock_dual_cache = MagicMock(spec=DualCache)
        cache = CooldownCache(cache=mock_dual_cache, default_cooldown_time=60.0)

        assert cache.exception_masker.visible_prefix == 50
        assert cache.exception_masker.visible_suffix == 0
        assert cache.exception_masker.mask_char == "*"

        long_string = "A" * 100
        masked = cache.exception_masker._mask_value(long_string)

        expected = "A" * 50 + "*" * 50
        assert masked == expected


def test_get_active_cooldowns_ignores_expired_cache_payload(monkeypatch):
    cache = DualCache()
    cooldown_cache = CooldownCache(cache=cache, default_cooldown_time=120)
    model_id = "deployment-1"

    monkeypatch.setattr("litellm.router_utils.cooldown_cache.time.time", lambda: 300)
    cache.set_cache(
        key=CooldownCache.get_cooldown_cache_key(model_id),
        value={
            "exception_received": "rate limit",
            "status_code": "429",
            "timestamp": 100,
            "cooldown_time": 120,
        },
        ttl=600,
    )

    assert (
        cooldown_cache.get_active_cooldowns(
            model_ids=[model_id],
            parent_otel_span=None,
        )
        == []
    )


@pytest.mark.asyncio
async def test_async_get_active_cooldowns_ignores_expired_cache_payload(monkeypatch):
    cache = DualCache()
    cooldown_cache = CooldownCache(cache=cache, default_cooldown_time=120)
    model_id = "deployment-1"

    monkeypatch.setattr("litellm.router_utils.cooldown_cache.time.time", lambda: 300)
    cache.set_cache(
        key=CooldownCache.get_cooldown_cache_key(model_id),
        value={
            "exception_received": "rate limit",
            "status_code": "429",
            "timestamp": 100,
            "cooldown_time": 120,
        },
        ttl=600,
    )

    assert (
        await cooldown_cache.async_get_active_cooldowns(
            model_ids=[model_id],
            parent_otel_span=None,
        )
        == []
    )


def test_get_min_cooldown_ignores_expired_cache_payload(monkeypatch):
    cache = DualCache()
    cooldown_cache = CooldownCache(cache=cache, default_cooldown_time=120)
    expired_model_id = "expired-deployment"
    active_model_id = "active-deployment"

    monkeypatch.setattr("litellm.router_utils.cooldown_cache.time.time", lambda: 300)
    cache.set_cache(
        key=CooldownCache.get_cooldown_cache_key(expired_model_id),
        value={
            "exception_received": "rate limit",
            "status_code": "429",
            "timestamp": 100,
            "cooldown_time": 10,
        },
        ttl=600,
    )
    cache.set_cache(
        key=CooldownCache.get_cooldown_cache_key(active_model_id),
        value={
            "exception_received": "rate limit",
            "status_code": "429",
            "timestamp": 250,
            "cooldown_time": 120,
        },
        ttl=600,
    )

    assert (
        cooldown_cache.get_min_cooldown(
            model_ids=[expired_model_id, active_model_id],
            parent_otel_span=None,
        )
        == 120
    )
