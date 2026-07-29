"""
Unit tests for is_thinking_enabled method in BaseConfig.

Tests the fix for issue #28576: handle None thinking param without crashing.
"""

import pytest
from litellm.llms.base_llm.chat.transformation import BaseConfig


class TestIsThinkingEnabled:
    """Test is_thinking_enabled handles various thinking parameter values."""

    @pytest.fixture
    def transformer(self):
        """Create a BaseConfig instance for testing."""
        # BaseConfig is abstract, so we create a minimal concrete subclass
        class ConcreteConfig(BaseConfig):
            def __init__(self):
                pass

            def get_complete_url(self, *args, **kwargs):
                return ""

            def validate_environment(self, *args, **kwargs):
                return {}

            def transform_request(self, *args, **kwargs):
                return {}, {}

            def transform_response(self, *args, **kwargs):
                return None

            def get_supported_openai_params(self, model: str):
                return []

            def map_openai_params(self, *args, **kwargs):
                return {}

            def get_error_class(self, *args, **kwargs):
                from litellm.llms.base_llm.chat.transformation import BaseLLMException
                return BaseLLMException(500, "test error")

        return ConcreteConfig()

    @pytest.mark.parametrize(
        "non_default_params,expected",
        [
            # thinking=None should not crash, returns False
            ({"thinking": None}, False),
            # thinking={'type': 'enabled'} returns True
            ({"thinking": {"type": "enabled"}}, True),
            # thinking key missing returns False
            ({}, False),
            # thinking={} returns False
            ({"thinking": {}}, False),
            # thinking with different type returns False
            ({"thinking": {"type": "disabled"}}, False),
            # reasoning_effort present returns True
            ({"reasoning_effort": "medium"}, True),
            # both thinking enabled and reasoning_effort returns True
            ({"thinking": {"type": "enabled"}, "reasoning_effort": "high"}, True),
            # falsy thinking values should not crash
            ({"thinking": False}, False),
            ({"thinking": 0}, False),
            ({"thinking": ""}, False),
        ],
    )
    def test_is_thinking_enabled(self, transformer, non_default_params, expected):
        """Test is_thinking_enabled with various parameter combinations."""
        result = transformer.is_thinking_enabled(non_default_params)
        assert result == expected, (
            f"Expected {expected} for params {non_default_params}, got {result}"
        )
