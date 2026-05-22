"""
Unit tests for BaseLLMChatTransformation helper methods.
"""

import pytest

from litellm.llms.base_llm.chat.transformation import BaseConfig


class _ConcreteTransformation(BaseConfig):
    """Minimal concrete subclass for testing — stubs out all abstract methods."""

    def get_supported_openai_params(self, model: str) -> list:
        return []

    def map_openai_params(self, non_default_params, optional_params, model, drop_params):  # type: ignore[override]
        return optional_params

    def validate_environment(self, headers, model, messages, optional_params, litellm_params, api_key=None, api_base=None):  # type: ignore[override]
        return headers

    def transform_request(self, model, messages, optional_params, litellm_params, headers):  # type: ignore[override]
        return {}

    def transform_response(self, model, raw_response, model_response, logging_obj, request_data, messages, optional_params, litellm_params, encoding, api_key=None, json_mode=None):  # type: ignore[override]
        return model_response

    def get_error_class(self, error_message, status_code, headers=None):  # type: ignore[override]
        raise NotImplementedError


class TestIsThinkingEnabled:
    """Tests for BaseLLMChatTransformation.is_thinking_enabled."""

    def setup_method(self):
        self.transformation = _ConcreteTransformation()

    def test_thinking_none_returns_false(self):
        """thinking=None must not raise AttributeError — returns False (GitHub #28576)."""
        assert self.transformation.is_thinking_enabled({"thinking": None}) is False

    def test_thinking_enabled(self):
        """thinking={'type': 'enabled', 'budget_tokens': 1024} returns True."""
        assert (
            self.transformation.is_thinking_enabled(
                {"thinking": {"type": "enabled", "budget_tokens": 1024}}
            )
            is True
        )

    def test_thinking_disabled_type(self):
        """thinking={'type': 'disabled'} returns False."""
        assert (
            self.transformation.is_thinking_enabled({"thinking": {"type": "disabled"}})
            is False
        )

    def test_thinking_key_absent_returns_false(self):
        """No thinking key in params returns False."""
        assert self.transformation.is_thinking_enabled({}) is False

    def test_non_default_params_none_returns_false(self):
        """None dict returns False without raising."""
        assert self.transformation.is_thinking_enabled(None) is False  # type: ignore[arg-type]

    def test_reasoning_effort_set_returns_true(self):
        """reasoning_effort present (without thinking key) returns True."""
        assert (
            self.transformation.is_thinking_enabled({"reasoning_effort": "high"})
            is True
        )

    def test_reasoning_effort_none_returns_false(self):
        """reasoning_effort=None is treated as absent — returns False."""
        assert (
            self.transformation.is_thinking_enabled({"reasoning_effort": None}) is False
        )

    def test_thinking_non_dict_returns_false(self):
        """thinking set to a non-dict value (e.g. a string) returns False without raising."""
        assert self.transformation.is_thinking_enabled({"thinking": "enabled"}) is False
        assert self.transformation.is_thinking_enabled({"thinking": 1}) is False

    def test_thinking_and_reasoning_effort_both_set(self):
        """Both thinking enabled and reasoning_effort set returns True."""
        assert (
            self.transformation.is_thinking_enabled(
                {
                    "thinking": {"type": "enabled", "budget_tokens": 512},
                    "reasoning_effort": "medium",
                }
            )
            is True
        )
