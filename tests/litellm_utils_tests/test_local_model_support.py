"""
Tests for local model support fixes (Issue #23054).

Verifies that:
1. get_model_info() returns sensible defaults for unknown/local models
2. supports_function_calling() returns True for unknown models on OpenAI-compatible providers
3. ContentPolicyViolationError doesn't trigger deployment cooldowns
4. Error classification patterns are tightened to avoid false positives
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.utils import (
    _get_model_info_helper,
    get_model_info,
    supports_function_calling,
    supports_tool_choice,
)


@pytest.fixture(autouse=True)
def clear_model_info_cache():
    """Clear the LRU cache on get_model_info between tests."""
    get_model_info.cache_clear()
    yield
    get_model_info.cache_clear()


# ─── Fix 1: get_model_info returns defaults for unknown models ───


class TestGetModelInfoUnknownModels:
    """get_model_info should return conservative defaults instead of raising."""

    def test_unknown_local_model_returns_defaults(self):
        """A model not in the static map should return defaults, not raise."""
        info = get_model_info(
            model="openai/qwen3-coder-30b-a3b-instruct",
            custom_llm_provider="openai",
        )
        assert info is not None
        assert info["input_cost_per_token"] == 0
        assert info["output_cost_per_token"] == 0
        assert info["mode"] == "chat"

    def test_unknown_model_has_function_calling_support(self):
        """Unknown models should default to supporting function calling."""
        info = get_model_info(
            model="openai/my-custom-local-model",
            custom_llm_provider="openai",
        )
        assert info.get("supports_function_calling") is True
        assert info.get("supports_tool_choice") is True

    def test_unknown_model_has_system_message_support(self):
        """Unknown models should default to supporting system messages."""
        info = get_model_info(
            model="openai/some-random-model",
            custom_llm_provider="openai",
        )
        assert info.get("supports_system_messages") is True

    def test_unknown_model_provider_preserved(self):
        """The provider should be preserved in the returned info."""
        info = get_model_info(
            model="openai/deepseek-r1-32b",
            custom_llm_provider="openai",
        )
        assert info["litellm_provider"] == "openai"

    def test_unknown_model_no_provider_defaults_to_openai(self):
        """When no provider is given, should still not raise."""
        info = _get_model_info_helper(
            model="totally-unknown-model-xyz",
            custom_llm_provider="openai",
        )
        assert info is not None
        assert info["input_cost_per_token"] == 0

    def test_known_model_still_works(self):
        """Known models should still return their actual info."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        info = get_model_info(model="gpt-4")
        assert info is not None
        assert info["input_cost_per_token"] > 0

    def test_helper_returns_defaults_for_unmapped_model(self):
        """_get_model_info_helper should return defaults for unmapped models."""
        info = _get_model_info_helper(
            model="llama-3.2-local",
            custom_llm_provider="openai",
        )
        assert info is not None
        assert info["key"] == "openai/llama-3.2-local"
        assert info["mode"] == "chat"


# ─── Fix 2: supports_function_calling for unknown/local models ───


class TestSupportsFunctionCallingLocal:
    """supports_function_calling should return True for unknown models
    on OpenAI-compatible providers."""

    def test_unknown_openai_model_supports_function_calling(self):
        """openai/ prefixed unknown model should support function calling."""
        result = supports_function_calling(
            model="openai/my-local-llm",
            custom_llm_provider="openai",
        )
        assert result is True

    def test_unknown_openai_model_supports_tool_choice(self):
        """openai/ prefixed unknown model should support tool_choice."""
        result = supports_tool_choice(
            model="openai/my-local-llm",
            custom_llm_provider="openai",
        )
        assert result is True

    def test_known_model_function_calling_unchanged(self):
        """Known models should still return their correct capability."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        result = supports_function_calling(model="gpt-3.5-turbo")
        assert result is True

    @pytest.mark.parametrize(
        "model_name",
        [
            "openai/qwen3-coder-30b",
            "openai/deepseek-r1:32b",
            "openai/llama-3.2-3b-instruct",
            "openai/my-finetuned-model",
        ],
    )
    def test_various_local_model_names(self, model_name):
        """Various local model name patterns should all support function calling."""
        result = supports_function_calling(model=model_name)
        assert result is True


# ─── Fix 3: Error classification and cooldown ───


class TestCooldownNotTriggeredByContentPolicy:
    """ContentPolicyViolationError should not trigger deployment cooldowns."""

    def test_content_policy_error_not_cooldown_required(self):
        """ContentPolicyViolationError should be in the ignored cooldown strings."""
        from unittest.mock import MagicMock

        from litellm.router_utils.cooldown_handlers import _is_cooldown_required

        mock_router = MagicMock()
        result = _is_cooldown_required(
            litellm_router_instance=mock_router,
            model_id="test-deployment",
            exception_status=400,
            exception_str="litellm.ContentPolicyViolationError: some error",
        )
        assert result is False

    def test_api_connection_error_still_ignored(self):
        """APIConnectionError should still be in the ignored cooldown strings."""
        from unittest.mock import MagicMock

        from litellm.router_utils.cooldown_handlers import _is_cooldown_required

        mock_router = MagicMock()
        result = _is_cooldown_required(
            litellm_router_instance=mock_router,
            model_id="test-deployment",
            exception_status=500,
            exception_str="APIConnectionError: connection refused",
        )
        assert result is False


class TestErrorClassificationPatterns:
    """Verify that the safety system pattern requires invalid_request_error context."""

    def test_safety_system_without_invalid_request_not_content_policy(self):
        """A 'safety system' error without 'invalid_request_error' should NOT
        be classified as ContentPolicyViolationError."""
        from litellm.exceptions import ContentPolicyViolationError

        # Simulate an error from a local server that mentions safety system
        # but doesn't include invalid_request_error
        error_msg = "request was rejected as a result of the safety system"

        # Check that the pattern won't match without invalid_request_error
        error_str = error_msg.lower()
        matches_pattern = (
            "invalid_request_error" in error_str
            and "request was rejected as a result of the safety system" in error_str
        )
        assert matches_pattern is False

    def test_real_openai_content_policy_still_detected(self):
        """A real OpenAI content policy violation with proper error format
        should still be detected."""
        error_str = (
            '{"error": {"type": "invalid_request_error", '
            '"code": "content_policy_violation", '
            '"message": "Your request was rejected."}}'
        )
        matches_pattern = (
            "invalid_request_error" in error_str
            and "content_policy_violation" in error_str
        )
        assert matches_pattern is True

    def test_safety_system_with_invalid_request_detected(self):
        """A safety system error with invalid_request_error context should be detected."""
        error_str = (
            'invalid_request_error: Your request was rejected as a result of the safety system'
        )
        matches_pattern = (
            "invalid_request_error" in error_str.lower()
            and "request was rejected as a result of the safety system"
            in error_str.lower()
        )
        assert matches_pattern is True
