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


@pytest.fixture(autouse=True)
def restore_global_state():
    """Save and restore global state that tests may mutate."""
    orig_model_cost = litellm.model_cost.copy()
    orig_env = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    yield
    litellm.model_cost = orig_model_cost
    if orig_env is None:
        os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    else:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = orig_env


# ─── Fix 1: get_model_info returns defaults for unknown models ───


class TestGetModelInfoUnknownModels:
    """Unknown models should return provider-aware defaults instead of raising."""

    def test_unknown_local_model_returns_provider_aware_defaults(self):
        """A model not in the static map should return safe OpenAI-compatible defaults."""
        info = get_model_info(
            model="openai/qwen3-coder-30b-a3b-instruct",
            custom_llm_provider="openai",
        )

        assert info["input_cost_per_token"] == 0
        assert info["output_cost_per_token"] == 0
        assert info["mode"] == "chat"
        assert info["supports_function_calling"] is True
        assert info["supports_tool_choice"] is True

    def test_known_model_still_works(self):
        """Known models should still return their actual info."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        info = get_model_info(model="gpt-4")
        assert info is not None
        assert info["input_cost_per_token"] > 0

    def test_helper_returns_defaults_for_unmapped_openai_model(self):
        """_get_model_info_helper should return defaults for unmapped OpenAI-compatible models."""
        info = _get_model_info_helper(
            model="llama-3.2-local",
            custom_llm_provider="openai",
        )

        assert info["supports_function_calling"] is True
        assert info["supports_tool_choice"] is True
        assert info["input_cost_per_token"] == 0
        assert info["output_cost_per_token"] == 0

    def test_helper_disables_tool_capabilities_for_text_completion_provider(self):
        """text-completion-openai should keep tool capabilities disabled."""
        info = _get_model_info_helper(
            model="davinci-002-local",
            custom_llm_provider="text-completion-openai",
        )

        assert info["supports_function_calling"] is False
        assert info["supports_tool_choice"] is False

    def test_unknown_model_no_provider_defaults_to_openai(self):
        """When no provider is given, should still not raise for supports_* checks."""
        result = supports_function_calling(
            model="openai/my-local-llm",
            custom_llm_provider=None,
        )
        assert result is True


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

    def test_text_completion_openai_no_function_calling(self):
        """text-completion-openai uses /v1/completions which does not support
        function calling or tool choice."""
        result = supports_function_calling(
            model="davinci-002",
            custom_llm_provider="text-completion-openai",
        )
        assert result is False

    def test_text_completion_openai_no_tool_choice(self):
        """text-completion-openai should not support tool_choice."""
        result = supports_tool_choice(
            model="davinci-002",
            custom_llm_provider="text-completion-openai",
        )
        assert result is False

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

    def test_content_policy_error_not_cooldown_required_5xx(self):
        """ContentPolicyViolationError with 5xx status should still not
        trigger cooldown (per-request, not deployment-level)."""
        from unittest.mock import MagicMock

        from litellm.router_utils.cooldown_handlers import _is_cooldown_required

        mock_router = MagicMock()
        result = _is_cooldown_required(
            litellm_router_instance=mock_router,
            model_id="test-deployment",
            exception_status=500,
            exception_str="litellm.ContentPolicyViolationError: some error",
        )
        assert result is False

    def test_content_policy_error_not_cooldown_required_4xx(self):
        """ContentPolicyViolationError with 4xx status should not trigger cooldown."""
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
    """Verify that the safety system pattern requires invalid_request_error context
    by exercising exception_type() directly."""

    def test_safety_system_without_invalid_request_not_content_policy(self):
        """A 'safety system' error without 'invalid_request_error' should NOT
        be classified as ContentPolicyViolationError via exception_type()."""
        from litellm.exceptions import ContentPolicyViolationError
        from litellm.litellm_core_utils.exception_mapping_utils import exception_type

        original_exc = Exception(
            "request was rejected as a result of the safety system"
        )

        raised_exc = None
        try:
            exception_type(
                model="my-local-model",
                original_exception=original_exc,
                custom_llm_provider="openai",
            )
        except ContentPolicyViolationError:
            raised_exc = "ContentPolicyViolationError"
        except Exception:
            raised_exc = "other"

        assert raised_exc != "ContentPolicyViolationError", (
            "Safety system error without invalid_request_error should not raise "
            "ContentPolicyViolationError"
        )

    def test_real_openai_content_policy_detected_via_exception_type(self):
        """A real OpenAI content policy violation should be classified as
        ContentPolicyViolationError via exception_type()."""
        from litellm.exceptions import ContentPolicyViolationError
        from litellm.litellm_core_utils.exception_mapping_utils import exception_type

        error_msg = (
            '{"error": {"type": "invalid_request_error", '
            '"code": "content_policy_violation", '
            '"message": "Your request was rejected."}}'
        )
        original_exc = Exception(error_msg)
        original_exc.status_code = 400

        with pytest.raises(ContentPolicyViolationError):
            exception_type(
                model="gpt-4",
                original_exception=original_exc,
                custom_llm_provider="openai",
            )

    def test_safety_system_with_invalid_request_detected_via_exception_type(self):
        """A safety system error WITH invalid_request_error context should be
        classified as ContentPolicyViolationError via exception_type()."""
        from litellm.exceptions import ContentPolicyViolationError
        from litellm.litellm_core_utils.exception_mapping_utils import exception_type

        error_msg = (
            "Invalid_Request_Error: Your request was rejected "
            "as a result of the safety system"
        )
        original_exc = Exception(error_msg)
        original_exc.status_code = 400

        with pytest.raises(ContentPolicyViolationError):
            exception_type(
                model="gpt-4",
                original_exception=original_exc,
                custom_llm_provider="openai",
            )

    def test_safety_system_with_mixed_case_detected(self):
        """Case-insensitive matching: mixed-case 'Invalid_Request_Error'
        and 'Safety System' should still be detected."""
        from litellm.exceptions import ContentPolicyViolationError
        from litellm.litellm_core_utils.exception_mapping_utils import exception_type

        error_msg = (
            "Invalid_Request_Error: Request was rejected "
            "as a result of the Safety System"
        )
        original_exc = Exception(error_msg)
        original_exc.status_code = 400

        with pytest.raises(ContentPolicyViolationError):
            exception_type(
                model="gpt-4",
                original_exception=original_exc,
                custom_llm_provider="openai",
            )
