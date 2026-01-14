"""
Tests for OpenAI GPT transformation (litellm/llms/openai/chat/gpt_transformation.py)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class TestOpenAIGPTConfig:
    """Tests for OpenAIGPTConfig class"""

    def setup_method(self):
        self.config = OpenAIGPTConfig()

    def test_user_param_supported_for_regular_models(self):
        """Test that 'user' param is in supported params for regular OpenAI models."""
        supported_params = self.config.get_supported_openai_params("gpt-4o")
        assert "user" in supported_params

        supported_params = self.config.get_supported_openai_params("gpt-4.1-mini")
        assert "user" in supported_params

    def test_user_param_supported_for_responses_api_models(self):
        """Test that 'user' param is in supported params for responses API models.

        Regression test for: https://github.com/BerriAI/litellm/issues/17633
        When using model="openai/responses/gpt-4.1", the 'user' parameter should
        be included in supported params so it reaches OpenAI and SpendLogs.
        """
        # responses/gpt-4.1-mini should support 'user' just like gpt-4.1-mini
        supported_params = self.config.get_supported_openai_params("responses/gpt-4.1-mini")
        assert "user" in supported_params

        supported_params = self.config.get_supported_openai_params("responses/gpt-4o")
        assert "user" in supported_params

        supported_params = self.config.get_supported_openai_params("responses/gpt-4.1")
        assert "user" in supported_params

    def test_model_normalization_for_responses_prefix(self):
        """Test that models with 'responses/' prefix are normalized correctly.

        The fix normalizes 'responses/gpt-4.1' to 'gpt-4.1' when checking
        if the model is in the list of supported OpenAI models.
        """
        # Both should have the same supported params
        regular_params = self.config.get_supported_openai_params("gpt-4.1-mini")
        responses_params = self.config.get_supported_openai_params("responses/gpt-4.1-mini")

        # 'user' should be in both
        assert "user" in regular_params
        assert "user" in responses_params

    def test_base_params_always_included(self):
        """Test that base params are always included regardless of model."""
        base_expected_params = [
            "frequency_penalty",
            "max_tokens",
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
        ]

        supported_params = self.config.get_supported_openai_params("responses/gpt-4.1-mini")

        for param in base_expected_params:
            assert param in supported_params, f"Expected '{param}' in supported params"


class TestGetOptionalParamsIntegration:
    """Integration tests using litellm.get_optional_params()"""

    def test_user_in_optional_params_for_responses_model(self):
        """Test that 'user' ends up in optional_params when using responses API models.

        Regression test for: https://github.com/BerriAI/litellm/issues/17633
        This verifies the full flow through get_optional_params().
        """
        from litellm.utils import get_optional_params

        # Test with responses model
        optional_params = get_optional_params(
            model="responses/gpt-4.1-mini",
            custom_llm_provider="openai",
            user="test-user-123",
        )
        assert optional_params.get("user") == "test-user-123"

    def test_user_in_optional_params_for_regular_model(self):
        """Test that 'user' ends up in optional_params for regular OpenAI models."""
        from litellm.utils import get_optional_params

        optional_params = get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
            user="test-user-456",
        )
        assert optional_params.get("user") == "test-user-456"

    def test_user_param_consistency_between_regular_and_responses(self):
        """Test that 'user' param behavior is consistent between regular and responses models."""
        from litellm.utils import get_optional_params

        regular_params = get_optional_params(
            model="gpt-4.1-mini",
            custom_llm_provider="openai",
            user="my-end-user",
        )

        responses_params = get_optional_params(
            model="responses/gpt-4.1-mini",
            custom_llm_provider="openai",
            user="my-end-user",
        )

        # Both should include user
        assert regular_params.get("user") == "my-end-user"
        assert responses_params.get("user") == "my-end-user"
