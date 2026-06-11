"""
Unit tests for DeepSeek chat transformation.

Tests the thinking and reasoning_effort parameter handling for DeepSeek models.
"""

import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped from thinking param (DeepSeek doesn't support it)."""
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should strip budget_tokens, only pass type
        assert result["thinking"] == {"type": "enabled"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_map_reasoning_effort_medium(self):
        """Test that reasoning_effort='medium' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_none_does_not_enable_thinking(self):
        """Test that reasoning_effort='none' does not enable thinking."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_map_reasoning_effort_null_does_not_enable_thinking(self):
        """Test that reasoning_effort=None does not enable thinking."""
        non_default_params = {"reasoning_effort": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_takes_precedence_over_reasoning_effort(self):
        """Test that thinking param takes precedence when both are provided."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # thinking should be set, reasoning_effort should not override
        assert result["thinking"] == {"type": "enabled"}

    def test_invalid_thinking_type_ignored(self):
        """Test that invalid thinking type values are ignored."""
        non_default_params = {"thinking": {"type": "invalid"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_none_value_ignored(self):
        """Test that thinking=None is ignored."""
        non_default_params = {"thinking": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_map_thinking_disabled_passed_through(self):
        """thinking={"type": "disabled"} must be forwarded so users can opt out
        of DeepSeek V4's default-on thinking mode."""
        result = self.config.map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="deepseek-v4-flash",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}


class TestDeepSeekV4DefaultThinkingMode:
    """
    DeepSeek V4 models run in thinking mode BY DEFAULT and require
    `reasoning_content` to be passed back on assistant messages
    (https://github.com/BerriAI/litellm/issues/26395).
    """

    def setup_method(self):
        self.config = DeepSeekChatConfig()

    # --- registry ---

    @pytest.mark.parametrize(
        "model",
        ["deepseek-v4-flash", "deepseek-v4-pro"],
    )
    def test_v4_models_registered_with_reasoning(self, model):
        from litellm.utils import supports_reasoning

        assert supports_reasoning(model=model, custom_llm_provider="deepseek")
        assert supports_reasoning(model=f"deepseek/{model}")

    # --- _thinking_mode_active guard ---

    @pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
    def test_thinking_active_by_default_for_v4(self, model):
        assert self.config._thinking_mode_active(model=model, optional_params={})

    @pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
    def test_thinking_inactive_when_explicitly_disabled(self, model):
        assert not self.config._thinking_mode_active(
            model=model, optional_params={"thinking": {"type": "disabled"}}
        )

    @pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
    def test_thinking_active_when_explicitly_enabled(self, model):
        assert self.config._thinking_mode_active(
            model=model, optional_params={"thinking": {"type": "enabled"}}
        )

    def test_reasoning_capable_models_active_by_default(self):
        """Any reasoning-capable DeepSeek model counts as potentially
        thinking-mode (V4 enables thinking by default). Injection is harmless
        when thinking is not actually active: the live API ignores
        reasoning_content in non-thinking requests."""
        assert self.config._thinking_mode_active(
            model="deepseek-v3.2", optional_params={}
        )
        assert self.config._thinking_mode_active(
            model="deepseek-v3.2", optional_params={"thinking": {"type": "enabled"}}
        )

    def test_non_reasoning_model_unaffected(self):
        assert not self.config._thinking_mode_active(
            model="deepseek-chat", optional_params={}
        )

    # --- end-to-end transform_request ---

    def _tool_call_history(self):
        return [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_client_generated_id",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
                # reasoning_content stripped, tool_call id rewritten:
                # standard behavior of frontends/agent frameworks, and the
                # exact request shape DeepSeek rejects with
                # "The `reasoning_content` in the thinking mode must be passed back to the API."
            },
            {
                "role": "tool",
                "tool_call_id": "call_client_generated_id",
                "content": '{"weather": "Sunny", "temp_c": 28}',
            },
        ]

    def test_transform_request_injects_reasoning_content_for_v4_by_default(self):
        body = self.config.transform_request(
            model="deepseek-v4-flash",
            messages=self._tool_call_history(),
            optional_params={},
            litellm_params={},
            headers={},
        )
        assistant_msg = body["messages"][1]
        assert assistant_msg["reasoning_content"] == " "

    def test_transform_request_no_injection_when_thinking_disabled(self):
        body = self.config.transform_request(
            model="deepseek-v4-flash",
            messages=self._tool_call_history(),
            optional_params={"thinking": {"type": "disabled"}},
            litellm_params={},
            headers={},
        )
        assistant_msg = body["messages"][1]
        assert "reasoning_content" not in assistant_msg

    def test_transform_request_preserves_existing_reasoning_content(self):
        messages = self._tool_call_history()
        messages[1]["reasoning_content"] = "I should check the weather tool."
        body = self.config.transform_request(
            model="deepseek-v4-pro",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assistant_msg = body["messages"][1]
        assert assistant_msg["reasoning_content"] == "I should check the weather tool."
