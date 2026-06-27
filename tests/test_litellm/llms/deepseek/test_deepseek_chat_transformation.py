"""Tests for DeepSeek V4 default-on thinking mode / reasoning_content pass-back."""

import pytest

from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


class TestDeepSeekV4DefaultThinkingMode:
    def setup_method(self):
        self.config = DeepSeekChatConfig()

    @pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
    def test_v4_models_registered_with_reasoning(self, model):
        from litellm.utils import supports_reasoning

        assert supports_reasoning(model=model, custom_llm_provider="deepseek")
        assert supports_reasoning(model=f"deepseek/{model}")

    def test_map_thinking_disabled_passed_through(self):
        result = self.config.map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="deepseek-v4-flash",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

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

    def test_non_reasoning_model_unaffected(self):
        assert not self.config._thinking_mode_active(
            model="deepseek-chat", optional_params={}
        )

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
        assert body["messages"][1]["reasoning_content"] == " "

    def test_transform_request_no_injection_when_thinking_disabled(self):
        body = self.config.transform_request(
            model="deepseek-v4-flash",
            messages=self._tool_call_history(),
            optional_params={"thinking": {"type": "disabled"}},
            litellm_params={},
            headers={},
        )
        assert "reasoning_content" not in body["messages"][1]

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
        assert body["messages"][1]["reasoning_content"] == "I should check the weather tool."

    @pytest.mark.asyncio
    async def test_async_transform_request_injects_reasoning_content(self):
        body = await self.config.async_transform_request(
            model="deepseek-v4-flash",
            messages=self._tool_call_history(),
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["messages"][1]["reasoning_content"] == " "
