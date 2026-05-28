"""
LIT-2594: cache_control_injection_points is broken with Responses API

These tests pin the propagation of message-level `cache_control` from the
Responses-API input items onto the resulting Chat-Completion messages, so that
`anthropic_messages_pt` / `AnthropicConfig.translate_system_message` can apply
it to system/user/assistant blocks.

Before the fix, AnthropicCacheControlHook wrote `cache_control` onto each input
message (chat-completion format), but the subsequent
`_transform_responses_api_input_item_to_chat_completion_message` call dropped
every field except `role`+`content`, so `cache_control` never reached the
provider transform and prompt caching silently no-op-ed.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.anthropic_cache_control_hook import (
    AnthropicCacheControlHook,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    anthropic_messages_pt,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


class TestCacheControlPropagationLit2594:
    def _ephemeral(self):
        return {"type": "ephemeral"}

    def test_string_user_message_with_cache_control_is_preserved(self):
        input_item = {
            "role": "user",
            "content": "What is the capital of France?",
            "cache_control": self._ephemeral(),
        }
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item,
        )
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert msg.get("cache_control") == self._ephemeral()

    def test_string_system_message_with_cache_control_is_preserved(self):
        input_item = {
            "role": "system",
            "content": "You are a helpful assistant.",
            "cache_control": self._ephemeral(),
        }
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item,
        )
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "system"
        assert msg.get("cache_control") == self._ephemeral()

    def test_assistant_message_with_cache_control_is_preserved(self):
        input_item = {
            "role": "assistant",
            "content": "France is a country in Europe.",
            "cache_control": self._ephemeral(),
        }
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item,
        )
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg.get("cache_control") == self._ephemeral()

    def test_message_without_cache_control_has_no_key(self):
        """No cache_control on the input item => the resulting message must not have a stray cache_control key."""
        input_item = {"role": "user", "content": "Hi"}
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item,
        )
        assert len(result) == 1
        assert "cache_control" not in result[0]

    def test_list_content_with_cache_control_is_preserved(self):
        input_item = {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
            "cache_control": self._ephemeral(),
        }
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item,
        )
        assert len(result) == 1
        assert result[0].get("cache_control") == self._ephemeral()

    def test_none_content_short_circuits_to_empty(self):
        input_item = {
            "role": "user",
            "content": None,
            "cache_control": self._ephemeral(),
        }
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
            input_item=input_item,
        )
        assert result == []

    def _build_after_hook_messages(self):
        responses_input = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about France."},
            {"role": "assistant", "content": "France is a country in Europe."},
            {"role": "user", "content": "What is its capital?"},
        ]
        cache_control_injection_points = [
            {"location": "message", "role": "system", "control": self._ephemeral()},
            {"location": "message", "index": -1, "control": self._ephemeral()},
        ]
        hook = AnthropicCacheControlHook()
        non_default_params = {
            "cache_control_injection_points": list(cache_control_injection_points)
        }
        _, after_hook, _ = hook.get_chat_completion_prompt(
            model="anthropic/claude-3-5-sonnet",
            messages=responses_input,
            non_default_params=non_default_params,
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},  # type: ignore[arg-type]
        )
        return after_hook

    def test_anthropic_system_block_carries_cache_control(self):
        after_hook = self._build_after_hook_messages()
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=after_hook,
        )
        system_blocks = AnthropicConfig().translate_system_message(messages=messages)  # type: ignore[arg-type]
        assert len(system_blocks) == 1
        assert system_blocks[0].get("cache_control") == self._ephemeral()

    def test_anthropic_last_user_message_carries_cache_control(self):
        after_hook = self._build_after_hook_messages()
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=after_hook,
        )
        non_system = [m for m in messages if m.get("role") != "system"]
        anthropic_msgs = anthropic_messages_pt(
            messages=non_system,  # type: ignore[arg-type]
            model="claude-3-5-sonnet",
            llm_provider="anthropic",
        )
        last_user = anthropic_msgs[-1]
        assert last_user["role"] == "user"
        last_block = last_user["content"][-1]
        assert last_block.get("cache_control") == self._ephemeral()

    def test_anthropic_middle_user_message_has_no_cache_control(self):
        after_hook = self._build_after_hook_messages()
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=after_hook,
        )
        non_system = [m for m in messages if m.get("role") != "system"]
        anthropic_msgs = anthropic_messages_pt(
            messages=non_system,  # type: ignore[arg-type]
            model="claude-3-5-sonnet",
            llm_provider="anthropic",
        )
        first_user = anthropic_msgs[0]
        assert first_user["role"] == "user"
        for block in first_user["content"]:
            assert "cache_control" not in block

    def test_baseline_without_injection_points_unchanged(self):
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )
        for m in messages:
            assert "cache_control" not in m
