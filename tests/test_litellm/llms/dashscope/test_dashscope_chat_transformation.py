"""
Unit tests for DashScope configuration.

These tests validate the DashScopeConfig class which extends OpenAIGPTConfig.
DashScope is an OpenAI-compatible provider with minor customizations.
"""



import json

import pytest

import litellm
from litellm import completion
from litellm.llms.dashscope.chat.transformation import DashScopeChatConfig
from litellm.types.llms.openai import AllMessageValues


class TestDashScopeConfig:
    """Test class for DashScope functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = DashScopeChatConfig()
        headers = {}
        api_key = "fake-dashscope-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="qwen-turbo",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

        # We can't directly test the api_base value here since validate_environment
        # only returns the headers, but we can verify it doesn't raise an exception
        # which would happen if api_base handling was incorrect

    @pytest.mark.respx()
    def test_dashscope_completion_mock(self, respx_mock):
        """
        Mock test for Dashscope completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )

        # Set up environment variables for the test
        api_key = "fake-dashscope-key"
        api_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        model = "dashscope/qwen-turbo"
        model_name = "qwen-turbo"  # The actual model name without provider prefix

        # Mock the HTTP request to the dashscope API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```python\nprint("Hey from LiteLLM!")\n```\n\nThis simple Python code prints a greeting message from LiteLLM.',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "write code for saying hey from LiteLLM"}
            ],
            api_key=api_key,
            api_base=api_base,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert response.choices[0].message.content is not None

        # Check for specific content in the response
        assert "```python" in response.choices[0].message.content
        assert "Hey from LiteLLM" in response.choices[0].message.content

    def test_dashscope_no_longer_transforms_content_list(self):
        """
        Test that DashScopeChatConfig does not transform content lists to strings.
        This ensures that the transformation logic specific to content lists is not applied,
        as DashScope should handle content in list format natively.
        """
        config = DashScopeChatConfig()

        # Create a message with content in list format
        messages: list[AllMessageValues] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }
        ]

        # Call the _transform_messages method directly
        transformed_messages = config._transform_messages(
            messages=messages, model="qwen-turbo", is_async=False
        )

        # Verify that the content is still in list format and has not been transformed to a string
        assert isinstance(transformed_messages[0]["content"], list)
        assert len(transformed_messages[0]["content"]) == 2
        assert transformed_messages[0]["content"][0]["type"] == "text"
        assert transformed_messages[0]["content"][0]["text"] == "Hello"
        assert transformed_messages[0]["content"][1]["type"] == "text"
        assert transformed_messages[0]["content"][1]["text"] == "World"

    def test_dashscope_preserves_cache_control_in_messages(self):
        """DashScope should NOT strip cache_control from messages."""
        config = DashScopeChatConfig()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "role": "user",
                "content": "Hello, world!",
            },
        ]

        transformed_messages, _ = (
            config.remove_cache_control_flag_from_messages_and_tools(
                model="dashscope/qwen-turbo", messages=messages
            )
        )

        assert transformed_messages[0].get("cache_control") == {"type": "ephemeral"}

    def test_dashscope_preserves_cache_control_in_tools(self):
        """DashScope should NOT strip cache_control from tools."""
        config = DashScopeChatConfig()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {"type": "object", "properties": {}},
                },
                "cache_control": {"type": "ephemeral"},
            }
        ]

        _, transformed_tools = config.remove_cache_control_flag_from_messages_and_tools(
            model="dashscope/qwen-turbo", messages=[], tools=tools
        )

        assert transformed_tools[0].get("cache_control") == {"type": "ephemeral"}


class TestDashScopeThinkingParams:
    """thinking and reasoning_effort reach DashScope as enable_thinking, thinking_budget and reasoning_effort."""

    @staticmethod
    def _map(**non_default_params: object) -> dict[str, object]:
        return DashScopeChatConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="qwen3.8-max",
            drop_params=False,
        )

    @pytest.mark.parametrize(
        ("thinking", "expected"),
        [
            ({"type": "enabled", "budget_tokens": 4096}, {"enable_thinking": True, "thinking_budget": 4096}),
            ({"type": "enabled"}, {"enable_thinking": True}),
            ({"type": "disabled"}, {"enable_thinking": False}),
            ({"type": "disabled", "budget_tokens": 4096}, {"enable_thinking": False}),
        ],
    )
    def test_thinking_maps_to_enable_thinking_and_budget(self, thinking, expected):
        assert self._map(thinking=thinking)["extra_body"] == expected

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "minimal"])
    def test_reasoning_effort_enables_thinking_and_is_forwarded(self, effort):
        params = self._map(reasoning_effort=effort)

        assert params["extra_body"] == {"enable_thinking": True, "reasoning_effort": effort}

    @pytest.mark.parametrize("effort", ["none", "disable"])
    def test_reasoning_effort_off_disables_thinking(self, effort):
        assert self._map(reasoning_effort=effort)["extra_body"] == {"enable_thinking": False}

    def test_thinking_budget_wins_over_reasoning_effort(self):
        params = self._map(thinking={"type": "enabled", "budget_tokens": 512}, reasoning_effort="high")

        assert params["extra_body"] == {"enable_thinking": True, "thinking_budget": 512}

    def test_no_thinking_params_leaves_extra_body_absent(self):
        params = self._map(temperature=0.5, max_tokens=16)

        assert "extra_body" not in params
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 16

    def test_thinking_params_never_become_top_level_kwargs(self):
        params = self._map(thinking={"type": "enabled"}, reasoning_effort="high")

        assert "thinking" not in params
        assert "reasoning_effort" not in params

    def test_existing_extra_body_is_preserved_and_not_mutated(self):
        caller_extra_body = {"enable_search": True}
        params = DashScopeChatConfig().map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={"extra_body": caller_extra_body},
            model="qwen3.8-max",
            drop_params=False,
        )

        assert params["extra_body"] == {"enable_search": True, "enable_thinking": True}
        assert caller_extra_body == {"enable_search": True}

    def test_caller_optional_params_are_not_mutated(self):
        caller_optional_params = {"temperature": 0.2, "extra_body": {"enable_search": True}}
        params = DashScopeChatConfig().map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}, "max_tokens": 8},
            optional_params=caller_optional_params,
            model="qwen3.8-max",
            drop_params=False,
        )

        assert params == {
            "temperature": 0.2,
            "max_tokens": 8,
            "extra_body": {"enable_search": True, "enable_thinking": False},
        }
        assert params is not caller_optional_params
        assert caller_optional_params == {"temperature": 0.2, "extra_body": {"enable_search": True}}

    def test_get_optional_params_accepts_thinking_without_drop_params(self):
        params = litellm.get_optional_params(
            model="qwen3.8-max",
            custom_llm_provider="dashscope",
            thinking={"type": "disabled"},
            drop_params=False,
        )

        assert params["extra_body"]["enable_thinking"] is False
        assert "thinking" not in params

    @pytest.mark.respx()
    def test_thinking_disabled_reaches_the_wire_as_enable_thinking(self, respx_mock):
        litellm.disable_aiohttp_transport = True
        api_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        route = respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-456",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "qwen3.8-max",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            },
            status_code=200,
        )

        completion(
            model="dashscope/qwen3.8-max",
            messages=[{"role": "user", "content": "2+2?"}],
            api_key="fake-dashscope-key",
            api_base=api_base,
            thinking={"type": "disabled"},
            drop_params=False,
        )

        sent = json.loads(route.calls.last.request.content)
        assert sent["enable_thinking"] is False
        assert "thinking" not in sent
        assert "extra_body" not in sent
