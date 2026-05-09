"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math
from typing import cast

import pytest
import respx

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token


@pytest.fixture
def zai_response():
    """Mock response from Z.AI API"""
    return {
        "id": "chatcmpl-zai-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def test_get_llm_provider_zai():
    """Test that get_llm_provider correctly identifies zai provider"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("zai/glm-4.6")
    assert model == "glm-4.6"
    assert provider == "zai"
    assert api_base == "https://api.z.ai/api/paas/v4"


def test_zai_in_provider_lists():
    """Test that zai is registered in all necessary provider lists"""
    assert "zai" in litellm.openai_compatible_providers
    assert "zai" in litellm.provider_list


def test_zai_models_in_model_cost():
    """Test that ZAI models are in the model cost map"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    zai_models = [
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5",
        "zai/glm-4.5v",
        "zai/glm-4.5-x",
        "zai/glm-4.5-air",
        "zai/glm-4.5-airx",
        "zai/glm-4-32b-0414-128k",
        "zai/glm-4.5-flash",
    ]

    for model in zai_models:
        assert model in litellm.model_cost, f"Model {model} not found in model_cost"
        assert litellm.model_cost[model]["litellm_provider"] == "zai"


def test_zai_glm46_cost_calculation():
    """Test the cost calculation for glm-4.6"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.6"
    info = litellm.model_cost[key]

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.6",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.6: $0.6/M input, $2.2/M output
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


def test_zai_flash_model_is_free():
    """Test that glm-4.5-flash has zero cost"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.5-flash"
    info = litellm.model_cost[key]

    assert info["input_cost_per_token"] == 0
    assert info["output_cost_per_token"] == 0


def test_glm47_supports_reasoning():
    """Test that GLM-4.7 supports reasoning"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.7"
    assert key in litellm.model_cost, f"Model {key} not found in model_cost"

    info = litellm.model_cost[key]
    assert info["supports_reasoning"] is True


def test_glm47_cost_calculation():
    """Test cost calculation for GLM-4.7"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.7",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.7: $0.6/M input, $2.2/M output (same as GLM-4.6)
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


@pytest.mark.asyncio
async def test_zai_completion_call(respx_mock, zai_response, monkeypatch):
    """Test completion call with zai provider using mocked response"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = await litellm.acompletion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert "api.z.ai" in str(request.url)
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"


def test_zai_sync_completion(respx_mock, zai_response, monkeypatch):
    """Test synchronous completion call"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25


class TestZAIMessageTransformation:
    """Tests for ZAI message content flattening.

    Issue: https://github.com/BerriAI/litellm/issues/25868
    GLM's Jinja chat template checks ``m.content is string`` and silently drops
    list-format content. ZAIChatConfig._transform_messages must flatten these
    before forwarding to z.ai.
    """

    def test_flatten_tool_message_content_list(self):
        """Tool message with list-format content is flattened to a plain string."""
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        config = ZAIChatConfig()
        messages = cast(
            list,
            [
                {"role": "user", "content": "What is the temperature in Tokyo?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_temp", "arguments": '{"city": "Tokyo"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "text", "text": "22.5\u00b0C, partly cloudy."}],
                },
            ],
        )

        result = config._transform_messages(messages=messages, model="glm-5.1")

        tool_msg = result[2]
        assert isinstance(tool_msg["content"], str), (
            f"Expected str content, got {type(tool_msg['content'])}"
        )
        assert tool_msg["content"] == "22.5\u00b0C, partly cloudy."

    def test_flatten_assistant_message_content_list(self):
        """Assistant message with list-format content is flattened to a plain string."""
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        config = ZAIChatConfig()
        messages = cast(
            list,
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Let me think about this."}],
                },
            ],
        )

        result = config._transform_messages(messages=messages, model="glm-5.1")

        assert result[0]["content"] == "Let me think about this."

    def test_string_content_passes_through_unchanged(self):
        """String content is not modified by the flattening step."""
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        config = ZAIChatConfig()
        messages = cast(
            list,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "tool", "tool_call_id": "c1", "content": "Result data"},
            ],
        )

        result = config._transform_messages(messages=messages, model="glm-5.1")

        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi there!"
        assert result[2]["content"] == "Result data"

    def test_flatten_content_parts_helper_multipart(self):
        """Multiple text parts are joined with newline."""
        from litellm.llms.zai.chat.transformation import _flatten_content_parts

        content = [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]
        assert _flatten_content_parts(content) == "Line 1\nLine 2"

    def test_flatten_content_parts_helper_empty_list(self):
        """Empty list returns empty string."""
        from litellm.llms.zai.chat.transformation import _flatten_content_parts

        assert _flatten_content_parts([]) == ""

    def test_flatten_content_parts_helper_string_passthrough(self):
        """Plain string passes through unchanged."""
        from litellm.llms.zai.chat.transformation import _flatten_content_parts

        assert _flatten_content_parts("already a string") == "already a string"

    def test_flatten_content_parts_helper_none(self):
        """None passes through unchanged."""
        from litellm.llms.zai.chat.transformation import _flatten_content_parts

        assert _flatten_content_parts(None) is None
