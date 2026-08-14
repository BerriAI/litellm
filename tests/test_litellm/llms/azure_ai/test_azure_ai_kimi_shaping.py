"""Tests for Azure AI Foundry / FW-Kimi K3 request shaping."""

from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig


config = AzureAIStudioConfig()
responses_config = OpenAIResponsesAPIConfig()


def test_is_kimi_reasoning_model_matches_fw_and_native_ids():
    assert config._is_kimi_reasoning_model("FW-Kimi-K3") is True
    assert config._is_kimi_reasoning_model("azure_ai/FW-Kimi-K3") is True
    assert config._is_kimi_k3_model("FW-Kimi-K3") is True
    assert config._is_kimi_reasoning_model("kimi-k2.5") is True
    assert config._is_kimi_reasoning_model("gpt-4o") is False


def test_map_openai_params_drops_fixed_sampling_and_medium_effort():
    optional = config.map_openai_params(
        non_default_params={
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 2,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "reasoning_effort": "medium",
        },
        optional_params={
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 2,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "reasoning_effort": "medium",
        },
        model="FW-Kimi-K3",
        drop_params=True,
    )
    for key in (
        "temperature",
        "top_p",
        "n",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
    ):
        assert key not in optional


def test_map_openai_params_keeps_valid_reasoning_effort():
    optional = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="FW-Kimi-K3",
        drop_params=True,
    )
    assert optional.get("reasoning_effort") == "high"


def test_fill_reasoning_content_placeholder_for_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
    ]
    out = config.fill_reasoning_content(messages)
    assert out[0]["reasoning_content"] == " "


def test_fill_reasoning_content_strips_thinking_blocks_into_reasoning_content():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [
                {"type": "thinking", "thinking": "plan the tool call", "signature": ""}
            ],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
    ]
    out = config.fill_reasoning_content(messages)
    assert "thinking_blocks" not in out[0]
    assert out[0]["reasoning_content"] == "plan the tool call"


def test_shape_kimi_responses_inserts_reasoning_before_function_call():
    req = {
        "model": "FW-Kimi-K3",
        "temperature": 0.5,
        "reasoning": {"effort": "medium"},
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "thinking..."}],
            },
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
        ],
    }
    shaped = responses_config._shape_kimi_responses_request(req)
    assert "temperature" not in shaped
    assert "reasoning" not in shaped
    types = [x["type"] for x in shaped["input"]]
    assert types == ["message", "reasoning", "function_call"]
