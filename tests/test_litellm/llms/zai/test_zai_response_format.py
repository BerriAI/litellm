"""
Tests for response_format support on the Z.AI (GLM) provider.

GLM ignores response_format but honours a forced tool call, so litellm translates a
schema into one. Regression cover for BerriAI/litellm#37720.
"""

import json

import httpx
import pytest

import litellm
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.llms.zai.chat.transformation import ZAIChatConfig
from litellm.utils import get_optional_params

ZAI_URL = "https://api.z.ai/api/paas/v4/chat/completions"

SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
    "additionalProperties": False,
}
JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "ents", "strict": True, "schema": SCHEMA},
}
TOOL_ARGUMENTS = json.dumps({"entities": ["Alice", "service"]})
MESSAGES = [{"role": "user", "content": "Extract entities from: Alice deployed a service."}]


@pytest.fixture
def tool_call_reply():
    """Z.AI reply to a forced json_tool_call: the schema-valid payload is the arguments."""
    return {
        "id": "chatcmpl-zai-rf",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "glm-5-turbo",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": RESPONSE_FORMAT_TOOL_NAME,
                                "arguments": TOOL_ARGUMENTS,
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_response_format_is_a_supported_param():
    """response_format must be allowlisted, or get_optional_params drops it before mapping."""
    assert "response_format" in ZAIChatConfig().get_supported_openai_params(model="glm-5-turbo")


def test_json_schema_becomes_a_forced_tool_call():
    optional_params = get_optional_params(
        model="glm-5-turbo",
        custom_llm_provider="zai",
        response_format=JSON_SCHEMA_FORMAT,
    )

    assert "response_format" not in optional_params, "GLM ignores response_format; it must be translated"
    assert optional_params["tools"] == [
        {
            "type": "function",
            "function": {"name": RESPONSE_FORMAT_TOOL_NAME, "parameters": SCHEMA},
        }
    ]
    assert optional_params["tool_choice"] == {
        "type": "function",
        "function": {"name": RESPONSE_FORMAT_TOOL_NAME},
    }
    assert optional_params["json_mode"] is True


def test_json_schema_is_translated_even_when_drop_params_is_set():
    """The silent-drop path in the bug report: drop_params must no longer swallow the schema."""
    optional_params = get_optional_params(
        model="glm-5-turbo",
        custom_llm_provider="zai",
        response_format=JSON_SCHEMA_FORMAT,
        drop_params=True,
    )

    assert optional_params["tools"][0]["function"]["parameters"] == SCHEMA
    assert optional_params["json_mode"] is True


def test_json_object_is_forwarded_unchanged():
    """No schema to translate. GLM returns arbitrary-shaped but valid JSON, so forward it."""
    optional_params = get_optional_params(
        model="glm-5-turbo",
        custom_llm_provider="zai",
        response_format={"type": "json_object"},
    )

    assert optional_params["response_format"] == {"type": "json_object"}
    assert "json_mode" not in optional_params
    assert "tools" not in optional_params


def _caller_tool():
    return {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
    }


def test_caller_supplied_tools_are_left_alone():
    """
    Forcing json_tool_call would hijack a caller that is genuinely using tools, and an
    unforced one would leave nothing constraining the model. Neither is translated.
    """
    optional_params = get_optional_params(
        model="glm-5-turbo",
        custom_llm_provider="zai",
        tools=[_caller_tool()],
        response_format=JSON_SCHEMA_FORMAT,
    )

    assert optional_params["tools"] == [_caller_tool()]
    assert "tool_choice" not in optional_params
    assert optional_params["response_format"] == JSON_SCHEMA_FORMAT
    assert "json_mode" not in optional_params


def test_caller_tools_list_is_not_mutated_across_calls():
    """
    _map_openai_params aliases the caller's list into optional_params, so appending the
    generated tool would corrupt a module-level TOOLS list on every turn of an agent loop.
    """
    caller_tools = [_caller_tool()]

    for _ in range(3):
        get_optional_params(
            model="glm-5-turbo",
            custom_llm_provider="zai",
            tools=caller_tools,
            response_format=JSON_SCHEMA_FORMAT,
        )

    assert [tool["function"]["name"] for tool in caller_tools] == ["get_weather"]


def test_base_helper_does_not_mutate_the_tools_it_is_given():
    """Direct cover for the shared helper, which azure and fireworks_ai also call."""
    caller_tools = [_caller_tool()]
    optional_params = ZAIChatConfig()._add_response_format_to_tools(
        optional_params={"tools": caller_tools},
        value=JSON_SCHEMA_FORMAT,
        is_response_format_supported=False,
    )

    assert len(optional_params["tools"]) == 2
    assert [tool["function"]["name"] for tool in caller_tools] == ["get_weather"]


def test_falsy_response_schema_is_not_silently_dropped():
    """
    The helper extracts response_schema by key presence, so a present-but-empty one makes
    it a no-op. Translating anyway would pop response_format and lose the schema entirely.
    """
    response_format = {
        "type": "json_schema",
        "response_schema": {},
        "json_schema": {"name": "ents", "schema": SCHEMA},
    }

    optional_params = get_optional_params(
        model="glm-5-turbo",
        custom_llm_provider="zai",
        response_format=response_format,
    )

    assert optional_params["response_format"] == response_format
    assert "tools" not in optional_params


def test_completion_unwraps_the_tool_call_into_content(respx_mock, tool_call_reply, monkeypatch):
    """End-to-end: json_mode must not reach the SDK, and the arguments become the content."""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post(ZAI_URL).mock(return_value=httpx.Response(200, json=tool_call_reply))

    response = litellm.completion(
        model="zai/glm-5-turbo",
        messages=MESSAGES,
        response_format=JSON_SCHEMA_FORMAT,
        max_tokens=100,
    )

    assert json.loads(response.choices[0].message.content) == {"entities": ["Alice", "service"]}
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].finish_reason == "stop"

    request_body = json.loads(route.calls[0].request.content)
    assert "json_mode" not in request_body, "json_mode is internal; the OpenAI SDK rejects it"
    assert request_body["tool_choice"]["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME


@pytest.mark.asyncio
async def test_acompletion_unwraps_the_tool_call_into_content(respx_mock, tool_call_reply, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post(ZAI_URL).mock(return_value=httpx.Response(200, json=tool_call_reply))

    response = await litellm.acompletion(
        model="zai/glm-5-turbo",
        messages=MESSAGES,
        response_format=JSON_SCHEMA_FORMAT,
        max_tokens=100,
    )

    assert json.loads(response.choices[0].message.content) == {"entities": ["Alice", "service"]}
    assert response.choices[0].finish_reason == "stop"

    request_body = json.loads(route.calls[0].request.content)
    assert "json_mode" not in request_body


def test_streaming_is_left_untranslated(respx_mock, monkeypatch):
    """
    The unwrap back into message.content only runs on non-streaming responses, so a
    forced tool call here would hand the caller a stream of unusable tool_calls.
    """
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    chunk = {
        "id": "chatcmpl-zai-rf",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "glm-5-turbo",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }
    route = respx_mock.post(ZAI_URL).mock(
        return_value=httpx.Response(
            200,
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )

    chunks = list(
        litellm.completion(
            model="zai/glm-5-turbo",
            messages=MESSAGES,
            response_format=JSON_SCHEMA_FORMAT,
            max_tokens=100,
            stream=True,
        )
    )

    assert [chunk.choices[0].delta.content for chunk in chunks][0] == "hi"
    request_body = json.loads(route.calls[0].request.content)
    assert "json_mode" not in request_body
    assert "tool_choice" not in request_body, "a forced tool call cannot be unwrapped mid-stream"
    assert request_body["response_format"] == JSON_SCHEMA_FORMAT
    assert request_body["stream"] is True
