"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""
import json
import math

import pytest
import respx
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionAssistantMessageParam, \
    ChatCompletionToolMessageParam, ChatCompletionMessageFunctionToolCallParam
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition

import litellm
from litellm import completion

ZENMUX_API_KEY="sk-ai-v1-xxx"

def test_get_llm_provider_zenmux():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    model, provider, api_key, api_base = get_llm_provider("zenmux/anthropic/claude-opus-4")
    assert model == "anthropic/claude-opus-4"
    assert provider == "zenmux"
    assert api_base == "https://zenmux.ai/api/v1"


def test_zenmux_stream_completion(monkeypatch):
    monkeypatch.setenv("ZENMUX_API_KEY", ZENMUX_API_KEY)
    litellm.disable_aiohttp_transport = True

    response = litellm.completion(
        model="zenmux/openai/gpt-5.1",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
        stream=True,
        stream_options={
            "include_usage":True
        }
    )
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)
        usage = getattr(chunk,"usage",None)
    print(usage)


def test_zenmux_completion_provider(monkeypatch):
    monkeypatch.setenv("ZENMUX_API_KEY", ZENMUX_API_KEY)
    litellm.disable_aiohttp_transport = True

    response = litellm.completion(
        model="zenmux/anthropic/claude-opus-4",
        messages=[{"role": "user", "content": "what is the meaning of life"}],
        stream=True,
        reasoning_effort="high",
        provider={
            "routing": {
                "type": "order",
                "providers": [
                    "amazon-bedrock"
                ]
            }
        }
    )
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)

def test_zenmux_completion_routing(monkeypatch):
    monkeypatch.setenv("ZENMUX_API_KEY", ZENMUX_API_KEY)
    litellm.disable_aiohttp_transport = True

    response = litellm.completion(
        model="zenmux/mistralai/mistral-large-2512",
        messages=[{"role": "user", "content": "计算1+1"}],
        stream=True,
        reasoning_effort="low",
        stream_options={
            "include_usage":True
        },
        provider={
            "routing": {
                "type": "order",
                "providers": [
                    "mistral"
                ]
            }
        }
    )
    for chunk in response:
        print(chunk)
        print(chunk.choices[0].delta.content, end="", flush=True)

@pytest.mark.asyncio
async def test_zenmux_async_completion(monkeypatch):
    monkeypatch.setenv("ZENMUX_API_KEY", ZENMUX_API_KEY)
    litellm.disable_aiohttp_transport = True

    response = await litellm.acompletion(
        model="zenmux/anthropic/claude-opus-4",
        messages=[{"role": "user", "content": "Hello，what can you do for me"}],
        max_tokens=20,
    )
    print(response)

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 20


weather_tool = ChatCompletionFunctionToolParam(
    function=FunctionDefinition(
        name="search_city_weather",
        description="get the weather of a city of a specified day",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "city name"},
                "date": {"type": "string", "description": "yyyy-mm-dd datetime"}},
            "required": ["city", "date"],
            "additionalProperties": False
        }
    ),
    type="function",
)
weather_response = """{"city":"Tokyo","date":"2025-08-15","week":"1","dayweather":"cloudy","nightweather":"cloudy","daywind":"southwest","nightwind":"southeast","daypower":"1-3","nightpower":"1-3","daytemp_float":"35.0","nighttemp_float":"28.0"}"""


def test_zenmux_completion_tools(monkeypatch):
    model = "zenmux/openai/gpt-5.1"
    monkeypatch.setenv("ZENMUX_API_KEY", ZENMUX_API_KEY)
    litellm.disable_aiohttp_transport = True

    messages = []
    messages.append({"role": "user", "content": "what is the weather of Tokyo of 2024.8.15"})
    response = litellm.completion(
        model=model,
        messages=messages,
        stream=True,
        # reasoning_effort="medium",
        tools=[weather_tool],
    )
    response_chunks=[]
    for chunk in response:
        response_chunks.append(chunk)
    response_content = "".join([chunk.choices[0].delta.content for chunk in response_chunks if chunk.choices[0].delta and getattr(chunk.choices[0].delta,"content",None)])
    reasoning_content = "".join([chunk.choices[0].delta.reasoning_content for chunk in response_chunks if chunk.choices[0].delta and getattr(chunk.choices[0].delta,"reasoning_content",None)])
    print("response_content:",response_content)
    print("reasoning_content:",reasoning_content)

    tool_calls = dict()
    for chunk in response_chunks:
        if chunk.choices and chunk.choices[0].delta.tool_calls is not None and len(
                chunk.choices[0].delta.tool_calls) > 0:
            for tool_call_delta in chunk.choices[0].delta.tool_calls:
                current_index = tool_call_delta.index
                if current_index in tool_calls:
                    tool_calls[current_index].function.arguments += tool_call_delta.function.arguments
                else:
                    if tool_call_delta.function.arguments is None:
                        tool_call_delta.function.arguments = ""
                    tool_calls[current_index] = tool_call_delta
    tool_call_list = [tool_calls[index] for index in sorted(tool_calls.keys())]
    print(f"tool_calls:{tool_call_list}")

    assistant_message=ChatCompletionAssistantMessageParam(
        role="assistant",
        tool_calls=[ChatCompletionMessageFunctionToolCallParam(
            function=Function(name=tool_calls[0].function.name, arguments=tool_calls[0].function.arguments),
            id=tool_calls[0].id,
            type="function")
        ],
        content=response_content
    )
    messages.append(assistant_message)
    tool_call_response = ChatCompletionToolMessageParam(
        role="tool",
        tool_call_id=tool_calls[0].id,
        content=weather_response,
    )
    messages.append(tool_call_response)

    response = litellm.completion(
        model=model,
        messages=messages,
        stream=True,
        # reasoning_effort="medium",
        tools=[weather_tool]
    )
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)
