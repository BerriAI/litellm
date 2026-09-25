import json
from typing import Final

import httpx
import respx

import litellm


def test_completion_accepts_tool_call_without_content(respx_mock: respx.MockRouter):
    upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-tool-call",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "my-custom-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
            },
        )
    )

    response: Final = litellm.completion(
        model="databricks/my-custom-model",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        api_base="https://example.databricks.test/serving-endpoints",
        api_key="fake-databricks-api-key",
        num_retries=0,
    )

    assert upstream.call_count == 1
    assert response.choices[0].finish_reason == "tool_calls"
    message: Final = response.choices[0].message
    assert message.content is None
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "call_weather"
    assert message.tool_calls[0].function.name == "get_weather"
    assert message.tool_calls[0].function.arguments == '{"city":"Paris"}'


def test_completion_merges_leading_system_and_developer_messages_for_chat_template_models(
    respx_mock: respx.MockRouter,
):
    upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "my-custom-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            },
        )
    )

    response: Final = litellm.completion(
        model="databricks/my-custom-model",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "developer", "content": "Skills: none."},
            {"role": "user", "content": "Hello"},
        ],
        api_base="https://example.databricks.test/serving-endpoints",
        api_key="fake-databricks-api-key",
        num_retries=0,
    )

    assert upstream.call_count == 1
    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["messages"] == [
        {"role": "system", "content": "You are terse.\n\nSkills: none."},
        {"role": "user", "content": "Hello"},
    ]
    assert response.choices[0].message.content == "Answer"


def test_completion_merges_system_messages_when_one_has_empty_content(respx_mock: respx.MockRouter):
    upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "my-custom-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            },
        )
    )

    litellm.completion(
        model="databricks/my-custom-model",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ],
        api_base="https://example.databricks.test/serving-endpoints",
        api_key="fake-databricks-api-key",
        num_retries=0,
    )

    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
    ]
