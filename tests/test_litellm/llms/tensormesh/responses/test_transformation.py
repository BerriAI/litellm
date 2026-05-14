"""
Tests for Tensormesh Responses API transformation.
"""

import json
import os
import sys
from typing import Any

import httpx

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)


class _FakeLogging:
    def post_call(self, *args: Any, **kwargs: Any) -> None:
        pass


def _make_http_handler(handler):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_tensormesh_responses_config():
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="tensormesh",
        model="openai/gpt-oss-120b",
    )

    assert config is not None
    assert config.custom_llm_provider == "tensormesh"
    assert (
        config.get_complete_url(api_base=None, litellm_params={})
        == "https://serverless.tensormesh.ai/v1/responses"
    )
    assert config.supports_native_websocket() is False


def test_tensormesh_responses_builds_sdk_compatible_request(monkeypatch):
    import litellm

    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_tensormesh_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "max_output_tokens": json_body.get("max_output_tokens"),
                "model": json_body["model"],
                "output": [
                    {
                        "type": "message",
                        "id": "msg_tensormesh_test",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "previous_response_id": None,
                "reasoning": {"effort": None, "summary": None},
                "store": True,
                "temperature": 1.0,
                "text": {"format": {"type": "text"}},
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
                "truncation": "disabled",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "user": None,
                "metadata": {},
            },
            request=request,
        )

    client = _make_http_handler(handler)
    try:
        response = litellm.responses(
            model="tensormesh/openai/gpt-oss-120b",
            input="Say ok.",
            api_key="tm-responses-test-key",
            max_output_tokens=7,
            client=client,
        )
    finally:
        client.close()

    assert response.output[0].content[0].text == "ok"
    assert captured["url"] == "https://serverless.tensormesh.ai/v1/responses"
    assert captured["headers"]["authorization"] == "Bearer tm-responses-test-key"
    assert captured["body"]["model"] == "openai/gpt-oss-120b"
    assert captured["body"]["input"] == "Say ok."
    assert captured["body"]["max_output_tokens"] == 7
    assert response.usage.input_tokens == 1
    assert response.usage.output_tokens == 1
    assert response.usage.total_tokens == 2


def test_tensormesh_responses_forwards_structured_output_and_tools(monkeypatch):
    import litellm

    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)
    captured: dict[str, Any] = {}
    text_format = {
        "format": {
            "type": "json_schema",
            "name": "weather_answer",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": True,
        }
    }
    weather_tool = {
        "type": "function",
        "name": "weather",
        "description": "Return weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_tensormesh_tools_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_tensormesh_test",
                        "call_id": "call_tensormesh_test",
                        "name": "weather",
                        "arguments": json.dumps({"city": "Bangkok"}),
                        "status": "completed",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=request,
        )

    client = _make_http_handler(handler)
    try:
        litellm.responses(
            model="tensormesh/MiniMaxAI/MiniMax-M2.7",
            input="What is the weather in Bangkok?",
            api_key="tm-responses-tools-test-key",
            text=text_format,
            tools=[weather_tool],
            tool_choice="required",
            max_output_tokens=64,
            client=client,
        )
    finally:
        client.close()

    assert captured["body"]["model"] == "MiniMaxAI/MiniMax-M2.7"
    assert captured["body"]["text"] == text_format
    assert captured["body"]["tools"] == [weather_tool]
    assert captured["body"]["tool_choice"] == "required"


def test_tensormesh_responses_normalizes_usage_before_model_construct():
    from litellm.llms.tensormesh.responses.transformation import (
        TensormeshResponsesConfig,
    )
    from litellm.types.llms.openai import ResponseAPIUsage

    raw_response = httpx.Response(
        200,
        json={
            "id": "resp_tensormesh_construct_test",
            "object": "response",
            "created_at": 1,
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 1},
            },
        },
        request=httpx.Request("POST", "https://serverless.tensormesh.ai/v1/responses"),
    )

    response = TensormeshResponsesConfig().transform_response_api_response(
        model="MiniMaxAI/MiniMax-M2.7",
        raw_response=raw_response,
        logging_obj=_FakeLogging(),
    )

    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.input_tokens == 2
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 5
    assert response.usage.input_tokens_details is not None
    assert response.usage.input_tokens_details.cached_tokens == 1


def test_tensormesh_responses_streaming_normalizes_usage_without_mutating_to_model():
    from litellm.llms.tensormesh.responses.transformation import (
        TensormeshResponsesConfig,
    )
    from litellm.types.llms.openai import ResponseAPIUsage

    parsed_chunk = {
        "type": "response.completed",
        "response": {
            "id": "resp_tensormesh_stream_test",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "MiniMaxAI/MiniMax-M2.7",
            "output": [],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 1},
            },
        },
    }

    response = TensormeshResponsesConfig().transform_streaming_response(
        model="MiniMaxAI/MiniMax-M2.7",
        parsed_chunk=parsed_chunk,
        logging_obj=_FakeLogging(),
    )

    usage = parsed_chunk["response"]["usage"]
    assert isinstance(usage, dict)
    assert not isinstance(usage, ResponseAPIUsage)
    assert usage["input_tokens"] == 2
    assert usage["output_tokens"] == 3
    assert usage["total_tokens"] == 5
    assert usage["input_tokens_details"]["cached_tokens"] == 1
    assert response.response.usage.input_tokens == 2
    assert response.response.usage.output_tokens == 3
