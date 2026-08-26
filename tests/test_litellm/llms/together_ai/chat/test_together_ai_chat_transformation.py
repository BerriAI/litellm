import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
)
from litellm.llms.together_ai.chat.transformation import TogetherAIChatConfig
from litellm.types.utils import LlmProviders, ModelResponse

TOOL_CALLING_MODEL = "openai/gpt-oss-20b"
REASONING_MODEL = "deepseek-ai/DeepSeek-V3.1"
UNMAPPED_MODEL = "example-org/brand-new-model"
NO_TOOLS_MODEL = "example-org/no-tools-model"
NO_SCHEMA_MODEL = "example-org/no-schema-model"

TOOL_PARAMS = ("tools", "tool_choice", "function_call")

WEATHER_TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

VOICE_NOTE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["title", "summary"],
    "additionalProperties": False,
}
JSON_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "voice_note", "schema": VOICE_NOTE_SCHEMA, "strict": True},
}
REGEX_RESPONSE_FORMAT = {"type": "regex", "pattern": "(positive|neutral|negative)"}


@pytest.fixture(autouse=True)
def force_local_model_cost(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url))


@pytest.fixture(autouse=True)
def isolate_together_api_base_env(monkeypatch):
    monkeypatch.delenv("TOGETHER_AI_API_BASE", raising=False)


@pytest.fixture
def registry_disables_function_calling(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        f"together_ai/{NO_TOOLS_MODEL}",
        {"litellm_provider": "together_ai", "mode": "chat", "supports_function_calling": False},
    )


@pytest.fixture
def registry_disables_response_schema(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        f"together_ai/{NO_SCHEMA_MODEL}",
        {"litellm_provider": "together_ai", "mode": "chat", "supports_response_schema": False},
    )


@pytest.fixture
def together_warning_log(caplog):
    from litellm._logging import verbose_logger

    verbose_logger.addHandler(caplog.handler)
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        yield caplog
    verbose_logger.removeHandler(caplog.handler)


def test_supported_params_tool_calling_model():
    supported = TogetherAIChatConfig().get_supported_openai_params(model=TOOL_CALLING_MODEL)

    for param in (*TOOL_PARAMS, "response_format"):
        assert param in supported


def test_supported_params_unmapped_model_keeps_tool_params():
    supported = TogetherAIChatConfig().get_supported_openai_params(model=UNMAPPED_MODEL)

    for param in TOOL_PARAMS:
        assert param in supported
    assert "response_format" in supported
    assert "stream" in supported
    assert "temperature" in supported


def test_supported_params_no_tools_model_keeps_tool_params(registry_disables_function_calling):
    supported = TogetherAIChatConfig().get_supported_openai_params(model=NO_TOOLS_MODEL)

    for param in TOOL_PARAMS:
        assert param in supported
    assert "response_format" in supported


def test_map_openai_params_tool_calling_model_passes_tools():
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"tools": WEATHER_TOOLS, "tool_choice": "auto"},
        optional_params={},
        model=TOOL_CALLING_MODEL,
        drop_params=False,
    )

    assert mapped["tools"] == WEATHER_TOOLS
    assert mapped["tool_choice"] == "auto"


@pytest.mark.parametrize("drop_params", [False, True])
def test_map_openai_params_unmapped_model_passes_tools_through(drop_params, together_warning_log):
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"tools": WEATHER_TOOLS, "tool_choice": "required"},
        optional_params={},
        model=UNMAPPED_MODEL,
        drop_params=drop_params,
    )

    assert mapped["tools"] == WEATHER_TOOLS
    assert mapped["tool_choice"] == "required"
    assert UNMAPPED_MODEL in together_warning_log.text
    assert "passing tools, tool_choice through" in together_warning_log.text


def test_map_openai_params_no_tools_model_drops_tools_with_warning(
    registry_disables_function_calling, together_warning_log
):
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"tools": WEATHER_TOOLS, "temperature": 0.5},
        optional_params={},
        model=NO_TOOLS_MODEL,
        drop_params=True,
    )

    assert "tools" not in mapped
    assert mapped["temperature"] == 0.5
    assert NO_TOOLS_MODEL in together_warning_log.text
    assert "dropping tools" in together_warning_log.text


def test_map_openai_params_no_tools_model_raises_without_drop_params(registry_disables_function_calling):
    with pytest.raises(UnsupportedParamsError, match="does not support parameters"):
        TogetherAIChatConfig().map_openai_params(
            non_default_params={"tools": WEATHER_TOOLS},
            optional_params={},
            model=NO_TOOLS_MODEL,
            drop_params=False,
        )


def test_map_openai_params_reasoning_model_passes_sampling_params():
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"temperature": 0.2, "max_tokens": 512},
        optional_params={},
        model=REASONING_MODEL,
        drop_params=False,
    )

    assert mapped["temperature"] == 0.2
    assert mapped["max_tokens"] == 512


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "text"},
        {"type": "json_object"},
        {"type": "json_object", "schema": VOICE_NOTE_SCHEMA},
        JSON_SCHEMA_RESPONSE_FORMAT,
        REGEX_RESPONSE_FORMAT,
    ],
)
def test_map_openai_params_schema_model_passes_response_format_through(response_format):
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"response_format": response_format},
        optional_params={},
        model=TOOL_CALLING_MODEL,
        drop_params=False,
    )

    assert mapped["response_format"] == response_format


@pytest.mark.parametrize("drop_params", [False, True])
def test_map_openai_params_unmapped_model_passes_response_format_through(drop_params, together_warning_log):
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"response_format": JSON_SCHEMA_RESPONSE_FORMAT},
        optional_params={},
        model=UNMAPPED_MODEL,
        drop_params=drop_params,
    )

    assert mapped["response_format"] == JSON_SCHEMA_RESPONSE_FORMAT
    assert UNMAPPED_MODEL in together_warning_log.text
    assert "passing response_format through" in together_warning_log.text


def test_map_openai_params_no_schema_model_drops_response_format_with_warning(
    registry_disables_response_schema, together_warning_log
):
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"response_format": JSON_SCHEMA_RESPONSE_FORMAT, "temperature": 0.5},
        optional_params={},
        model=NO_SCHEMA_MODEL,
        drop_params=True,
    )

    assert "response_format" not in mapped
    assert mapped["temperature"] == 0.5
    assert NO_SCHEMA_MODEL in together_warning_log.text
    assert "dropping response_format" in together_warning_log.text


def test_map_openai_params_no_schema_model_raises_without_drop_params(registry_disables_response_schema):
    with pytest.raises(UnsupportedParamsError, match="response_format"):
        TogetherAIChatConfig().map_openai_params(
            non_default_params={"response_format": JSON_SCHEMA_RESPONSE_FORMAT},
            optional_params={},
            model=NO_SCHEMA_MODEL,
            drop_params=False,
        )


def _transform_response(message: dict) -> ModelResponse:
    raw_response_json = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": REASONING_MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.return_value = raw_response_json
    mock_response.text = json.dumps(raw_response_json)
    mock_response.headers = {}
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.post_call = MagicMock()
    logging_obj.model_call_details = {}

    return TogetherAIChatConfig().transform_response(
        model=REASONING_MODEL,
        raw_response=mock_response,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "What is 2+2?"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key="test-key",
        json_mode=False,
    )


def test_transform_response_maps_reasoning_to_reasoning_content():
    result = _transform_response({"role": "assistant", "content": "4", "reasoning": "2+2 equals 4"})

    assert result.choices[0].message.content == "4"
    assert result.choices[0].message.reasoning_content == "2+2 equals 4"


def test_transform_response_preserves_reasoning_content_field():
    result = _transform_response({"role": "assistant", "content": "4", "reasoning_content": "adding 2 and 2"})

    assert result.choices[0].message.reasoning_content == "adding 2 and 2"


def test_streaming_chunk_maps_delta_reasoning_to_reasoning_content():
    iterator = TogetherAIChatConfig().get_model_response_iterator(streaming_response=iter(()), sync_stream=True)
    assert isinstance(iterator, OpenAIChatCompletionStreamingHandler)

    parsed = iterator.chunk_parser(
        {
            "id": "chunk-1",
            "created": 1234567890,
            "model": REASONING_MODEL,
            "choices": [{"index": 0, "delta": {"reasoning": "thinking about 2+2"}}],
        }
    )

    assert parsed.choices[0]["delta"]["reasoning_content"] == "thinking about 2+2"


def test_streaming_chunk_preserves_tool_call_index_and_id():
    iterator = TogetherAIChatConfig().get_model_response_iterator(streaming_response=iter(()), sync_stream=True)

    def parse_tool_call_chunk(tool_call: dict):
        parsed = iterator.chunk_parser(
            {
                "id": "chunk-1",
                "created": 1234567890,
                "model": TOOL_CALLING_MODEL,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}],
            }
        )
        return parsed.choices[0]["delta"]["tool_calls"][0]

    opener = parse_tool_call_chunk(
        {
            "index": 1,
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": ""},
        }
    )
    continuation = parse_tool_call_chunk(
        {"index": 1, "id": "", "type": "function", "function": {"arguments": '{"city": "San'}}
    )

    assert opener["index"] == 1
    assert opener["id"] == "call_abc123"
    assert opener["function"]["name"] == "get_weather"
    assert continuation["index"] == 1
    assert continuation["function"]["arguments"] == '{"city": "San'


REPLAYED_ASSISTANT_MESSAGE = {
    "role": "assistant",
    "content": "The digit sum is 11.",
    "reasoning_content": "The secret number is 47. 4 + 7 = 11.",
    "thinking_blocks": [{"type": "thinking", "thinking": "The secret number is 47.", "signature": ""}],
    "provider_specific_fields": {"thinking_blocks": [{"type": "thinking", "thinking": "The secret number is 47."}]},
}

PRESERVED_THINKING_MESSAGES = [
    {"role": "user", "content": "Pick a secret two-digit number and tell me only its digit sum."},
    REPLAYED_ASSISTANT_MESSAGE,
    {"role": "user", "content": "What was the secret number?"},
]


def _assert_internal_fields_stripped_reasoning_kept(transformed_messages: Sequence[Mapping[str, object]]):
    assistant_message = transformed_messages[1]
    assert assistant_message["reasoning_content"] == REPLAYED_ASSISTANT_MESSAGE["reasoning_content"]
    assert "thinking_blocks" not in assistant_message
    assert "provider_specific_fields" not in assistant_message
    assert assistant_message["content"] == REPLAYED_ASSISTANT_MESSAGE["content"]
    assert transformed_messages[0] == PRESERVED_THINKING_MESSAGES[0]
    assert transformed_messages[2] == PRESERVED_THINKING_MESSAGES[2]


def test_transform_request_keeps_reasoning_content_strips_internal_fields():
    request = TogetherAIChatConfig().transform_request(
        model=REASONING_MODEL,
        messages=[dict(message) for message in PRESERVED_THINKING_MESSAGES],
        optional_params={},
        litellm_params={"custom_llm_provider": "together_ai"},
        headers={},
    )

    _assert_internal_fields_stripped_reasoning_kept(request["messages"])


async def test_async_transform_request_keeps_reasoning_content_strips_internal_fields():
    request = await TogetherAIChatConfig().async_transform_request(
        model=REASONING_MODEL,
        messages=[dict(message) for message in PRESERVED_THINKING_MESSAGES],
        optional_params={},
        litellm_params={"custom_llm_provider": "together_ai"},
        headers={},
    )

    _assert_internal_fields_stripped_reasoning_kept(request["messages"])


def test_completion_sends_chat_template_kwargs_and_preserved_reasoning():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    captured_requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-together-preserved",
                "object": "chat.completion",
                "created": 1234567890,
                "model": REASONING_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "47"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))

    litellm.completion(
        model=f"together_ai/{REASONING_MODEL}",
        messages=[dict(message) for message in PRESERVED_THINKING_MESSAGES],
        chat_template_kwargs={"clear_thinking": False},
        api_key="fake-key",
        client=client,
    )

    request_body = json.loads(captured_requests[0].content)
    assert request_body["chat_template_kwargs"] == {"clear_thinking": False}
    assert "extra_body" not in request_body
    _assert_internal_fields_stripped_reasoning_kept(request_body["messages"])


def test_together_ai_config_alias_points_at_chat_config():
    assert litellm.TogetherAIConfig is litellm.TogetherAIChatConfig
    config = litellm.TogetherAIConfig(max_tokens=10)
    assert isinstance(config, TogetherAIChatConfig)


def test_provider_config_manager_returns_together_chat_config():
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(model=REASONING_MODEL, provider=LlmProviders.TOGETHER_AI)

    assert isinstance(config, TogetherAIChatConfig)


def test_completion_routes_through_together_chat_config():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    captured_requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-together",
                "object": "chat.completion",
                "created": 1234567890,
                "model": REASONING_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "4",
                            "reasoning": "2+2 equals 4",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))

    response = litellm.completion(
        model=f"together_ai/{REASONING_MODEL}",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        api_key="fake-key",
        client=client,
    )

    request = captured_requests[0]
    assert str(request.url) == "https://api.together.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer fake-key"
    assert json.loads(request.content)["model"] == REASONING_MODEL
    assert response.choices[0].message.content == "4"
    assert response.choices[0].message.reasoning_content == "2+2 equals 4"


def test_completion_unmapped_model_sends_tools_to_together():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    captured_requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-together-tools",
                "object": "chat.completion",
                "created": 1234567890,
                "model": UNMAPPED_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "San Francisco"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))

    response = litellm.completion(
        model=f"together_ai/{UNMAPPED_MODEL}",
        messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
        tools=WEATHER_TOOLS,
        tool_choice="auto",
        api_key="fake-key",
        client=client,
    )

    request_body = json.loads(captured_requests[0].content)
    assert request_body["tools"] == WEATHER_TOOLS
    assert request_body["tool_choice"] == "auto"
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"city": "San Francisco"}


def _capture_completion_request(model: str, **completion_kwargs) -> dict:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    captured_requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-together-structured",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"title": "t", "summary": "s"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    litellm.completion(
        model=f"together_ai/{model}",
        messages=[{"role": "user", "content": "Summarize with a title and summary."}],
        api_key="fake-key",
        client=client,
        **completion_kwargs,
    )
    return json.loads(captured_requests[0].content)


def test_completion_unmapped_model_sends_json_schema_to_together():
    request_body = _capture_completion_request(
        UNMAPPED_MODEL, response_format=JSON_SCHEMA_RESPONSE_FORMAT, drop_params=True
    )

    assert request_body["response_format"] == JSON_SCHEMA_RESPONSE_FORMAT


def test_completion_pydantic_response_format_sends_json_schema_to_together():
    from pydantic import BaseModel

    class VoiceNote(BaseModel):
        title: str
        summary: str

    request_body = _capture_completion_request(TOOL_CALLING_MODEL, response_format=VoiceNote)

    sent = request_body["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["name"] == "VoiceNote"
    assert sent["json_schema"]["strict"] is True
    assert sent["json_schema"]["schema"]["required"] == ["title", "summary"]


def test_completion_regex_response_format_sends_pattern_to_together():
    request_body = _capture_completion_request(TOOL_CALLING_MODEL, response_format=REGEX_RESPONSE_FORMAT)

    assert request_body["response_format"] == REGEX_RESPONSE_FORMAT


TOGETHER_CHAT_URL = "https://api.together.ai/v1/chat/completions"

WEATHER_AND_TIME_TOOLS = [
    *WEATHER_TOOLS,
    {"type": "function", "function": {"name": "get_time", "parameters": {}}},
]

ANTHROPIC_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _chat_completion(message: Mapping[str, object], finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-together",
        "object": "chat.completion",
        "created": 1234567890,
        "model": UNMAPPED_MODEL,
        "choices": [{"index": 0, "message": dict(message), "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _chunk(delta: Mapping[str, object], finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-together-stream",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": UNMAPPED_MODEL,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }


def _sse(*events: Mapping[str, object]) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"


def _sse_response(*events: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(200, content=_sse(*events), headers={"Content-Type": "text/event-stream"})


def _sync_client(captured_requests: list[httpx.Request], response: httpx.Response) -> HTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return response

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))


async def _async_client(captured_requests: list[httpx.Request], response: httpx.Response) -> AsyncHTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return response

    handler = AsyncHTTPHandler()
    await handler.close()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return handler


PARALLEL_TOOL_CALL_STREAM = (
    _chunk({"role": "assistant", "reasoning": "Need weather "}),
    _chunk({"reasoning": "and time."}),
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "San'}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' Francisco"}'}}]}),
    _chunk(
        {
            "tool_calls": [
                {"index": 1, "id": "call_time", "type": "function", "function": {"name": "get_time", "arguments": ""}}
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"tz": "PST"}'}}]}, finish_reason="tool_calls"),
)


def test_streaming_completion_rebuilds_reasoning_and_parallel_tool_calls():
    captured_requests: list[httpx.Request] = []
    client = _sync_client(captured_requests, _sse_response(*PARALLEL_TOOL_CALL_STREAM))

    chunks = list(
        litellm.completion(
            model=f"together_ai/{UNMAPPED_MODEL}",
            messages=[{"role": "user", "content": "Weather and time in San Francisco?"}],
            tools=WEATHER_AND_TIME_TOOLS,
            stream=True,
            api_key="fake-key",
            client=client,
        )
    )

    request_body = json.loads(captured_requests[0].content)
    assert str(captured_requests[0].url) == TOGETHER_CHAT_URL
    assert request_body["stream"] is True
    assert request_body["tools"] == WEATHER_AND_TIME_TOOLS

    streamed_reasoning = "".join(getattr(chunk.choices[0].delta, "reasoning_content", None) or "" for chunk in chunks)
    assert streamed_reasoning == "Need weather and time."

    rebuilt = litellm.stream_chunk_builder(chunks)
    message = rebuilt.choices[0].message
    assert message.reasoning_content == "Need weather and time."
    assert rebuilt.choices[0].finish_reason == "tool_calls"
    calls = {call.id: call for call in message.tool_calls}
    assert calls["call_weather"].function.name == "get_weather"
    assert json.loads(calls["call_weather"].function.arguments) == {"city": "San Francisco"}
    assert calls["call_time"].function.name == "get_time"
    assert json.loads(calls["call_time"].function.arguments) == {"tz": "PST"}


async def test_async_streaming_completion_strips_internal_fields_and_streams_reasoning():
    captured_requests: list[httpx.Request] = []
    client = await _async_client(
        captured_requests,
        _sse_response(
            _chunk({"role": "assistant", "reasoning": "Recalling 47."}),
            _chunk({"content": "47"}, finish_reason="stop"),
        ),
    )

    try:
        stream = await litellm.acompletion(
            model=f"together_ai/{REASONING_MODEL}",
            messages=[dict(message) for message in PRESERVED_THINKING_MESSAGES],
            chat_template_kwargs={"clear_thinking": False},
            stream=True,
            api_key="fake-key",
            client=client,
        )
        chunks = [chunk async for chunk in stream]
    finally:
        await client.client.aclose()

    request_body = json.loads(captured_requests[0].content)
    assert str(captured_requests[0].url) == TOGETHER_CHAT_URL
    assert request_body["chat_template_kwargs"] == {"clear_thinking": False}
    _assert_internal_fields_stripped_reasoning_kept(request_body["messages"])

    rebuilt = litellm.stream_chunk_builder(chunks)
    assert rebuilt.choices[0].message.reasoning_content == "Recalling 47."
    assert rebuilt.choices[0].message.content == "47"


@pytest.mark.parametrize("api_base", ["https://api.together.ai/v1", "https://api.together.xyz/v1"])
def test_completion_bare_model_with_together_api_base_uses_together_config(api_base):
    captured_requests: list[httpx.Request] = []
    client = _sync_client(
        captured_requests,
        httpx.Response(200, json=_chat_completion({"role": "assistant", "content": "4", "reasoning": "2+2"})),
    )

    response = litellm.completion(
        model=UNMAPPED_MODEL,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        api_base=api_base,
        api_key="fake-key",
        client=client,
    )

    assert str(captured_requests[0].url) == f"{api_base}/chat/completions"
    assert captured_requests[0].headers["authorization"] == "Bearer fake-key"
    assert response._hidden_params["custom_llm_provider"] == "together_ai"
    assert response.choices[0].message.reasoning_content == "2+2"


def test_completion_honors_together_ai_api_base_env(monkeypatch):
    monkeypatch.setenv("TOGETHER_AI_API_BASE", "https://together.internal.example/v1")
    captured_requests: list[httpx.Request] = []
    client = _sync_client(
        captured_requests,
        httpx.Response(200, json=_chat_completion({"role": "assistant", "content": "4"})),
    )

    litellm.completion(
        model=f"together_ai/{REASONING_MODEL}",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        api_key="fake-key",
        client=client,
    )

    assert str(captured_requests[0].url) == "https://together.internal.example/v1/chat/completions"


def test_responses_api_sends_tools_and_maps_reasoning_and_function_call():
    captured_requests: list[httpx.Request] = []
    client = _sync_client(
        captured_requests,
        httpx.Response(
            200,
            json=_chat_completion(
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "Need the weather tool.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "San Francisco"}'},
                        }
                    ],
                },
                finish_reason="tool_calls",
            ),
        ),
    )

    response = litellm.responses(
        model=f"together_ai/{UNMAPPED_MODEL}",
        input="What is the weather in San Francisco?",
        tools=[{"type": "function", "name": "get_weather", "parameters": {}}],
        api_key="fake-key",
        client=client,
    )

    request_body = json.loads(captured_requests[0].content)
    assert str(captured_requests[0].url) == TOGETHER_CHAT_URL
    assert [tool["function"]["name"] for tool in request_body["tools"]] == ["get_weather"]
    outputs = {item.type: item for item in response.output}
    assert outputs["reasoning"].content[0].text == "Need the weather tool."
    assert outputs["function_call"].name == "get_weather"
    assert json.loads(outputs["function_call"].arguments) == {"city": "San Francisco"}


ANTHROPIC_TOOL_LOOP_MESSAGES = [
    {"role": "user", "content": "What is the weather in San Francisco?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "I should call get_weather.", "signature": ""},
            {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "San Francisco"}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny, 18C"}],
    },
]


def test_anthropic_messages_replays_tool_loop_and_maps_reasoning_to_thinking_block():
    captured_requests: list[httpx.Request] = []
    client = _sync_client(
        captured_requests,
        httpx.Response(
            200,
            json=_chat_completion({"role": "assistant", "content": "Sunny in SF.", "reasoning": "Tool said sunny."}),
        ),
    )

    response = litellm.anthropic.messages.create(
        model=f"together_ai/{UNMAPPED_MODEL}",
        max_tokens=100,
        messages=[dict(message) for message in ANTHROPIC_TOOL_LOOP_MESSAGES],
        tools=[ANTHROPIC_WEATHER_TOOL],
        api_key="fake-key",
        client=client,
    )

    request_body = json.loads(captured_requests[0].content)
    assert str(captured_requests[0].url) == TOGETHER_CHAT_URL
    assert [tool["function"]["name"] for tool in request_body["tools"]] == ["get_weather"]
    assistant_turn = request_body["messages"][1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["reasoning_content"] == "I should call get_weather."
    assert "thinking_blocks" not in assistant_turn
    replayed_call = assistant_turn["tool_calls"][0]
    assert replayed_call["id"] == "toolu_01"
    assert replayed_call["function"]["name"] == "get_weather"
    assert json.loads(replayed_call["function"]["arguments"]) == {"city": "San Francisco"}
    tool_turn = request_body["messages"][2]
    assert tool_turn["role"] == "tool"
    assert tool_turn["tool_call_id"] == "toolu_01"
    assert tool_turn["content"] == "Sunny, 18C"

    blocks = {block["type"]: block for block in response["content"]}
    assert blocks["thinking"]["thinking"] == "Tool said sunny."
    assert blocks["text"]["text"] == "Sunny in SF."
    assert response["stop_reason"] == "end_turn"


def _anthropic_sse_events(stream: Iterator[bytes]) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for raw in stream
        for line in raw.decode().splitlines()
        if line.startswith("data: ")
    ]


def test_anthropic_messages_streams_together_tool_call_as_input_json_delta():
    captured_requests: list[httpx.Request] = []
    client = _sync_client(captured_requests, _sse_response(*PARALLEL_TOOL_CALL_STREAM))

    events = _anthropic_sse_events(
        litellm.anthropic.messages.create(
            model=f"together_ai/{UNMAPPED_MODEL}",
            max_tokens=100,
            messages=[{"role": "user", "content": "Weather and time in San Francisco?"}],
            tools=[ANTHROPIC_WEATHER_TOOL, {"name": "get_time", "input_schema": {"type": "object"}}],
            stream=True,
            api_key="fake-key",
            client=client,
        )
    )

    assert json.loads(captured_requests[0].content)["stream"] is True
    tool_starts = {
        event["index"]: event["content_block"]
        for event in events
        if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
    }
    input_json_deltas = [
        event
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "input_json_delta"
    ]
    tool_inputs = {
        block["name"]: json.loads(
            "".join(delta["delta"]["partial_json"] for delta in input_json_deltas if delta["index"] == index)
        )
        for index, block in tool_starts.items()
    }
    assert {block["id"] for block in tool_starts.values()} == {"call_weather", "call_time"}
    assert tool_inputs == {"get_weather": {"city": "San Francisco"}, "get_time": {"tz": "PST"}}
    thinking_text = "".join(
        event["delta"]["thinking"]
        for event in events
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "thinking_delta"
    )
    assert thinking_text == "Need weather and time."
    assert [event["delta"]["stop_reason"] for event in events if event["type"] == "message_delta"] == ["tool_use"]
