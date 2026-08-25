import json
import logging
from collections.abc import Mapping, Sequence
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
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
    result = _transform_response(
        {"role": "assistant", "content": "4", "reasoning": "2+2 equals 4"}
    )

    assert result.choices[0].message.content == "4"
    assert result.choices[0].message.reasoning_content == "2+2 equals 4"


def test_transform_response_preserves_reasoning_content_field():
    result = _transform_response(
        {"role": "assistant", "content": "4", "reasoning_content": "adding 2 and 2"}
    )

    assert result.choices[0].message.reasoning_content == "adding 2 and 2"


def test_streaming_chunk_maps_delta_reasoning_to_reasoning_content():
    iterator = TogetherAIChatConfig().get_model_response_iterator(
        streaming_response=iter(()), sync_stream=True
    )
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
    iterator = TogetherAIChatConfig().get_model_response_iterator(
        streaming_response=iter(()), sync_stream=True
    )

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

    config = ProviderConfigManager.get_provider_chat_config(
        model=REASONING_MODEL, provider=LlmProviders.TOGETHER_AI
    )

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
