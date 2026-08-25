import json
import logging
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
PLAIN_MODEL = "Qwen/Qwen3-235B-A22B-fp8-tput"
UNMAPPED_MODEL = "example-org/brand-new-model"
NO_TOOLS_MODEL = "example-org/no-tools-model"
ADJUSTABLE_REASONING_MODEL = "openai/gpt-oss-120b"
HYBRID_REASONING_MODEL = "Qwen/Qwen3.5-9B"
HIGH_MAX_REASONING_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
REGISTRY_FLAGGED_REASONING_MODEL = "zai-org/GLM-4.6"
NON_REASONING_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

TOOL_PARAMS = ("tools", "tool_choice", "function_call")

WEATHER_TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


def _map_reasoning_effort(model: str, effort: str) -> dict:
    return TogetherAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": effort},
        optional_params={},
        model=model,
        drop_params=False,
    )


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
    assert "response_format" not in supported
    assert "stream" in supported
    assert "temperature" in supported


def test_supported_params_no_tools_model_keeps_tool_params(registry_disables_function_calling):
    supported = TogetherAIChatConfig().get_supported_openai_params(model=NO_TOOLS_MODEL)

    for param in TOOL_PARAMS:
        assert param in supported
    assert "response_format" not in supported


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


def test_map_openai_params_drops_text_response_format():
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"response_format": {"type": "text"}, "temperature": 0.5},
        optional_params={},
        model=REASONING_MODEL,
        drop_params=False,
    )

    assert "response_format" not in mapped
    assert mapped["temperature"] == 0.5


def test_map_openai_params_keeps_json_response_format():
    response_format = {"type": "json_object"}

    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"response_format": response_format},
        optional_params={},
        model=TOOL_CALLING_MODEL,
        drop_params=False,
    )

    assert mapped["response_format"] == response_format


@pytest.mark.parametrize(
    "model",
    [ADJUSTABLE_REASONING_MODEL, HYBRID_REASONING_MODEL, HIGH_MAX_REASONING_MODEL, REGISTRY_FLAGGED_REASONING_MODEL],
)
def test_supported_params_includes_reasoning_effort_for_reasoning_models(model):
    supported = TogetherAIChatConfig().get_supported_openai_params(model=model)

    assert "reasoning_effort" in supported


@pytest.mark.parametrize("model", [NON_REASONING_MODEL, PLAIN_MODEL])
def test_supported_params_excludes_reasoning_effort_for_non_reasoning_models(model):
    supported = TogetherAIChatConfig().get_supported_openai_params(model=model)

    assert "reasoning_effort" not in supported


@pytest.mark.parametrize(
    "effort, expected",
    [("low", "low"), ("medium", "medium"), ("high", "high"), ("minimal", "low"), ("xhigh", "high"), ("max", "high")],
)
def test_adjustable_model_translates_reasoning_effort(effort, expected):
    mapped = _map_reasoning_effort(ADJUSTABLE_REASONING_MODEL, effort)

    assert mapped["reasoning_effort"] == expected
    assert "reasoning" not in mapped


def test_adjustable_model_cannot_disable_reasoning_so_none_becomes_low():
    mapped = _map_reasoning_effort(ADJUSTABLE_REASONING_MODEL, "none")

    assert mapped["reasoning_effort"] == "low"
    assert "reasoning" not in mapped


@pytest.mark.parametrize(
    "effort, expected",
    [("low", "low"), ("medium", "medium"), ("high", "high"), ("minimal", "low"), ("xhigh", "high"), ("max", "high")],
)
def test_hybrid_model_translates_reasoning_effort(effort, expected):
    mapped = _map_reasoning_effort(HYBRID_REASONING_MODEL, effort)

    assert mapped["reasoning_effort"] == expected
    assert "reasoning" not in mapped


@pytest.mark.parametrize("model", [HYBRID_REASONING_MODEL, HIGH_MAX_REASONING_MODEL, REGISTRY_FLAGGED_REASONING_MODEL])
def test_reasoning_effort_none_becomes_reasoning_toggle(model):
    mapped = _map_reasoning_effort(model, "none")

    assert mapped["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in mapped


def test_reasoning_effort_none_does_not_clobber_user_reasoning():
    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={"reasoning": {"enabled": True}},
        model=HYBRID_REASONING_MODEL,
        drop_params=False,
    )

    assert mapped["reasoning"] == {"enabled": True}
    assert "reasoning_effort" not in mapped


@pytest.mark.parametrize(
    "effort, expected",
    [("minimal", "high"), ("low", "high"), ("medium", "high"), ("high", "max"), ("xhigh", "max"), ("max", "max")],
)
def test_deepseek_v4_pro_remaps_to_high_max(effort, expected):
    mapped = _map_reasoning_effort(HIGH_MAX_REASONING_MODEL, effort)

    assert mapped["reasoning_effort"] == expected


def test_deepseek_v4_pro_dated_variant_remaps_via_prefix():
    mapped = _map_reasoning_effort(f"{HIGH_MAX_REASONING_MODEL}-0813", "low")

    assert mapped["reasoning_effort"] == "high"


@pytest.mark.parametrize("model", [ADJUSTABLE_REASONING_MODEL, HYBRID_REASONING_MODEL, HIGH_MAX_REASONING_MODEL])
def test_reasoning_effort_default_is_dropped(model):
    mapped = _map_reasoning_effort(model, "default")

    assert "reasoning_effort" not in mapped
    assert "reasoning" not in mapped


def test_get_optional_params_translates_reasoning_effort_for_together():
    optional_params = litellm.get_optional_params(
        model=ADJUSTABLE_REASONING_MODEL,
        custom_llm_provider="together_ai",
        reasoning_effort="max",
    )

    assert optional_params["reasoning_effort"] == "high"


def test_get_optional_params_rejects_reasoning_effort_for_non_reasoning_together_model():
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.get_optional_params(
            model=NON_REASONING_MODEL,
            custom_llm_provider="together_ai",
            reasoning_effort="low",
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
