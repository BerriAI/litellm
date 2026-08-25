import json
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
)
from litellm.llms.together_ai.chat.transformation import TogetherAIChatConfig
from litellm.types.utils import LlmProviders, ModelResponse

TOOL_CALLING_MODEL = "openai/gpt-oss-20b"
REASONING_MODEL = "deepseek-ai/DeepSeek-V3.1"
PLAIN_MODEL = "Qwen/Qwen3-235B-A22B-fp8-tput"
UNMAPPED_MODEL = "MiniMaxAI/MiniMax-M3"

FUNCTION_CALLING_PARAMS = ("tools", "tool_choice", "function_call", "response_format")


@pytest.fixture(autouse=True)
def force_local_model_cost(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url))


def test_supported_params_tool_calling_model():
    supported = TogetherAIChatConfig().get_supported_openai_params(model=TOOL_CALLING_MODEL)

    for param in FUNCTION_CALLING_PARAMS:
        assert param in supported


def test_supported_params_plain_model():
    supported = TogetherAIChatConfig().get_supported_openai_params(model=PLAIN_MODEL)

    for param in FUNCTION_CALLING_PARAMS:
        assert param not in supported
    assert "temperature" in supported
    assert "max_tokens" in supported


def test_supported_params_unmapped_model_treated_as_plain():
    supported = TogetherAIChatConfig().get_supported_openai_params(model=UNMAPPED_MODEL)

    for param in FUNCTION_CALLING_PARAMS:
        assert param not in supported
    assert "stream" in supported


def test_map_openai_params_tool_calling_model_passes_tools():
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

    mapped = TogetherAIChatConfig().map_openai_params(
        non_default_params={"tools": tools, "tool_choice": "auto"},
        optional_params={},
        model=TOOL_CALLING_MODEL,
        drop_params=False,
    )

    assert mapped["tools"] == tools
    assert mapped["tool_choice"] == "auto"


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
