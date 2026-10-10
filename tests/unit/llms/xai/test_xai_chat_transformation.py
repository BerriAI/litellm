import json
import os
from typing import Final
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.xai.chat.transformation import (
    XAI_API_BASE,
    XAIChatCompletionStreamingHandler,
    XAIChatConfig,
)
from litellm.llms.xai.cost_calculator import cost_per_token
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ModelResponse,
    Usage,
)


class TestXAIReasoningTokenFolding:
    """``fold_reasoning_tokens_into_completion`` re-aligns xAI Usage to the OpenAI invariant."""

    @staticmethod
    def _make_response(
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        reasoning_tokens: int = 0,
    ) -> ModelResponse:
        details = CompletionTokensDetailsWrapper(reasoning_tokens=reasoning_tokens) if reasoning_tokens else None
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=details,
        )
        response = ModelResponse()
        setattr(response, "usage", usage)
        return response

    def test_should_fold_when_total_explained_by_reasoning_gap(self):
        # xAI live shape: 14 + 10 + 312 == 336.
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=336,
            reasoning_tokens=312,
        )

        XAIChatConfig.fold_reasoning_tokens_into_completion(response)

        usage = response.usage
        assert usage.completion_tokens == 322
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_should_not_fold_when_already_normalised(self):
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=322,
            total_tokens=336,
            reasoning_tokens=312,
        )

        XAIChatConfig.fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 322

    def test_should_skip_when_no_reasoning_tokens(self):
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=24,
            reasoning_tokens=0,
        )

        XAIChatConfig.fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 10

    def test_should_skip_when_gap_does_not_match_reasoning(self):
        # Refuse to fold if xAI accounting changes (gap != reasoning_tokens).
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=999,
            reasoning_tokens=312,
        )

        XAIChatConfig.fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 10
        assert response.usage.total_tokens == 999


def test_max_completion_tokens_is_accepted_and_mapped_to_max_tokens() -> None:
    optional_params = litellm.get_optional_params(
        model="grok-4.20",
        custom_llm_provider="xai",
        max_completion_tokens=64,
    )
    assert optional_params["max_tokens"] == 64, optional_params
    assert "max_completion_tokens" not in optional_params, optional_params


class TestXAIParallelToolCalls:
    """Test suite for XAI parallel tool calls functionality."""

    def test_get_supported_openai_params_includes_parallel_tool_calls(self):
        """Test that parallel_tool_calls is in supported parameters."""
        config = XAIChatConfig()
        supported_params = config.get_supported_openai_params("xai/grok-4.20")
        assert "parallel_tool_calls" in supported_params

    def test_transform_request_preserves_parallel_tool_calls(self):
        """Test that transform_request preserves parallel_tool_calls parameter."""
        config = XAIChatConfig()

        messages = [{"role": "user", "content": "What's the weather like?"}]
        optional_params = {"parallel_tool_calls": True}

        result = config.transform_request(
            model="xai/grok-4.20",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result.get("parallel_tool_calls") is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


class TestXAIChatWebSearchOptions:
    """XAI answers /chat/completions requests carrying web_search_options with a 410 (Live Search retired)"""

    def test_transform_request_drops_web_search_options(self):
        config = XAIChatConfig()

        result = config.transform_request(
            model="xai/grok-4.6",
            messages=[{"role": "user", "content": "newest litellm version?"}],
            optional_params={"web_search_options": {"search_context_size": "medium"}, "temperature": 0.5},
            litellm_params={},
            headers={},
        )

        assert "web_search_options" not in result
        assert result["temperature"] == 0.5


class TestXAIUsageNormalization:
    def test_preserves_reasoning_tokens_in_total_usage(self):
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=200)

        XAIChatConfig.normalize_openai_compatible_usage_totals(usage)

        assert usage.total_tokens == 200

    def test_preserves_reasoning_tokens_in_streaming_usage(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 200}

        XAIChatConfig.normalize_openai_compatible_usage_totals(usage)

        assert usage["total_tokens"] == 200


@pytest.mark.respx(assert_all_called=True)
def test_completion_strips_message_names_from_request(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(f"{XAI_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-xai",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "grok-4.20-beta-latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )
    )

    response: Final = litellm.completion(
        model="xai/grok-4.20-beta-latest",
        messages=[
            {"role": "system", "content": "Be concise", "name": "system_prompt"},
            {"role": "user", "content": "Say OK", "name": "caller"},
            {"role": "assistant", "content": "OK", "name": "responder"},
            {"role": "user", "content": "Again", "name": "caller"},
        ],
        api_key="xai-test-key",
        api_base=XAI_API_BASE,
    )

    request_body: Final = json.loads(route.calls.last.request.content)
    assert request_body["messages"] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Say OK", "name": "caller"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "Again", "name": "caller"},
    ]
    assert response.choices[0].message.content == "OK"


@pytest.mark.respx(assert_all_called=True)
def test_streaming_include_usage_surfaces_xai_final_usage_chunk(respx_mock: respx.MockRouter) -> None:
    frames: Final = (
        {
            "id": "chatcmpl-xai-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "grok-4.20-beta-latest",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-xai-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "grok-4.20-beta-latest",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-xai-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "grok-4.20-beta-latest",
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )
    sse_body: Final = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    respx_mock.post(f"{XAI_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )
    )

    chunks: Final = list(
        litellm.completion(
            model="xai/grok-4.20-beta-latest",
            messages=[{"role": "user", "content": "Say OK"}],
            stream=True,
            stream_options={"include_usage": True},
            api_key="xai-test-key",
            api_base=XAI_API_BASE,
        )
    )

    usage_chunks: Final = [chunk for chunk in chunks if getattr(chunk, "usage", None) is not None]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "OK"
    assert len(usage_chunks) == 1
    assert (
        usage_chunks[0].usage.prompt_tokens,
        usage_chunks[0].usage.completion_tokens,
        usage_chunks[0].usage.total_tokens,
    ) == (2, 1, 3)


class TestXAIChatWebSearchBilling:
    _TOOL_DETAILS = {
        "web_search_calls": 3,
        "x_search_calls": 0,
        "code_interpreter_calls": 0,
        "file_search_calls": 0,
        "mcp_calls": 0,
        "document_search_calls": 0,
    }

    @staticmethod
    def _response_with_usage() -> ModelResponse:
        response = ModelResponse(model="grok-4")
        setattr(
            response,
            "usage",
            Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )
        return response

    def test_enhance_copies_details_and_mirrors_web_search_requests(self):
        response = self._response_with_usage()

        XAIChatConfig()._enhance_usage_with_xai_web_search_fields(
            response,
            {"usage": {"server_side_tool_usage_details": self._TOOL_DETAILS}},
        )

        usage = response.usage
        assert getattr(usage, "server_side_tool_usage_details") == self._TOOL_DETAILS
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.web_search_requests == 3

    def test_enhance_noop_without_details(self):
        response = self._response_with_usage()

        XAIChatConfig()._enhance_usage_with_xai_web_search_fields(response, {"usage": {"prompt_tokens": 100}})

        assert response.usage.prompt_tokens_details is None
        assert getattr(response.usage, "server_side_tool_usage_details", None) is None

    @pytest.mark.parametrize(
        ("tool_details", "expected_web_search_requests"),
        [(_TOOL_DETAILS, 3), (None, None)],
    )
    def test_transform_response_reads_tool_usage_details_from_the_response_body(
        self,
        tool_details: dict[str, int] | None,
        expected_web_search_requests: int | None,
    ):
        raw_response: Final = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-xai",
                "object": "chat.completion",
                "created": 0,
                "model": "grok-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "server_side_tool_usage_details": tool_details,
                },
            },
        )

        response: Final = XAIChatConfig().transform_response(
            model="grok-4",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert getattr(response.usage, "server_side_tool_usage_details", None) == tool_details
        assert (
            getattr(response.usage.prompt_tokens_details, "web_search_requests", None) == expected_web_search_requests
        )
        assert response.usage.total_tokens == 120


class TestXAIReportedCost:
    """xAI reports what it charged; the transformation moves it to where litellm bills from.

    ``cost`` is the field litellm already carries a provider stated cost in, so restating
    ``cost_in_usd_ticks`` there is what lets ``llms/xai/cost_calculator.py`` bill the
    reported figure. At 10^10 ticks to the dollar, 37756000 ticks is $0.0037756.
    """

    @staticmethod
    def _transformed_usage(usage: dict) -> Usage:
        raw_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-xai",
                "object": "chat.completion",
                "created": 0,
                "model": "grok-4-latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )

        response = XAIChatConfig().transform_response(
            model="grok-4-latest",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        return response.usage

    def test_reported_cost_reaches_the_cost_calculator(self):
        usage = self._transformed_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "cost_in_usd_ticks": 37756000,
            }
        )

        assert usage.cost == 0.0037756
        assert cost_per_token(model="grok-4-latest", usage=usage) == (0.0, 0.0037756)

    def test_usage_without_a_reported_cost_is_left_alone(self):
        usage = self._transformed_usage({"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300})

        assert getattr(usage, "cost", None) is None

    def test_negative_reported_cost_is_not_carried(self):
        """A caller who can set api_base must not be able to report negative spend."""
        usage = self._transformed_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "cost_in_usd_ticks": -37756000,
            }
        )

        assert getattr(usage, "cost", None) is None

    def test_streamed_reported_cost_survives_chunk_aggregation(self):
        """Streamed spend only matches if the conversion happens on the chunk.

        Chunk aggregation rebuilds usage from the fields it models plus ``cost``, so a
        chunk still carrying only ``cost_in_usd_ticks`` loses the reported amount.
        """
        handler = XAIChatCompletionStreamingHandler(streaming_response=iter([]), sync_stream=True)

        parsed = handler.chunk_parser(
            {
                "id": "chatcmpl-xai",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "grok-4-latest",
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                    "cost_in_usd_ticks": 37756000,
                },
            }
        )

        assert parsed.usage.cost == 0.0037756

        assembled = litellm.stream_chunk_builder(chunks=[parsed])
        assert assembled.usage.cost == 0.0037756
        assert cost_per_token(model="grok-4-latest", usage=assembled.usage) == (
            0.0,
            0.0037756,
        )


def test_xai_chat_config_get_openai_compatible_provider_info():
    config = XAIChatConfig()
    api_base, api_key = config.get_openai_compatible_provider_info(api_base=None, api_key=None)
    assert api_base == XAI_API_BASE
    assert api_key == os.environ.get("XAI_API_KEY")
    custom_api_key = "test_api_key"
    api_base, api_key = config.get_openai_compatible_provider_info(api_base=None, api_key=custom_api_key)
    assert api_base == XAI_API_BASE
    assert api_key == custom_api_key
    with patch.dict("os.environ", {"XAI_API_BASE": "https://env.x.ai/v1", "XAI_API_KEY": "env_api_key"}):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://env.x.ai/v1"
        assert api_key == "env_api_key"


def test_xai_chat_config_map_openai_params():
    """
    XAI is OpenAI compatible*

    Does not support all OpenAI parameters:
    - max_completion_tokens -> max_tokens

    """
    config = XAIChatConfig()
    non_default_params = {
        "max_completion_tokens": 100,
        "frequency_penalty": 0.5,
        "logit_bias": {"50256": -100},
        "logprobs": 5,
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "xai/grok-beta",
        "n": 2,
        "presence_penalty": 0.2,
        "response_format": {"type": "json_object"},
        "seed": 42,
        "stop": ["END"],
        "stream": True,
        "stream_options": {},
        "temperature": 0.7,
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "top_logprobs": 3,
        "top_p": 0.9,
        "user": "test_user",
        "unsupported_param": "value",
    }
    optional_params = {}
    model = "xai/grok-beta"
    result = config.map_openai_params(non_default_params, optional_params, model)
    assert result["max_tokens"] == 100
    assert result["frequency_penalty"] == 0.5
    assert result["logit_bias"] == {"50256": -100}
    assert result["logprobs"] == 5
    assert result["n"] == 2
    assert result["presence_penalty"] == 0.2
    assert result["response_format"] == {"type": "json_object"}
    assert result["seed"] == 42
    assert result["stop"] == ["END"]
    assert result["stream"] is True
    assert result["stream_options"] == {}
    assert result["temperature"] == 0.7
    assert result["tool_choice"] == "auto"
    assert result["tools"] == [{"type": "function", "function": {"name": "get_weather"}}]
    assert result["top_logprobs"] == 3
    assert result["top_p"] == 0.9
    assert result["user"] == "test_user"
    assert "unsupported_param" not in result


def test_xai_check_for_stop_in_supported_params():
    supported_params = XAIChatConfig().get_supported_openai_params(model="xai/grok-3-mini")
    assert "stop" not in supported_params


@pytest.mark.parametrize("model", ["xai/grok-4", "xai/grok-4-0709"])
def test_xai_grok_4_stop_not_supported(model):
    """
    Test that grok-4 models do not support the stop parameter

    Issue: https://github.com/BerriAI/litellm/issues/12635
    """
    supported_params = XAIChatConfig().get_supported_openai_params(model=model)
    assert "stop" not in supported_params


@pytest.mark.parametrize(
    "model", ["xai/grok-4", "xai/grok-4-0709", "xai/grok-4-latest", "xai/grok-code-fast", "xai/grok-code-fast-1"]
)
def test_xai_grok_4_frequency_penalty_not_supported(model):
    """
    Test that grok-4 models do not support the frequency_penalty parameter
    """
    supported_params = XAIChatConfig().get_supported_openai_params(model=model)
    assert "frequency_penalty" not in supported_params
