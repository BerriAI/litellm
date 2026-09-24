"""Tests for the Sail (sailresearch.com) JSON-configured provider."""

import json

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

SAIL_BASE_URL = "https://api.sailresearch.com/v1"
SAIL_CHAT_COMPLETIONS = f"{SAIL_BASE_URL}/chat/completions"
SAIL_RESPONSES = f"{SAIL_BASE_URL}/responses"

MODEL = "sail/zai-org/GLM-5.3"


def _chat_completion_payload() -> dict:
    return {
        "id": "chatcmpl-sail",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "zai-org/GLM-5.3",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "sail response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


def _chat_completion_stream() -> str:
    chunks = [
        {
            "id": "chatcmpl-sail-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "zai-org/GLM-5.3",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "sail"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-sail-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "zai-org/GLM-5.3",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def _responses_payload() -> dict:
    return {
        "id": "resp_sail",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": "zai-org/GLM-5.3",
        "output": [],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        "error": None,
    }


@pytest.fixture(autouse=True)
def _sail_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("SAIL_API_KEY", "sk-sail-test")
    monkeypatch.delenv("SAIL_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


class TestSailRequestShape:
    @pytest.mark.respx()
    def test_sail_chat_completions_url_and_auth(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert len(respx_mock.calls) == 1
        request = respx_mock.calls[0].request
        assert request.url == SAIL_CHAT_COMPLETIONS
        assert request.headers["Authorization"] == "Bearer sk-sail-test"

        _, provider, _, _ = get_llm_provider(
            model=MODEL, custom_llm_provider=None, api_base=None, api_key=None
        )
        assert provider == "sail"

    @pytest.mark.respx()
    def test_sail_api_base_env_overrides_url(
        self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SAIL_API_BASE", "https://sail.internal.example/v2")
        route = respx_mock.post("https://sail.internal.example/v2/chat/completions").respond(
            json=_chat_completion_payload()
        )

        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "hi"}])

        assert route.called

    @pytest.mark.respx()
    def test_sail_explicit_api_base_trailing_slash_no_double_slash(self, respx_mock: respx.Router):
        route = respx_mock.post("https://custom.sail.example/v1/chat/completions").respond(
            json=_chat_completion_payload()
        )

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://custom.sail.example/v1/",
        )

        assert route.called

    @pytest.mark.respx()
    def test_sail_max_tokens_sent_as_max_completion_tokens(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["max_completion_tokens"] == 100
        assert "max_tokens" not in body

    @pytest.mark.respx()
    def test_sail_no_metadata_key_when_caller_passes_none(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(model=MODEL, messages=[{"role": "user", "content": "hi"}])

        body = json.loads(respx_mock.calls[0].request.content)
        assert "metadata" not in body

    @pytest.mark.respx()
    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    def test_sail_completion_window_via_extra_body(self, respx_mock: respx.Router, stream: bool):
        if stream:
            respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(
                content=_chat_completion_stream(),
                headers={"content-type": "text/event-stream"},
            )
        else:
            respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            stream=stream,
            extra_body={"metadata": {"completion_window": "balanced"}},
        )
        if stream:
            list(response)

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"completion_window": "balanced"}

    @pytest.mark.respx()
    def test_sail_tools_survive_to_request_body(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "what is the weather"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_aresponses_posts_metadata_and_background(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_RESPONSES).respond(json=_responses_payload())

        await litellm.aresponses(
            model=MODEL,
            input="hi",
            metadata={"completion_window": "flex"},
            background=True,
        )

        assert len(respx_mock.calls) == 1
        request = respx_mock.calls[0].request
        assert request.url == SAIL_RESPONSES
        body = json.loads(request.content)
        assert body["metadata"] == {"completion_window": "flex"}
        assert body["background"] is True


class TestSailCostTracking:
    def test_cached_tokens_billed_at_sail_cache_read_rate(self, monkeypatch: pytest.MonkeyPatch):
        rates = litellm.model_cost[MODEL]
        prompt_tokens = 1000
        cached_tokens = 600
        completion_tokens = 200

        response = litellm.ModelResponse(
            model="zai-org/GLM-5.3",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
            ),
        )

        cost = litellm.completion_cost(
            completion_response=response,
            model=MODEL,
            custom_llm_provider="sail",
        )

        expected = (
            (prompt_tokens - cached_tokens) * rates["input_cost_per_token"]
            + cached_tokens * rates["cache_read_input_token_cost"]
            + completion_tokens * rates["output_cost_per_token"]
        )
        assert cost == pytest.approx(expected)
