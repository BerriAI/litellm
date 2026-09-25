"""Tests for the Sail (sailresearch.com) JSON-configured provider."""

import io
import json
import wave
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import (
    create_config_class,
    create_responses_config_class,
)
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

SAIL_BASE_URL = "https://api.sailresearch.com/v1"
SAIL_CHAT_COMPLETIONS = f"{SAIL_BASE_URL}/chat/completions"
SAIL_RESPONSES = f"{SAIL_BASE_URL}/responses"
SAIL_MESSAGES = f"{SAIL_BASE_URL}/messages"

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


def _messages_payload() -> dict:
    return {
        "id": "msg_sail",
        "type": "message",
        "role": "assistant",
        "model": "zai-org/GLM-5.3-Flash",
        "content": [{"type": "text", "text": "sail response"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 2},
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

        _, provider, _, _ = get_llm_provider(model=MODEL, custom_llm_provider=None, api_base=None, api_key=None)
        assert provider == "sail"

    @pytest.mark.respx()
    def test_sail_api_base_env_overrides_url(self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch):
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

    @pytest.mark.respx(assert_all_called=False)
    def test_sail_transcription_rejected_without_hitting_sail(self, respx_mock: respx.Router):
        route = respx_mock.post(f"{SAIL_BASE_URL}/audio/transcriptions")

        with pytest.raises(litellm.BadRequestError) as exc_info:
            litellm.transcription(model=MODEL, file=_wav_file())

        assert exc_info.value.status_code == 400
        assert str(exc_info.value) == (
            f"litellm.BadRequestError: sail does not support audio transcription. Model: {MODEL.split('/', 1)[1]}"
        )
        assert not route.called
        assert respx_mock.calls.call_count == 0

    @pytest.mark.respx(assert_all_called=False)
    @pytest.mark.asyncio
    async def test_sail_atranscription_rejected_without_hitting_sail(self, respx_mock: respx.Router):
        route = respx_mock.post(f"{SAIL_BASE_URL}/audio/transcriptions")

        with pytest.raises(litellm.BadRequestError) as exc_info:
            await litellm.atranscription(model=MODEL, file=_wav_file())

        assert exc_info.value.status_code == 400
        assert "sail does not support audio transcription" in str(exc_info.value)
        assert not route.called
        assert respx_mock.calls.call_count == 0

    @pytest.mark.respx()
    def test_sail_unsupported_params_dropped_with_drop_params(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            stop=["x"],
            seed=1,
            frequency_penalty=0.5,
            drop_params=True,
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert "stop" not in body
        assert "seed" not in body
        assert "frequency_penalty" not in body
        assert body["model"] == "zai-org/GLM-5.3"
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.respx(assert_all_called=False)
    def test_sail_unsupported_params_raise_without_drop_params(self, respx_mock: respx.Router):
        route = respx_mock.post(SAIL_CHAT_COMPLETIONS)

        with pytest.raises(litellm.UnsupportedParamsError):
            litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stop=["x"],
            )

        assert not route.called

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_anthropic_messages_posts_to_messages_endpoint(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model="sail/zai-org/GLM-5.3-Flash",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
        )

        assert len(respx_mock.calls) == 1
        request = respx_mock.calls[0].request
        assert request.url == SAIL_MESSAGES
        assert request.headers["Authorization"] == "Bearer sk-sail-test"
        body = json.loads(request.content)
        assert body["model"] == "zai-org/GLM-5.3-Flash"
        assert body["max_tokens"] == 50

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_anthropic_messages_honors_api_base_env(
        self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SAIL_API_BASE", "https://sail.internal.example/v2")
        route = respx_mock.post("https://sail.internal.example/v2/v1/messages").respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model="sail/zai-org/GLM-5.3-Flash",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
        )

        assert route.called

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_messages_maps_service_tier_to_completion_window(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            service_tier="flex",
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body == {
            "model": MODEL.split("/", 1)[1],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50,
            "metadata": {"completion_window": "flex"},
            "stream": False,
        }

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_messages_caller_window_wins_over_tier(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            service_tier="balanced",
            metadata={"user_id": "u"},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"user_id": "u", "completion_window": "balanced"}
        assert "service_tier" not in body

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_messages_extra_body_window_wins_and_is_stripped(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            service_tier="flex",
            extra_body={"metadata": {"completion_window": "asap"}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"completion_window": "asap"}
        assert "extra_body" not in body
        assert "service_tier" not in body

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_messages_extra_body_metadata_keys_reach_the_wire(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            service_tier="flex",
            extra_body={"metadata": {"trace_id": "t-1"}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"trace_id": "t-1", "completion_window": "flex"}
        assert "extra_body" not in body

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_messages_default_tier_maps_to_asap(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            service_tier="default",
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"completion_window": "asap"}

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_messages_no_tier_sends_no_metadata_key(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert "metadata" not in body
        assert "service_tier" not in body

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_sail_messages_streaming_carries_completion_window(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(
            content="event: message_start\ndata: {}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

        await litellm.anthropic_messages(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            stream=True,
            service_tier="flex",
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"completion_window": "flex"}
        assert body["stream"] is True

    def test_messages_translate_passthrough_params_identity_by_default(self):
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        config = JSONProviderAnthropicMessagesConfig(
            SimpleProviderConfig(
                "fake",
                {"base_url": "https://x.example/v1", "api_key_env": "FAKE_KEY"},
            )
        )
        optional_params = {"metadata": {"user_id": "u"}, "max_tokens": 5}
        assert config.translate_passthrough_params(
            optional_params, {"service_tier": "flex", "extra_body": {"metadata": {"a": 1}}}
        ) == dict(optional_params)


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


_MESSAGES = [{"role": "user", "content": "hi"}]


def _wav_file() -> io.BytesIO:
    wav = io.BytesIO()
    with wave.open(wav, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00" * 1600)
    wav.seek(0)
    return wav


def _sail_chat_body(optional_params: dict) -> dict:
    config = create_config_class(JSONProviderRegistry.get("sail"))()
    return config.transform_request(
        model="zai-org/GLM-5.3",
        messages=_MESSAGES,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )


def _sail_responses_body(optional_params: dict) -> dict:
    config = create_responses_config_class(JSONProviderRegistry.get("sail"))()
    return config.transform_responses_api_request(
        model="zai-org/GLM-5.3",
        input="hi",
        response_api_optional_request_params=dict(optional_params),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


class TestSailServiceTierAsCompletionWindow:
    @pytest.mark.parametrize(
        "service_tier,expected_window",
        [("flex", "flex"), ("balanced", "balanced"), ("priority", "asap"), ("default", "asap")],
    )
    def test_service_tier_maps_to_completion_window(self, service_tier: str, expected_window: str):
        body = _sail_chat_body({"service_tier": service_tier})
        assert "service_tier" not in body
        assert body["metadata"]["completion_window"] == expected_window

    def test_auto_service_tier_dropped_without_window(self):
        body = _sail_chat_body({"service_tier": "auto"})
        assert "service_tier" not in body
        assert "completion_window" not in body.get("metadata", {})

    def test_no_service_tier_leaves_body_alone(self):
        body = _sail_chat_body({})
        assert "metadata" not in body

    def test_caller_completion_window_wins_over_service_tier(self):
        body = _sail_chat_body(
            {
                "service_tier": "flex",
                "metadata": {"completion_window": "balanced", "trace": "abc"},
            }
        )
        assert "service_tier" not in body
        assert body["metadata"] == {"completion_window": "balanced", "trace": "abc"}

    def test_optional_params_not_mutated(self):
        optional_params = {"service_tier": "flex"}
        _sail_chat_body(optional_params)
        assert optional_params == {"service_tier": "flex"}

    def test_non_sail_provider_keeps_service_tier(self):
        config = create_config_class(JSONProviderRegistry.get("parasail"))()
        body = config.transform_request(
            model="x",
            messages=_MESSAGES,
            optional_params={"service_tier": "flex"},
            litellm_params={},
            headers={},
        )
        assert body["service_tier"] == "flex"
        assert "metadata" not in body

    @pytest.mark.parametrize(
        "service_tier,expected_window",
        [("flex", "flex"), ("balanced", "balanced"), ("priority", "asap")],
    )
    def test_responses_api_service_tier_maps_to_completion_window(self, service_tier: str, expected_window: str):
        body = _sail_responses_body({"service_tier": service_tier})
        assert "service_tier" not in body
        assert body["metadata"]["completion_window"] == expected_window

    def test_responses_api_caller_completion_window_wins(self):
        body = _sail_responses_body(
            {
                "service_tier": "flex",
                "metadata": {"completion_window": "balanced"},
            }
        )
        assert "service_tier" not in body
        assert body["metadata"] == {"completion_window": "balanced"}

    @pytest.mark.respx()
    def test_sail_completion_end_to_end_sends_window_not_tier(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            service_tier="flex",
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert "service_tier" not in body
        assert body["metadata"] == {"completion_window": "flex"}

    @pytest.mark.respx()
    def test_sail_completion_window_via_extra_body_and_tier(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            service_tier="flex",
            extra_body={"metadata": {"completion_window": "balanced"}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert "service_tier" not in body
        assert body["metadata"] == {"completion_window": "balanced"}


def _sail_completion_response(prompt_tokens: int, completion_tokens: int) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model="zai-org/GLM-5.3",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class TestSailTierPricing:
    @pytest.mark.parametrize("tier", ["flex", "balanced", "priority"])
    def test_service_tier_bills_at_tier_rates(self, tier: str):
        rates = litellm.model_cost[MODEL]
        prompt_tokens, completion_tokens = 1000, 200
        suffix = {"flex": "_flex", "balanced": "_balanced"}.get(tier, "")

        cost = litellm.completion_cost(
            completion_response=_sail_completion_response(prompt_tokens, completion_tokens),
            model=MODEL,
            custom_llm_provider="sail",
            optional_params={"service_tier": tier},
        )

        expected = (
            prompt_tokens * rates[f"input_cost_per_token{suffix}"]
            + completion_tokens * rates[f"output_cost_per_token{suffix}"]
        )
        assert cost == pytest.approx(expected)

    @pytest.mark.parametrize(
        "optional_params",
        [
            {"extra_body": {"metadata": {"completion_window": "balanced"}}},
            {"extra_body": {"metadata": {"completion_window": "flex"}}},
            {"metadata": {"completion_window": "balanced"}},
            {"service_tier": "auto", "extra_body": {"metadata": {"completion_window": "flex"}}},
        ],
        ids=["extra_body_balanced", "extra_body_flex", "metadata_balanced", "auto_tier_flex_window"],
    )
    def test_completion_window_in_optional_params_bills_at_tier_rates(self, optional_params: dict):
        rates = litellm.model_cost[MODEL]
        prompt_tokens, completion_tokens = 1000, 200
        window = (optional_params.get("extra_body", {}).get("metadata") or optional_params["metadata"])[
            "completion_window"
        ]

        cost = litellm.completion_cost(
            completion_response=_sail_completion_response(prompt_tokens, completion_tokens),
            model=MODEL,
            custom_llm_provider="sail",
            optional_params=optional_params,
        )

        expected = (
            prompt_tokens * rates[f"input_cost_per_token_{window}"]
            + completion_tokens * rates[f"output_cost_per_token_{window}"]
        )
        assert cost == pytest.approx(expected)

    @pytest.mark.parametrize(
        "service_tier,window",
        [("flex", "balanced"), ("balanced", "flex")],
        ids=["flex_tier_balanced_window", "balanced_tier_flex_window"],
    )
    def test_completion_window_overrides_service_tier_on_sail(self, service_tier: str, window: str):
        rates = litellm.model_cost[MODEL]
        prompt_tokens, completion_tokens = 1000, 200

        cost = litellm.completion_cost(
            completion_response=_sail_completion_response(prompt_tokens, completion_tokens),
            model=MODEL,
            custom_llm_provider="sail",
            optional_params={
                "service_tier": service_tier,
                "extra_body": {"metadata": {"completion_window": window}},
            },
        )

        expected = (
            prompt_tokens * rates[f"input_cost_per_token_{window}"]
            + completion_tokens * rates[f"output_cost_per_token_{window}"]
        )
        assert cost == pytest.approx(expected)

    @pytest.mark.parametrize(
        "optional_params",
        [
            {"extra_body": {"metadata": {"completion_window": "flex"}}},
            {"metadata": {"completion_window": "flex"}},
        ],
        ids=["extra_body_flex_window", "metadata_flex_window"],
    )
    def test_completion_window_ignored_for_non_sail_provider(self, optional_params: dict):
        rates = litellm.model_cost["azure/gpt-5.4"]
        prompt_tokens, completion_tokens = 1000, 200

        cost = litellm.completion_cost(
            completion_response=_sail_completion_response(prompt_tokens, completion_tokens),
            model="azure/gpt-5.4",
            custom_llm_provider="azure",
            optional_params=optional_params,
        )

        expected = prompt_tokens * rates["input_cost_per_token"] + completion_tokens * rates["output_cost_per_token"]
        assert cost == pytest.approx(expected)

    def test_completion_window_asap_bills_at_base_rates(self):
        rates = litellm.model_cost[MODEL]
        prompt_tokens, completion_tokens = 1000, 200

        cost = litellm.completion_cost(
            completion_response=_sail_completion_response(prompt_tokens, completion_tokens),
            model=MODEL,
            custom_llm_provider="sail",
            optional_params={"extra_body": {"metadata": {"completion_window": "asap"}}},
        )

        expected = prompt_tokens * rates["input_cost_per_token"] + completion_tokens * rates["output_cost_per_token"]
        assert cost == pytest.approx(expected)


class TestSailWireAndBillingConsistency:
    @staticmethod
    def _expected_cost(suffix: str, prompt_tokens: int = 2, completion_tokens: int = 2) -> float:
        rates = litellm.model_cost[MODEL]
        return (
            prompt_tokens * rates[f"input_cost_per_token{suffix}"]
            + completion_tokens * rates[f"output_cost_per_token{suffix}"]
        )

    @pytest.mark.parametrize("service_tier", ["flex", "balanced"])
    @pytest.mark.respx()
    def test_asap_window_wins_over_tier_and_bills_base(self, respx_mock: respx.Router, service_tier: str):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            service_tier=service_tier,
            extra_body={"metadata": {"completion_window": "asap"}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"completion_window": "asap"}
        assert "service_tier" not in body
        assert response._hidden_params["response_cost"] == pytest.approx(self._expected_cost(""))

    @pytest.mark.respx()
    def test_tier_window_survives_extra_body_metadata_merge(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            service_tier="flex",
            extra_body={"metadata": {"trace_id": "x"}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"trace_id": "x", "completion_window": "flex"}
        assert response._hidden_params["response_cost"] == pytest.approx(self._expected_cost("_flex"))

    def test_merge_extra_body_keeps_mapped_window_in_caller_metadata(self):
        config = create_config_class(JSONProviderRegistry.get("sail"))()
        body = config.transform_request(
            model="zai-org/GLM-5.3",
            messages=_MESSAGES,
            optional_params={"service_tier": "flex"},
            litellm_params={},
            headers={},
        )

        wire = config.merge_extra_body(body, {"metadata": {"trace_id": "t-1"}})

        assert wire["metadata"] == {"completion_window": "flex", "trace_id": "t-1"}

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_aresponses_tier_window_survives_extra_body_metadata_merge(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_RESPONSES).respond(json=_responses_payload())

        response = await litellm.aresponses(
            model=MODEL,
            input="hi",
            service_tier="flex",
            extra_body={"metadata": {"trace_id": "x"}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"trace_id": "x", "completion_window": "flex"}
        assert response._hidden_params["response_cost"] == pytest.approx(self._expected_cost("_flex"))

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_aresponses_extra_body_window_bills_at_window(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_RESPONSES).respond(json=_responses_payload())

        response = await litellm.aresponses(
            model=MODEL,
            input="hi",
            extra_body={"metadata": {"completion_window": "balanced"}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"completion_window": "balanced"}
        assert response._hidden_params["response_cost"] == pytest.approx(self._expected_cost("_balanced"))

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_aresponses_extra_body_window_overrides_tier(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_RESPONSES).respond(json=_responses_payload())

        response = await litellm.aresponses(
            model=MODEL,
            input="hi",
            service_tier="flex",
            extra_body={"metadata": {"completion_window": "balanced"}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"completion_window": "balanced"}
        assert response._hidden_params["response_cost"] == pytest.approx(self._expected_cost("_balanced"))

    @pytest.mark.parametrize(
        "completion_kwargs,expected_suffix",
        [
            ({"service_tier": "flex"}, "_flex"),
            ({"service_tier": "balanced"}, "_balanced"),
            ({"extra_body": {"metadata": {"completion_window": "flex"}}}, "_flex"),
        ],
        ids=["tier_flex", "tier_balanced", "extra_body_window"],
    )
    @pytest.mark.respx()
    def test_standalone_completion_cost_bills_by_wire_window(
        self, respx_mock: respx.Router, completion_kwargs: dict, expected_suffix: str
    ):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        response = litellm.completion(model=MODEL, messages=[{"role": "user", "content": "hi"}], **completion_kwargs)

        standalone_cost = litellm.completion_cost(completion_response=response, model=MODEL)
        assert standalone_cost == pytest.approx(response._hidden_params["response_cost"])
        assert standalone_cost == pytest.approx(self._expected_cost(expected_suffix))
        assert standalone_cost != pytest.approx(self._expected_cost(""))

    @pytest.mark.respx()
    def test_standalone_completion_cost_matches_logged_cost_on_openai(self, respx_mock: respx.Router):
        respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-openai",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            }
        )

        response = litellm.completion(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hi"}],
            service_tier="flex",
            api_key="sk-test",
        )

        assert response._hidden_params["response_cost"] > 0
        assert litellm.completion_cost(completion_response=response) == pytest.approx(
            response._hidden_params["response_cost"]
        )

    @pytest.mark.respx()
    def test_standalone_completion_cost_uses_echoed_tier_on_openai(self, respx_mock: respx.Router):
        respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-openai",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-5",
                "service_tier": "default",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            }
        )

        response = litellm.completion(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": "hi"}],
            service_tier="flex",
            api_key="sk-test",
        )

        rates = litellm.model_cost["gpt-5"]
        base_cost = 2 * rates["input_cost_per_token"] + 2 * rates["output_cost_per_token"]
        flex_cost = 2 * rates["input_cost_per_token_flex"] + 2 * rates["output_cost_per_token_flex"]
        standalone_cost = litellm.completion_cost(completion_response=response)
        assert standalone_cost == pytest.approx(base_cost)
        assert standalone_cost != pytest.approx(flex_cost)

    def test_window_override_applies_when_provider_inferred_from_model(self):
        rates = litellm.model_cost[MODEL]
        prompt_tokens, completion_tokens = 1000, 200

        cost = litellm.completion_cost(
            completion_response=_sail_completion_response(prompt_tokens, completion_tokens),
            model=MODEL,
            custom_llm_provider=None,
            optional_params={
                "service_tier": "balanced",
                "extra_body": {"metadata": {"completion_window": "flex"}},
            },
        )

        expected = (
            prompt_tokens * rates["input_cost_per_token_flex"] + completion_tokens * rates["output_cost_per_token_flex"]
        )
        assert cost == pytest.approx(expected)

    @pytest.mark.parametrize(
        "optional_params,expected_window",
        [
            ({"service_tier": "default", "background": True}, "asap"),
            ({"service_tier": "priority", "background": True}, "asap"),
            ({"service_tier": "flex", "background": True}, "flex"),
        ],
        ids=["background_default_emits_asap", "background_priority_emits_asap", "background_flex_kept"],
    )
    def test_background_keeps_mapped_window(self, optional_params: dict, expected_window: str):
        body = _sail_chat_body(optional_params)
        assert "service_tier" not in body
        assert body["metadata"]["completion_window"] == expected_window

    @pytest.mark.respx()
    def test_vendor_kwarg_folds_into_request_body(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_budget=128,
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["reasoning_budget"] == 128

    @pytest.mark.respx()
    def test_non_sail_extra_body_metadata_stays_shallow(self, respx_mock: respx.Router):
        respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-openai",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            }
        )

        litellm.completion(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            metadata={"b": 2},
            extra_body={"metadata": {"a": 1}},
        )

        body = json.loads(respx_mock.calls[0].request.content)
        assert body["metadata"] == {"a": 1}


_TIER_COST_BASES = ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost")


def test_sail_tier_prices_are_monotone_and_complete():
    problems: list[str] = []
    for name, entry in litellm.model_cost.items():
        if not name.startswith("sail/"):
            continue
        for tier in ("balanced", "flex"):
            present = [base for base in _TIER_COST_BASES if entry.get(f"{base}_{tier}") is not None]
            if not present:
                continue
            if len(present) != len(_TIER_COST_BASES):
                problems.append(f"{name}: {tier} tier has {present}, expected all of {_TIER_COST_BASES}")
            for base in present:
                if entry.get(base) is None:
                    problems.append(f"{name}: has {base}_{tier} but no {base}")
                elif entry[f"{base}_{tier}"] > entry[base]:
                    problems.append(f"{name}: {base}_{tier}={entry[f'{base}_{tier}']} exceeds {base}={entry[base]}")
        for base in _TIER_COST_BASES:
            flex, balanced = entry.get(f"{base}_flex"), entry.get(f"{base}_balanced")
            if flex is not None and balanced is not None and flex > balanced:
                problems.append(f"{name}: {base}_flex={flex} exceeds {base}_balanced={balanced}")
    assert problems == []


class TestProviderListBlastRadius:
    def test_sail_stays_out_of_audio_transcription_providers(self):
        from litellm.constants import OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS, openai_compatible_providers

        assert OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS == frozenset(
            set(openai_compatible_providers) - {"sail"} | {"openai"}
        )
        assert "sail" not in OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS

    def test_get_llm_provider_unchanged_for_other_openai_compatible_bases(self):
        """Every openai_compatible_endpoints api_base resolves through the elif
        chain in get_llm_provider_logic or the JSON registry, and adding sail's
        endpoint is the only change this PR is allowed to make."""
        from litellm.constants import openai_compatible_endpoints, openai_compatible_providers
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        legacy_endpoint_providers = {
            "api.perplexity.ai": "perplexity",
            "api.endpoints.anyscale.com/v1": "anyscale",
            "api.deepinfra.com/v1/openai": "deepinfra",
            "api.mistral.ai/v1": "mistral",
            "codestral.mistral.ai/v1/chat/completions": "codestral",
            "codestral.mistral.ai/v1/fim/completions": "text-completion-codestral",
            "api.groq.com/openai/v1": "groq",
            "https://integrate.api.nvidia.com/v1": "nvidia_nim",
            "https://api.getnadir.com/v1": "nadir",
            "api.deepseek.com/v1": "deepseek",
            "api.together.ai/v1": "together_ai",
            "api.together.xyz/v1": "together_ai",
            "app.empower.dev/api/v1": "empower",
            "https://api.friendli.ai/serverless/v1": "friendliai",
            "ollama.com": "ollama",
            "https://api-inference.modelscope.cn/v1": "modelscope",
            "https://api.v0.dev/v1": "v0",
            "https://api.lambda.ai/v1": "lambda_ai",
            "https://api.inceptionlabs.ai/v1": "inception",
            "https://api.hyperbolic.xyz/v1": "hyperbolic",
            "https://ai-gateway.vercel.sh/v1": "vercel_ai_gateway",
            "https://api.edenai.run/v3": "edenai",
            "https://api.inference.wandb.ai/v1": "wandb",
            "https://gigachat.devices.sberbank.ru/api/v1": "gigachat",
        }
        providers_outside_openai_compatible_list = {
            "mistral",
            "text-completion-codestral",
            "ollama",
            "gigachat",
            "nadir",
        }

        assert len(openai_compatible_endpoints) == len(set(openai_compatible_endpoints))

        expected = {}
        for base in openai_compatible_endpoints:
            json_provider = JSONProviderRegistry.get_by_base_url(base)
            expected[base] = json_provider.slug if json_provider is not None else legacy_endpoint_providers.get(base)

        actual = {
            base: get_llm_provider(model="test-model", custom_llm_provider=None, api_base=base, api_key=None)[1]
            for base in openai_compatible_endpoints
        }
        assert actual == expected
        for base, provider in actual.items():
            json_provider = JSONProviderRegistry.get_by_base_url(base)
            if json_provider is not None:
                assert provider == json_provider.slug
                assert json_provider.base_url == base
            assert provider is None or (
                provider in openai_compatible_providers or provider in providers_outside_openai_compatible_list
            ), f"{base} resolved to {provider}, which is not a registered openai-compatible provider"


def test_model_info_balanced_fields_are_none_off_sail_and_set_on_sail():
    openai_rows = sorted(
        name
        for name, info in litellm.model_cost.items()
        if info.get("litellm_provider") == "openai" and name.startswith("gpt-")
    )
    openai_info = litellm.get_model_info(openai_rows[-1], custom_llm_provider="openai")
    assert openai_info["input_cost_per_token_balanced"] is None
    assert openai_info["output_cost_per_token_balanced"] is None

    sail_info = litellm.get_model_info(MODEL, custom_llm_provider="sail")
    assert sail_info["input_cost_per_token_balanced"] is not None
    assert sail_info["output_cost_per_token_balanced"] is not None


def _unknown_tier_message(tier: object) -> str:
    return (
        f"litellm.UnsupportedParamsError: sail does not support service_tier '{tier}'. "
        "Supported values: auto, default, flex, balanced, priority. "
        "To drop unsupported params set litellm.drop_params=True"
    )


_UNKNOWN_TIER_MESSAGE = _unknown_tier_message("bogus")


def _bad_window_message(window: object) -> str:
    return (
        f"litellm.UnsupportedParamsError: sail does not support completion_window {window!r}. "
        "Supported values: asap, flex, balanced"
    )


_INVALID_WINDOW_VALUES: Final = pytest.mark.parametrize(
    "window", ("", None, 0, "fast"), ids=["empty", "none", "zero", "fast"]
)


def _window_kwargs(shape: str, window: object, with_tier: bool) -> dict:
    kwargs = (
        {"extra_body": {"metadata": {"completion_window": window}}}
        if shape == "extra_body"
        else {"metadata": {"completion_window": window}}
    )
    return {**kwargs, **({"service_tier": "flex"} if with_tier else {})}


class TestSailCallerCompletionWindowRejected:
    @_INVALID_WINDOW_VALUES
    @pytest.mark.parametrize("with_tier", (False, True), ids=["no_tier", "with_tier"])
    @pytest.mark.respx()
    def test_chat_invalid_caller_window_raises_400(self, respx_mock: respx.Router, window: object, with_tier: bool):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            litellm.completion(model=MODEL, messages=_MESSAGES, **_window_kwargs("extra_body", window, with_tier))

        assert str(exc.value) == _bad_window_message(window)
        assert respx_mock.calls.call_count == 0

    @pytest.mark.parametrize(
        "window,shape",
        [
            *[(w, "metadata") for w in ("", "fast")],
            *[(w, "extra_body") for w in ("", None, 0, "fast")],
        ],
        ids=[
            "metadata_empty",
            "metadata_fast",
            "extra_body_empty",
            "extra_body_none",
            "extra_body_zero",
            "extra_body_fast",
        ],
    )
    @pytest.mark.parametrize("with_tier", (False, True), ids=["no_tier", "with_tier"])
    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_responses_invalid_caller_window_raises_400(
        self, respx_mock: respx.Router, window: object, shape: str, with_tier: bool
    ):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            await litellm.aresponses(model=MODEL, input="hi", **_window_kwargs(shape, window, with_tier))

        assert str(exc.value) == _bad_window_message(window)
        assert respx_mock.calls.call_count == 0

    @_INVALID_WINDOW_VALUES
    @pytest.mark.parametrize("with_tier", (False, True), ids=["no_tier", "with_tier"])
    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_messages_invalid_caller_window_raises_400(
        self, respx_mock: respx.Router, window: object, with_tier: bool
    ):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            await litellm.anthropic_messages(
                model=MODEL, messages=_MESSAGES, max_tokens=50, **_window_kwargs("extra_body", window, with_tier)
            )

        assert str(exc.value) == _bad_window_message(window)
        assert respx_mock.calls.call_count == 0


class TestSailUnknownServiceTierRejected:
    @pytest.mark.respx()
    def test_chat_unknown_tier_raises_400_before_the_wire(self, respx_mock: respx.Router):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            litellm.completion(model=MODEL, messages=_MESSAGES, service_tier="bogus")

        assert str(exc.value) == _UNKNOWN_TIER_MESSAGE
        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_responses_unknown_tier_raises_400_before_the_wire(self, respx_mock: respx.Router):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            await litellm.aresponses(model=MODEL, input="hi", service_tier="bogus")

        assert str(exc.value) == _UNKNOWN_TIER_MESSAGE
        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_messages_unknown_tier_raises_400_before_the_wire(self, respx_mock: respx.Router):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            await litellm.anthropic_messages(model=MODEL, messages=_MESSAGES, max_tokens=50, service_tier="bogus")

        assert str(exc.value) == _UNKNOWN_TIER_MESSAGE
        assert respx_mock.calls.call_count == 0

    @pytest.mark.respx()
    def test_chat_unknown_tier_dropped_with_drop_params(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(model=MODEL, messages=_MESSAGES, service_tier="bogus", drop_params=True)
        litellm.completion(model=MODEL, messages=_MESSAGES)

        bodies = [json.loads(call.request.content) for call in respx_mock.calls]
        assert bodies[0] == bodies[1] == {"model": MODEL.split("/", 1)[1], "messages": _MESSAGES}

    @pytest.mark.respx()
    def test_chat_unknown_tier_dropped_with_global_drop_params(
        self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
    ):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())
        monkeypatch.setattr(litellm, "drop_params", True)

        litellm.completion(model=MODEL, messages=_MESSAGES, service_tier="bogus")

        assert len(respx_mock.calls) == 1
        body = json.loads(respx_mock.calls[0].request.content)
        assert "service_tier" not in body
        assert "metadata" not in body

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_responses_unknown_tier_dropped_with_drop_params(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_RESPONSES).respond(json=_responses_payload())

        await litellm.aresponses(model=MODEL, input="hi", service_tier="bogus", drop_params=True)
        await litellm.aresponses(model=MODEL, input="hi")

        bodies = [json.loads(call.request.content) for call in respx_mock.calls]
        assert bodies[0] == bodies[1]
        assert "service_tier" not in bodies[0]

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_messages_unknown_tier_dropped_with_drop_params(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(
            model=MODEL, messages=_MESSAGES, max_tokens=50, service_tier="bogus", drop_params=True
        )
        await litellm.anthropic_messages(model=MODEL, messages=_MESSAGES, max_tokens=50)

        bodies = [json.loads(call.request.content) for call in respx_mock.calls]
        assert bodies[0] == bodies[1]
        assert "service_tier" not in bodies[0]
        assert "extra_body" not in bodies[0]

    @pytest.mark.respx()
    def test_chat_uppercase_tier_raises_400_before_the_wire(self, respx_mock: respx.Router):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            litellm.completion(model=MODEL, messages=_MESSAGES, service_tier="FLEX")

        assert str(exc.value) == _unknown_tier_message("FLEX")
        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_responses_uppercase_tier_raises_400_before_the_wire(self, respx_mock: respx.Router):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            await litellm.aresponses(model=MODEL, input="hi", service_tier="FLEX")

        assert str(exc.value) == _unknown_tier_message("FLEX")
        assert respx_mock.calls.call_count == 0

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_messages_uppercase_tier_raises_400_before_the_wire(self, respx_mock: respx.Router):
        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            await litellm.anthropic_messages(model=MODEL, messages=_MESSAGES, max_tokens=50, service_tier="FLEX")

        assert str(exc.value) == _unknown_tier_message("FLEX")
        assert respx_mock.calls.call_count == 0

    @pytest.mark.respx()
    def test_chat_auto_tier_sends_no_tier_no_metadata(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(json=_chat_completion_payload())

        litellm.completion(model=MODEL, messages=_MESSAGES, service_tier="auto")

        body = json.loads(respx_mock.calls[0].request.content)
        assert "service_tier" not in body
        assert "metadata" not in body

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_responses_auto_tier_sends_no_tier_no_metadata(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_RESPONSES).respond(json=_responses_payload())

        await litellm.aresponses(model=MODEL, input="hi", service_tier="auto")

        body = json.loads(respx_mock.calls[0].request.content)
        assert "service_tier" not in body
        assert "metadata" not in body

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_messages_auto_tier_sends_no_tier_no_metadata(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_MESSAGES).respond(json=_messages_payload())

        await litellm.anthropic_messages(model=MODEL, messages=_MESSAGES, max_tokens=50, service_tier="auto")

        body = json.loads(respx_mock.calls[0].request.content)
        assert "service_tier" not in body
        assert "metadata" not in body

    def test_non_sail_json_provider_keeps_unknown_tier_passthrough(self):
        config = create_config_class(
            SimpleProviderConfig("fake", {"base_url": "https://x.example/v1", "api_key_env": "FAKE_KEY"})
        )()
        optional_params: dict = {}
        assert config.map_openai_params({"service_tier": "bogus"}, optional_params, "m", False) == optional_params

        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        messages_config = JSONProviderAnthropicMessagesConfig(
            SimpleProviderConfig("fake", {"base_url": "https://x.example/v1", "api_key_env": "FAKE_KEY"})
        )
        optional = {"metadata": {"user_id": "u"}, "max_tokens": 5}
        assert messages_config.translate_passthrough_params(
            optional, {"service_tier": "bogus", "extra_body": {"a": 1}}
        ) == dict(optional)


def _chat_completion_stream_with_usage() -> str:
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
        {
            "id": "chatcmpl-sail-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "zai-org/GLM-5.3",
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        },
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


class TestSailCallerWindowOnMergeExtraBody:
    @pytest.mark.parametrize(
        "experimental_handler", ["true", "false"], ids=["experimental_handler", "default_handler"]
    )
    @pytest.mark.respx(assert_all_called=False)
    def test_chat_extra_body_empty_window_raises_400(
        self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch, experimental_handler: str
    ):
        monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", experimental_handler)

        with pytest.raises(litellm.UnsupportedParamsError) as exc:
            litellm.completion(
                model=MODEL,
                messages=_MESSAGES,
                extra_body={"metadata": {"completion_window": ""}},
            )

        assert str(exc.value) == _bad_window_message("")
        assert respx_mock.calls.call_count == 0


class TestSailStreamingRebuildCost:
    @staticmethod
    def _expected_cost(suffix: str, prompt_tokens: int = 2, completion_tokens: int = 2) -> float:
        rates = litellm.model_cost[MODEL]
        return (
            prompt_tokens * rates[f"input_cost_per_token{suffix}"]
            + completion_tokens * rates[f"output_cost_per_token{suffix}"]
        )

    @pytest.mark.respx()
    def test_stream_chunk_builder_bills_by_wire_window(self, respx_mock: respx.Router):
        respx_mock.post(SAIL_CHAT_COMPLETIONS).respond(
            content=_chat_completion_stream_with_usage(),
            headers={"content-type": "text/event-stream"},
        )

        response = litellm.completion(
            model=MODEL,
            messages=_MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
            service_tier="flex",
        )
        chunks = list(response)

        rebuilt = litellm.stream_chunk_builder(chunks)
        rebuilt_cost = litellm.completion_cost(completion_response=rebuilt, model=MODEL)
        assert rebuilt_cost == pytest.approx(self._expected_cost("_flex"))
        assert rebuilt_cost != pytest.approx(self._expected_cost(""))
        logged_cost = rebuilt._hidden_params.get("response_cost") or chunks[-1]._hidden_params.get("response_cost")
        assert rebuilt_cost == pytest.approx(logged_cost)

    @pytest.mark.respx()
    def test_stream_chunk_builder_openai_cost_unchanged(self, respx_mock: respx.Router):
        respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
            content=_chat_completion_stream_with_usage(),
            headers={"content-type": "text/event-stream"},
        )

        response = litellm.completion(
            model="openai/gpt-4.1-mini",
            messages=_MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
            api_key="sk-test",
        )
        chunks = list(response)

        rebuilt = litellm.stream_chunk_builder(chunks)
        rates = litellm.model_cost["gpt-4.1-mini"]
        expected = 2 * rates["input_cost_per_token"] + 2 * rates["output_cost_per_token"]
        assert litellm.completion_cost(completion_response=rebuilt, model="gpt-4.1-mini") == pytest.approx(expected)


class TestSailHiddenTierPrecedence:
    def test_hidden_optional_params_tier_beats_echoed_tier(self):
        response = _sail_completion_response(1000, 200)
        response.service_tier = "balanced"
        response._hidden_params["optional_params"] = {"service_tier": "flex"}

        cost = litellm.completion_cost(
            completion_response=response,
            model=MODEL,
            custom_llm_provider="sail",
        )

        rates = litellm.model_cost[MODEL]
        flex_cost = 1000 * rates["input_cost_per_token_flex"] + 200 * rates["output_cost_per_token_flex"]
        balanced_cost = 1000 * rates["input_cost_per_token_balanced"] + 200 * rates[
            "output_cost_per_token_balanced"
        ]
        assert cost == pytest.approx(flex_cost)
        assert cost != pytest.approx(balanced_cost)


class TestNonSailTranscription:
    @pytest.mark.respx(assert_all_called=False)
    def test_anthropic_transcription_raises_bad_request_without_upstream_call(self, respx_mock: respx.Router):
        with pytest.raises(litellm.BadRequestError) as exc_info:
            litellm.transcription(
                model="anthropic/claude-sonnet-4-5",
                file=_wav_file(),
                api_key="sk-test",
            )

        assert exc_info.value.status_code == 400
        assert "does not support audio transcription" in str(exc_info.value)
        assert respx_mock.calls.call_count == 0
