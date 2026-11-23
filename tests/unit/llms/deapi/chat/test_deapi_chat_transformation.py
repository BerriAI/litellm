"""deAPI (`deapi/...`) chat provider: an OpenAI-compatible gateway that reports the charge of every
answer in `usage.cost` and in the `X-DeAPI-Cost` header, so LiteLLM bills that amount instead of a price map."""

import json
from pathlib import Path

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params, response_cost_calculator
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.deapi.chat.transformation import DeapiChatConfig
from litellm.llms.deapi.common_utils import DeapiException
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

REPO_ROOT = Path(__file__).resolve().parents[5]
DEAPI_BASE = "https://oai.deapi.ai/v1"
DEAPI_CHAT_URL = f"{DEAPI_BASE}/chat/completions"
MODEL = "deapi/deepseek-v4.1-flash"
SELLER_MODEL = "deepseek-v4.1-flash"
REPORTED_COST = 0.00001733
USAGE = {
    "prompt_tokens": 7,
    "completion_tokens": 12,
    "total_tokens": 19,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0},
}
MESSAGES = [{"role": "user", "content": "Say hi."}]


def _chat_completion(cost: float | None = REPORTED_COST) -> httpx.Response:
    """Live `/v1/chat/completions` answer: OpenAI shape with `usage.cost`, mirrored in `X-DeAPI-Cost`."""
    usage = USAGE if cost is None else {**USAGE, "cost": cost}
    body = {
        "id": "chatcmpl-01a1f852",
        "object": "chat.completion",
        "created": 1795858752,
        "model": SELLER_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi there!"}}],
        "usage": usage,
    }
    headers = {} if cost is None else {"X-DeAPI-Cost": f"{cost:.8f}"}
    return httpx.Response(200, json=body, headers=headers)


def _stream_chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-01a1f853",
        "object": "chat.completion.chunk",
        "created": 1795858753,
        "model": SELLER_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _stream(cost: float | None = REPORTED_COST) -> httpx.Response:
    """Live stream with `include_usage`: the usage frame has no choices and carries `usage.cost`."""
    usage_frame = {
        **_stream_chunk({}),
        "choices": [],
        "usage": USAGE if cost is None else {**USAGE, "cost": cost},
    }
    frames = (
        _stream_chunk({"role": "assistant", "content": "Hi"}),
        _stream_chunk({"content": " there!"}),
        _stream_chunk({}, finish_reason="stop"),
        usage_frame,
    )
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _request_body(respx_mock) -> dict:
    return json.loads(respx_mock.calls.last.request.content)


class TestProviderResolution:
    def test_strips_only_the_deapi_prefix(self, deapi_key):
        assert get_llm_provider("deapi/deepseek-v4.1-flash:economy") == (
            "deepseek-v4.1-flash:economy",
            "deapi",
            deapi_key,
            DEAPI_BASE,
        )

    def test_explicit_credentials_win_over_env(self, deapi_key):
        _, _, api_key, api_base = get_llm_provider(MODEL, api_key="explicit-key", api_base="https://gw.internal/v1")

        assert (api_key, api_base) == ("explicit-key", "https://gw.internal/v1")

    def test_the_deapi_api_base_is_recognised_without_the_prefix(self, deapi_key):
        assert get_llm_provider(SELLER_MODEL, api_base=DEAPI_BASE) == (SELLER_MODEL, "deapi", deapi_key, DEAPI_BASE)


class TestRegistration:
    def test_provider_is_registered_everywhere_routing_looks(self):
        assert LlmProviders.DEAPI.value == "deapi"
        assert "deapi" in litellm.provider_list
        assert "deapi" in litellm.openai_compatible_providers
        assert DEAPI_BASE in litellm.openai_compatible_endpoints
        assert isinstance(
            ProviderConfigManager.get_provider_chat_config(model=SELLER_MODEL, provider=LlmProviders.DEAPI),
            DeapiChatConfig,
        )

    def test_reasoning_effort_is_supported_without_a_price_map_entry(self):
        supported = litellm.get_supported_openai_params(model=SELLER_MODEL, custom_llm_provider="deapi")

        assert supported is not None
        assert {"tools", "tool_choice", "response_format", "reasoning_effort", "stream_options"} <= set(supported)
        assert supported.count("reasoning_effort") == 1

    def test_validate_environment_names_the_deapi_key(self, monkeypatch):
        monkeypatch.delenv("DEAPI_API_KEY", raising=False)
        missing = litellm.validate_environment(model=MODEL)
        monkeypatch.setenv("DEAPI_API_KEY", "sk-dp-test-key")
        present = litellm.validate_environment(model=MODEL)

        assert (missing["keys_in_environment"], missing["missing_keys"]) == (False, ["DEAPI_API_KEY"])
        assert (present["keys_in_environment"], present["missing_keys"]) == (True, [])


class TestRequestTransformation:
    def _request(self, optional_params: dict) -> dict:
        return DeapiChatConfig().transform_request(
            model=SELLER_MODEL, messages=MESSAGES, optional_params=optional_params, litellm_params={}, headers={}
        )

    def test_streaming_request_asks_for_the_usage_frame_that_carries_the_cost(self):
        assert self._request({"stream": True})["stream_options"] == {"include_usage": True}

    def test_streaming_request_overrides_a_caller_opt_out(self):
        assert self._request({"stream": True, "stream_options": {"include_usage": False}})["stream_options"] == {
            "include_usage": True
        }

    def test_non_streaming_request_carries_no_stream_options(self):
        assert "stream_options" not in self._request({"max_tokens": 16})

    def test_extra_body_cannot_switch_off_the_usage_frame_of_a_stream(self):
        merged = DeapiChatConfig().transform_extra_body(
            extra_body={"stream_options": {"include_usage": False, "x": 1}},
            request=self._request({"stream": True}),
            model=SELLER_MODEL,
            litellm_params={},
        )
        assert merged["stream_options"] == {"include_usage": True, "x": 1}

    def test_a_stream_requested_through_extra_body_still_asks_for_the_usage_frame(self):
        merged = DeapiChatConfig().transform_extra_body(
            extra_body={"stream": True}, request=self._request({}), model=SELLER_MODEL, litellm_params={}
        )
        assert merged == {"stream": True, "stream_options": {"include_usage": True}}

    def test_extra_body_of_a_non_streaming_request_passes_through(self):
        extra_body = {"stream_options": {"include_usage": False}, "seed": 7}
        merged = DeapiChatConfig().transform_extra_body(
            extra_body=extra_body, request=self._request({}), model=SELLER_MODEL, litellm_params={}
        )
        assert merged == extra_body


class TestCompletion:
    def test_posts_to_deapi_with_the_bearer_key_and_the_model_id(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=_chat_completion())

        response = litellm.completion(model=MODEL, messages=MESSAGES, max_tokens=16, reasoning_effort="none")

        assert response.choices[0].message.content == "Hi there!"
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {deapi_key}"
        body = _request_body(respx_mock)
        assert (body["model"], body["messages"], body["max_tokens"], body["reasoning_effort"]) == (
            SELLER_MODEL,
            MESSAGES,
            16,
            "none",
        )

    def test_the_reported_cost_is_the_response_cost(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=_chat_completion())

        response = litellm.completion(model=MODEL, messages=MESSAGES, max_tokens=16)

        assert get_response_cost_from_hidden_params(response._hidden_params) == REPORTED_COST
        assert (
            response_cost_calculator(
                response_object=response,
                model=SELLER_MODEL,
                custom_llm_provider="deapi",
                call_type="completion",
                optional_params={},
            )
            == REPORTED_COST
        )

    def test_an_answer_without_cost_leaves_pricing_to_the_price_map(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=_chat_completion(cost=None))

        response = litellm.completion(model=MODEL, messages=MESSAGES, max_tokens=16)

        assert response.choices[0].message.content == "Hi there!"
        assert get_response_cost_from_hidden_params(response._hidden_params) is None

    @pytest.mark.asyncio
    async def test_async_call_logs_the_reported_cost_as_spend(
        self, deapi_key, httpx_transport, spend_capture, respx_mock
    ):
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=_chat_completion())

        await litellm.acompletion(model=MODEL, messages=MESSAGES, max_tokens=16, litellm_call_id=spend_capture.call_id)
        await spend_capture.settle()

        assert spend_capture.costs == [REPORTED_COST]


class TestStreaming:
    def test_include_usage_surfaces_the_cost_on_the_usage_chunk(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=_stream())

        chunks = list(
            litellm.completion(model=MODEL, messages=MESSAGES, stream=True, stream_options={"include_usage": True})
        )

        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "Hi there!"
        usage_chunks = [chunk for chunk in chunks if getattr(chunk, "usage", None) is not None]
        assert len(usage_chunks) == 1
        assert (usage_chunks[0].usage.total_tokens, usage_chunks[0].usage.cost) == (19, REPORTED_COST)

    @pytest.mark.asyncio
    async def test_a_stream_without_include_usage_is_still_billed_the_reported_cost(
        self, deapi_key, httpx_transport, spend_capture, respx_mock
    ):
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=_stream())

        stream = await litellm.acompletion(
            model=MODEL, messages=MESSAGES, stream=True, litellm_call_id=spend_capture.call_id
        )
        text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])
        await spend_capture.settle()

        assert _request_body(respx_mock)["stream_options"] == {"include_usage": True}
        assert text == "Hi there!"
        assert spend_capture.costs == [REPORTED_COST]

    @pytest.mark.asyncio
    async def test_extra_body_opting_out_of_usage_is_still_billed_the_reported_cost(
        self, deapi_key, httpx_transport, spend_capture, respx_mock
    ):
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=_stream())

        stream = await litellm.acompletion(
            model=MODEL,
            messages=MESSAGES,
            stream=True,
            extra_body={"stream_options": {"include_usage": False}},
            litellm_call_id=spend_capture.call_id,
        )
        text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])
        await spend_capture.settle()

        assert _request_body(respx_mock)["stream_options"] == {"include_usage": True}
        assert text == "Hi there!"
        assert spend_capture.costs == [REPORTED_COST]


class TestErrors:
    def test_invalid_key_maps_to_authentication_error(self, deapi_key, respx_mock):
        envelope = {
            "error": {
                "message": "Invalid API key.",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=httpx.Response(401, json=envelope))

        with pytest.raises(litellm.AuthenticationError, match="Invalid API key"):
            litellm.completion(model=MODEL, messages=MESSAGES)

    def test_unsupported_reasoning_effort_maps_to_bad_request(self, deapi_key, respx_mock):
        envelope = {
            "error": {
                "message": "Unsupported value: 'reasoning_effort' does not support 'none' with this model.",
                "type": "invalid_request_error",
                "param": "reasoning_effort",
                "code": "unsupported_value",
            }
        }
        respx_mock.post(DEAPI_CHAT_URL).mock(return_value=httpx.Response(400, json=envelope))

        with pytest.raises(litellm.BadRequestError, match="does not support 'none'"):
            litellm.completion(model="deapi/glm-5.3", messages=MESSAGES, reasoning_effort="none")

    def test_429_maps_to_rate_limit_error(self, deapi_key, respx_mock):
        envelope = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": "rate_limited"}}
        respx_mock.post(DEAPI_CHAT_URL).mock(
            return_value=httpx.Response(429, json=envelope, headers={"Retry-After": "7"})
        )

        with pytest.raises(litellm.RateLimitError, match="Rate limit exceeded"):
            litellm.completion(model=MODEL, messages=MESSAGES, num_retries=0)

    def test_error_class_is_the_deapi_exception(self):
        error = DeapiChatConfig().get_error_class("boom", 503, {"Content-Type": "application/json"})

        assert isinstance(error, DeapiException)
        assert isinstance(error, BaseLLMException)
        assert (error.message, error.status_code) == ("boom", 503)


class TestModelListing:
    CATALOG = {"object": "list", "data": [{"id": "deepseek-v4.1-flash", "object": "model"}, {"id": "Kokoro"}]}

    def test_lists_the_key_catalog_as_routable_model_names(self, deapi_key, respx_mock):
        respx_mock.get(f"{DEAPI_BASE}/models").mock(return_value=httpx.Response(200, json=self.CATALOG))

        assert DeapiChatConfig().get_models() == ["deapi/deepseek-v4.1-flash", "deapi/Kokoro"]
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {deapi_key}"

    def test_a_rejected_catalog_request_surfaces_the_status_and_body(self, deapi_key, respx_mock):
        respx_mock.get(f"{DEAPI_BASE}/models").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key."}})
        )

        with pytest.raises(DeapiException) as rejected:
            DeapiChatConfig().get_models()

        assert rejected.value.status_code == 401
        assert "Invalid API key" in rejected.value.message


class TestDashboardRegistration:
    def test_add_model_form_offers_deapi_with_a_required_key_and_optional_base(self):
        fields_path = REPO_ROOT / "litellm" / "proxy" / "public_endpoints" / "provider_create_fields.json"
        entries = [e for e in json.loads(fields_path.read_text()) if e["litellm_provider"] == "deapi"]

        assert len(entries) == 1
        entry = entries[0]
        assert (entry["provider"], entry["provider_display_name"]) == ("DEAPI", "deAPI")
        assert entry["default_model_placeholder"].startswith("deapi/")
        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert (fields["api_key"]["required"], fields["api_key"]["field_type"]) == (True, "password")
        assert (fields["api_base"]["required"], fields["api_base"]["placeholder"]) == (False, DEAPI_BASE)
