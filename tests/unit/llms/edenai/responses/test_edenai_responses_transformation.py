"""Eden AI `/v3/responses`: OpenAI's Responses API served by Eden's gateway. Eden reports the real
per-request cost at the top level of the body and, on streams, on the final usage frame."""

import json

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.edenai.responses.transformation import EdenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStreamEvents
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

EDEN_BASE = "https://api.edenai.run/v3"
EDEN_EU_BASE = "https://api.eu.edenai.run/v3"
EDEN_RESPONSES_URL = f"{EDEN_BASE}/responses"
EDEN_REPORTED_COST = 0.0042
MODEL = "edenai/openai/gpt-4.1-nano"
SELLER_MODEL = "openai/gpt-4.1-nano"


def _usage(cost: float | None) -> dict:
    usage = {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14}
    return usage if cost is None else {**usage, "cost": cost}


def _output(text: str = "OK") -> list[dict]:
    return [
        {
            "id": "msg_eden_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    ]


def _eden_response(cost: float | None = EDEN_REPORTED_COST) -> dict:
    """Live `/v3/responses` body: OpenAI shape plus Eden's top-level `cost` and `provider`."""
    body = {
        "id": "resp_eden_1",
        "object": "response",
        "created_at": 1788443790,
        "status": "completed",
        "model": "gpt-4.1-nano",
        "provider": "openai",
        "output": _output(),
        "usage": _usage(cost),
    }
    return body if cost is None else {**body, "cost": cost}


def _eden_stream_events(cost: float | None = EDEN_REPORTED_COST) -> tuple[dict, ...]:
    """Live stream: the `response.completed` frame carries Eden's cost on `usage` only."""
    in_progress = {
        "id": "resp_eden_1",
        "object": "response",
        "created_at": 1788443790,
        "status": "in_progress",
        "model": SELLER_MODEL,
        "output": [],
    }
    return (
        {"type": "response.created", "sequence_number": 0, "response": in_progress},
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "msg_eden_1",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "item_id": "msg_eden_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "OK",
        },
        {
            "type": "response.completed",
            "sequence_number": 3,
            "response": {**in_progress, "status": "completed", "output": _output(), "usage": _usage(cost)},
        },
    )


def _sse(events: tuple[dict, ...]) -> httpx.Response:
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _request_body(respx_mock) -> dict:
    return json.loads(respx_mock.calls.last.request.content)


class TestRegistration:
    def test_eden_is_a_native_responses_provider(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.EDENAI, model=SELLER_MODEL
        )

        assert isinstance(config, EdenAIResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.EDENAI

    def test_the_provider_string_resolves_too(self):
        assert isinstance(
            ProviderConfigManager.get_provider_responses_api_config(provider="edenai"), EdenAIResponsesAPIConfig
        )

    def test_websocket_callers_get_the_managed_handler(self):
        """Eden serves the Responses API over HTTP only, so a websocket client has to be bridged
        rather than dialled straight through to a wss:// endpoint Eden does not have."""
        assert EdenAIResponsesAPIConfig().supports_native_websocket() is False


class TestEndpointResolution:
    def test_defaults_to_the_global_endpoint(self, eden_key):
        assert EdenAIResponsesAPIConfig().get_complete_url(api_base=None, litellm_params={}) == EDEN_RESPONSES_URL

    def test_env_api_base_moves_to_the_eu_endpoint(self, eden_key, monkeypatch):
        monkeypatch.setenv("EDENAI_API_BASE", EDEN_EU_BASE)

        assert (
            EdenAIResponsesAPIConfig().get_complete_url(api_base=None, litellm_params={}) == f"{EDEN_EU_BASE}/responses"
        )

    def test_explicit_api_base_wins_and_loses_its_trailing_slash(self, eden_key, monkeypatch):
        monkeypatch.setenv("EDENAI_API_BASE", EDEN_EU_BASE)

        url = EdenAIResponsesAPIConfig().get_complete_url(api_base="https://eden.internal/v3/", litellm_params={})

        assert url == "https://eden.internal/v3/responses"


class TestAuthentication:
    def test_env_key_becomes_the_bearer_header(self, eden_key):
        headers = EdenAIResponsesAPIConfig().validate_environment(
            headers={"x-trace": "1"}, model=SELLER_MODEL, litellm_params=None
        )

        assert headers == {"x-trace": "1", "Authorization": f"Bearer {eden_key}"}

    def test_explicit_key_wins_over_env(self, eden_key):
        headers = EdenAIResponsesAPIConfig().validate_environment(
            headers={}, model=SELLER_MODEL, litellm_params=GenericLiteLLMParams(api_key="explicit-key")
        )

        assert headers["Authorization"] == "Bearer explicit-key"

    def test_missing_key_is_an_authentication_error(self, no_eden_key):
        with pytest.raises(litellm.AuthenticationError, match="EDENAI_API_KEY"):
            EdenAIResponsesAPIConfig().validate_environment(headers={}, model=SELLER_MODEL, litellm_params=None)


class TestResponses:
    def test_posts_to_eden_with_the_bearer_key_and_the_seller_model_id(self, eden_key, respx_mock):
        respx_mock.post(EDEN_RESPONSES_URL).mock(return_value=httpx.Response(200, json=_eden_response()))

        response = litellm.responses(model=MODEL, input="Say OK", max_output_tokens=16)

        assert isinstance(response, ResponsesAPIResponse)
        assert response.output[0].content[0].text == "OK"
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {eden_key}"
        body = _request_body(respx_mock)
        assert (body["model"], body["input"], body["max_output_tokens"]) == (SELLER_MODEL, "Say OK", 16)

    def test_eden_reported_cost_beats_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_RESPONSES_URL).mock(return_value=httpx.Response(200, json=_eden_response()))

        response = litellm.responses(model=MODEL, input="Say OK", max_output_tokens=16)

        assert get_response_cost_from_hidden_params(response._hidden_params) == EDEN_REPORTED_COST
        assert response._hidden_params["response_cost"] == EDEN_REPORTED_COST

    def test_a_body_without_cost_leaves_pricing_to_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_RESPONSES_URL).mock(return_value=httpx.Response(200, json=_eden_response(cost=None)))

        response = litellm.responses(model=MODEL, input="Say OK", max_output_tokens=16)

        assert response.output[0].content[0].text == "OK"
        assert get_response_cost_from_hidden_params(response._hidden_params) is None

    def test_stateful_params_pass_through_to_eden(self, eden_key, respx_mock):
        respx_mock.post(EDEN_RESPONSES_URL).mock(return_value=httpx.Response(200, json=_eden_response()))

        litellm.responses(
            model=MODEL,
            input="Say OK",
            previous_response_id="resp_previous",
            store=False,
            reasoning={"effort": "low"},
        )

        body = _request_body(respx_mock)
        assert (body["previous_response_id"], body["store"], body["reasoning"]) == (
            "resp_previous",
            False,
            {"effort": "low"},
        )

    def test_extra_body_forwards_eden_only_fields(self, eden_key, respx_mock):
        respx_mock.post(EDEN_RESPONSES_URL).mock(return_value=httpx.Response(200, json=_eden_response()))

        litellm.responses(
            model=MODEL,
            input="Say OK",
            extra_body={"fallbacks": ["anthropic/claude-sonnet-latest"], "routing": {"sort": "latency"}},
        )

        body = _request_body(respx_mock)
        assert body["fallbacks"] == ["anthropic/claude-sonnet-latest"]
        assert body["routing"] == {"sort": "latency"}
        assert "extra_body" not in body


class TestStreaming:
    def test_stream_forwards_eden_events_and_bills_the_usage_cost(self, eden_key, respx_mock):
        respx_mock.post(EDEN_RESPONSES_URL).mock(return_value=_sse(_eden_stream_events()))

        stream = litellm.responses(model=MODEL, input="Say OK", stream=True)
        events = list(stream)

        assert _request_body(respx_mock)["stream"] is True
        assert [event.type for event in events] == [
            ResponsesAPIStreamEvents.RESPONSE_CREATED,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        ]
        assert events[2].delta == "OK"
        assert events[-1].response.usage.cost == EDEN_REPORTED_COST
        assert stream.logging_obj.model_call_details["response_cost"] == EDEN_REPORTED_COST


class TestErrors:
    def test_401_detail_body_is_an_authentication_error(self, eden_key, respx_mock):
        respx_mock.post(EDEN_RESPONSES_URL).mock(return_value=httpx.Response(401, json={"detail": "Invalid token"}))

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            litellm.responses(model=MODEL, input="Say OK")

    def test_400_envelope_is_a_bad_request_error(self, eden_key, respx_mock):
        respx_mock.post(EDEN_RESPONSES_URL).mock(
            return_value=httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Model(s) not found or inactive: openai/does-not-exist",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_parameter",
                    }
                },
            )
        )

        with pytest.raises(litellm.BadRequestError, match="not found or inactive"):
            litellm.responses(model="edenai/openai/does-not-exist", input="Say OK")
