"""Eden AI `/v3/v1/messages`: Anthropic's Messages API served by Eden's gateway for every model in
its catalog. The Anthropic payload is forwarded untranslated, and Eden reports the real per-request
cost at the top level of a non-streaming body."""

import asyncio
import json
import time
import uuid

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.edenai.messages.transformation import EdenAIAnthropicMessagesConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

EDEN_BASE = "https://api.edenai.run/v3"
EDEN_EU_BASE = "https://api.eu.edenai.run/v3"
EDEN_MESSAGES_URL = f"{EDEN_BASE}/v1/messages"
EDEN_REPORTED_COST = 0.0042
MODEL = "edenai/openai/gpt-4.1-nano"
SELLER_MODEL = "openai/gpt-4.1-nano"
MESSAGES = [{"role": "user", "content": "Say OK"}]
BILLING_BLOCK = {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli"}
SYSTEM_BLOCK = {"type": "text", "text": "Be terse", "cache_control": {"type": "ephemeral"}}


def _eden_message(cost: float | None = EDEN_REPORTED_COST) -> dict:
    """Live body: Anthropic shape with the id sent to Eden echoed in `model` and Eden's top-level `cost`."""
    body = {
        "id": "chatcmpl-eden-1",
        "type": "message",
        "role": "assistant",
        "model": SELLER_MODEL,
        "stop_sequence": None,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 1},
        "content": [{"type": "text", "text": "OK"}],
    }
    return body if cost is None else {**body, "cost": cost}


def _eden_stream() -> httpx.Response:
    """Live stream: Anthropic events with token usage on `message_delta` and no cost anywhere."""
    message = {
        "id": "msg_eden_1",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": SELLER_MODEL,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    events = (
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
        {"type": "message_stop"},
    )
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _request_body(respx_mock) -> dict:
    return json.loads(respx_mock.calls.last.request.content)


def _logging_obj() -> Logging:
    return Logging(
        model=SELLER_MODEL,
        messages=MESSAGES,
        stream=False,
        call_type="anthropic_messages",
        start_time=time.time(),
        litellm_call_id="eden-messages-unit",
        function_id="eden-messages-unit",
    )


class TestRegistration:
    @pytest.mark.parametrize("model", [SELLER_MODEL, "anthropic/claude-sonnet-latest"])
    def test_eden_serves_anthropic_messages_natively_for_every_catalog_model(self, model):
        config = ProviderConfigManager.get_provider_anthropic_messages_config(model=model, provider=LlmProviders.EDENAI)

        assert isinstance(config, EdenAIAnthropicMessagesConfig)
        assert config.custom_llm_provider == "edenai"


class TestEndpointResolution:
    def _url(self, api_base: str | None) -> str:
        return EdenAIAnthropicMessagesConfig().get_complete_url(
            api_base=api_base, api_key=None, model=SELLER_MODEL, optional_params={}, litellm_params={}
        )

    def test_defaults_to_the_global_endpoint(self, eden_key):
        assert self._url(None) == EDEN_MESSAGES_URL

    def test_env_api_base_moves_to_the_eu_endpoint(self, eden_key, monkeypatch):
        monkeypatch.setenv("EDENAI_API_BASE", EDEN_EU_BASE)

        assert self._url(None) == f"{EDEN_EU_BASE}/v1/messages"

    def test_explicit_api_base_wins_over_env(self, eden_key, monkeypatch):
        monkeypatch.setenv("EDENAI_API_BASE", EDEN_EU_BASE)

        assert self._url("https://eden.internal/v3/") == "https://eden.internal/v3/v1/messages"


class TestAuthentication:
    def _headers(self, headers: dict, api_key: str | None = None) -> dict:
        resolved, _ = EdenAIAnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers=headers,
            model=SELLER_MODEL,
            messages=MESSAGES,
            optional_params={},
            litellm_params={},
            api_key=api_key,
        )
        return resolved

    def test_env_key_becomes_the_bearer_header_with_the_anthropic_version(self, eden_key):
        headers = self._headers({})

        assert headers == {
            "authorization": f"Bearer {eden_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def test_explicit_key_wins_over_env(self, eden_key):
        assert self._headers({}, api_key="explicit-key")["authorization"] == "Bearer explicit-key"

    def test_a_caller_supplied_authorization_header_is_kept(self, eden_key):
        headers = self._headers({"Authorization": "Bearer caller-token"})

        assert headers["Authorization"] == "Bearer caller-token"
        assert "authorization" not in headers

    def test_missing_key_is_an_authentication_error(self, no_eden_key):
        with pytest.raises(litellm.AuthenticationError, match="EDENAI_API_KEY"):
            self._headers({})


class TestResponseTransformation:
    def test_eden_reported_cost_becomes_the_call_spend(self):
        logging_obj = _logging_obj()

        response = EdenAIAnthropicMessagesConfig().transform_anthropic_messages_response(
            model=SELLER_MODEL, raw_response=httpx.Response(200, json=_eden_message()), logging_obj=logging_obj
        )

        assert response["content"] == [{"type": "text", "text": "OK"}]
        assert response["cost"] == EDEN_REPORTED_COST
        assert logging_obj.model_call_details["response_cost"] == EDEN_REPORTED_COST

    def test_a_body_without_cost_leaves_pricing_to_the_price_map(self):
        logging_obj = _logging_obj()

        EdenAIAnthropicMessagesConfig().transform_anthropic_messages_response(
            model=SELLER_MODEL, raw_response=httpx.Response(200, json=_eden_message(cost=None)), logging_obj=logging_obj
        )

        assert "response_cost" not in logging_obj.model_call_details


class TestMessages:
    @pytest.mark.asyncio
    async def test_posts_the_anthropic_payload_untranslated_with_the_bearer_key(
        self, eden_key, httpx_transport, respx_mock
    ):
        respx_mock.post(EDEN_MESSAGES_URL).mock(return_value=httpx.Response(200, json=_eden_message()))

        response = await litellm.anthropic.messages.acreate(
            model=MODEL,
            max_tokens=16,
            messages=MESSAGES,
            system=[SYSTEM_BLOCK],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        assert response["content"] == [{"type": "text", "text": "OK"}]
        assert response["cost"] == EDEN_REPORTED_COST
        request = respx_mock.calls.last.request
        assert request.headers["authorization"] == f"Bearer {eden_key}"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = _request_body(respx_mock)
        assert (body["model"], body["messages"], body["max_tokens"]) == (SELLER_MODEL, MESSAGES, 16)
        assert body["system"] == [SYSTEM_BLOCK]
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}

    @pytest.mark.asyncio
    async def test_claude_code_billing_blocks_are_stripped_from_the_system_prompt(
        self, eden_key, httpx_transport, respx_mock
    ):
        respx_mock.post(EDEN_MESSAGES_URL).mock(return_value=httpx.Response(200, json=_eden_message()))

        await litellm.anthropic.messages.acreate(
            model=MODEL, max_tokens=16, messages=MESSAGES, system=[BILLING_BLOCK, SYSTEM_BLOCK]
        )

        assert _request_body(respx_mock)["system"] == [SYSTEM_BLOCK]

    @pytest.mark.asyncio
    async def test_eden_reported_cost_is_logged_as_the_call_spend(
        self, eden_key, httpx_transport, respx_mock, spend_capture
    ):
        respx_mock.post(EDEN_MESSAGES_URL).mock(return_value=httpx.Response(200, json=_eden_message()))
        await litellm.anthropic.messages.acreate(
            model=MODEL, max_tokens=16, messages=MESSAGES, litellm_call_id=spend_capture.call_id
        )
        await spend_capture.settle()

        assert spend_capture.costs == [EDEN_REPORTED_COST]


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_forwards_eden_events_verbatim(self, eden_key, httpx_transport, respx_mock):
        respx_mock.post(EDEN_MESSAGES_URL).mock(return_value=_eden_stream())

        stream = await litellm.anthropic.messages.acreate(model=MODEL, max_tokens=16, messages=MESSAGES, stream=True)
        body = b"".join([chunk async for chunk in stream]).decode()

        assert _request_body(respx_mock)["stream"] is True
        assert "event: message_start" in body
        assert '"text_delta", "text": "OK"' in body or '"text_delta","text":"OK"' in body
        assert "event: message_stop" in body


class TestErrors:
    @pytest.mark.asyncio
    async def test_401_detail_body_is_an_authentication_error(self, eden_key, httpx_transport, respx_mock):
        respx_mock.post(EDEN_MESSAGES_URL).mock(return_value=httpx.Response(401, json={"detail": "Invalid token"}))

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            await litellm.anthropic.messages.acreate(model=MODEL, max_tokens=16, messages=MESSAGES)
