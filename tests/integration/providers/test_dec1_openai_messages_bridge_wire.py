"""Contract: `POST /v1/messages` (Anthropic format) served by OpenAI and the OpenAI-compatible providers.

Claude Code and the Anthropic SDK can point at the proxy and pick any model, so `/v1/messages` must work when the
deployment is not Anthropic. `litellm/llms/anthropic/messages/native/handler.py` picks one of three paths:

- `openai/` deployments go through the Responses API bridge (`_RESPONSES_API_PROVIDERS = {"openai"}`): the
  provider receives `POST <api_base>/responses` with `max_tokens` as `max_output_tokens` and `system` as
  `instructions`, except when the caller sends `stop_sequences`, which Responses cannot express, and then the
  chat completions bridge is used instead (`_should_route_to_responses_api`);
- DeepSeek has an Anthropic-compatible endpoint, so `deepseek/` deployments are served natively
  (`DeepSeekAnthropicMessagesConfig`): `POST <host>/anthropic/v1/messages` with `x-api-key` and
  `anthropic-version: 2023-06-01`, whatever `/v1` or `/beta` suffix the configured base had;
- every other OpenAI-compatible provider (Groq, xAI, Mistral, OpenRouter, Together, hosted vLLM) goes through the
  chat completions adapter (`anthropic/messages/adapters/transformation.py`): `system` becomes the first system
  message, `max_tokens` stays `max_tokens` (then each provider's own chat mapping applies), and the chat answer
  becomes an Anthropic `message` with `stop_reason` mapped from the finish reason.

All of these are Python-only today (`litellm/messages/dispatch.py`), and #51119 / #51721 just moved Anthropic and
Azure Foundry Messages to Rust, so these are next. Fireworks and Baseten bridges are already pinned in
`messages_endpoint/test_dec1_msgprov_openai_compatible.py` and are not repeated.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    CALL_COST,
    PROVIDER_KEY,
    REPLY_TEXT,
    Call,
    chat_json,
    data_events,
    deployment,
    json_reply,
    marker,
    messages,
    openai_error,
    peer,
    served,
    spend_row_by_group,
    stream_lines,
)
from pydantic import JsonValue

_CHAT_BRIDGED: Final = (
    ("groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
    ("xai/grok-3", "grok-3"),
    ("mistral/mistral-large-latest", "mistral-large-latest"),
    ("openrouter/openai/gpt-4o-mini", "openai/gpt-4o-mini"),
    ("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("hosted_vllm/dec1-served-model", "dec1-served-model"),
)
_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def test_an_openai_deployment_is_bridged_to_the_responses_api(gateway: Gateway) -> None:
    """A `/v1/messages` call with `system: "be brief"` and `max_tokens: 64` on `openai/gpt-4o-mini`.

    The provider receives `POST /v1/responses` (not chat completions) whose body has `model: "gpt-4o-mini"`,
    `max_output_tokens: 64` and `instructions: "be brief"`. The caller gets an Anthropic message
    (`type: "message"`, `role: "assistant"`) whose first text block is the provider's reply.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-4o-mini")
        body: Final = served(messages(gateway, model, system="be brief"))
        call: Final = provider.only()
        assert call.path == "/v1/responses", call.path
        assert (call.body["model"], call.body["max_output_tokens"], call.body["instructions"]) == (
            "gpt-4o-mini",
            64,
            "be brief",
        ), call.body
        assert (body["type"], body["role"]) == ("message", "assistant"), body
        assert object_value(list_value(body["content"])[0]) == {"type": "text", "text": REPLY_TEXT}, body


def test_stop_sequences_send_an_openai_deployment_through_chat_completions(gateway: Gateway) -> None:
    """The same OpenAI deployment with `stop_sequences: ["END"]`.

    Responses has no stop parameter, so the call is bridged to `POST /v1/chat/completions` with `stop: ["END"]`
    instead (`_should_route_to_responses_api` returns false when stop sequences are present).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-4o-mini")
        served(messages(gateway, model, stop_sequences=["END"]))
        call: Final = provider.only()
        assert call.path == "/v1/chat/completions", call.path
        assert call.body["stop"] == ["END"], call.body


def test_the_openai_responses_bridge_reports_anthropic_usage(gateway: Gateway) -> None:
    """The Responses answer reports 11 input and 7 output tokens.

    The caller's Anthropic `usage` has `input_tokens: 11` and `output_tokens: 7`, and `stop_reason` is
    `end_turn` for a completed response.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-4o-mini")
        body: Final = served(messages(gateway, model))
        reported: Final = object_value(body["usage"])
        assert (reported["input_tokens"], reported["output_tokens"]) == (11, 7), body
        assert body["stop_reason"] == "end_turn", body


def test_the_openai_responses_bridge_stream_is_anthropic_sse(gateway: Gateway) -> None:
    """The caller streams; the provider streams Responses events.

    The caller receives Anthropic SSE: the first event is `message_start`, the last is `message_stop`, and the
    `text_delta` pieces join to `dec1 openai reply`. #51305 (restore sync messages streams for OpenAI) touched
    exactly this path.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "openai/gpt-4o-mini")
        _, lines = stream_lines(
            gateway,
            "/v1/messages",
            {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": marker()}], "stream": True},
        )
        events: Final = data_events(lines)
        assert (events[0]["type"], events[-1]["type"]) == ("message_start", "message_stop"), events
        text: Final = "".join(
            string_value(object_value(event["delta"]).get("text") or "")
            for event in events
            if event["type"] == "content_block_delta"
        )
        assert text == REPLY_TEXT, events
        assert provider.only().body["stream"] is True


@pytest.mark.parametrize(("model_path", "upstream"), _CHAT_BRIDGED)
def test_each_compatible_provider_is_bridged_to_chat_completions(
    gateway: Gateway, model_path: str, upstream: str
) -> None:
    """A `/v1/messages` call with `system: "be brief"` on each OpenAI-compatible deployment.

    The provider receives `POST /v1/chat/completions` with `authorization: Bearer <deployment key>`, the upstream
    model, the system text as the first message (`{"role": "system", ...}`) and the user turn last. The caller gets
    an Anthropic message with the reply text and `stop_reason: "end_turn"`.
    """
    text: Final = marker()
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, model_path)
        body: Final = served(messages(gateway, model, system="be brief", messages=[{"role": "user", "content": text}]))
        call: Final = provider.only()
        assert (call.path, call.headers["authorization"]) == ("/v1/chat/completions", f"Bearer {PROVIDER_KEY}")
        assert call.body["model"] == upstream, call.body
        sent_messages: Final = list_value(call.body["messages"])
        assert object_value(sent_messages[0])["role"] == "system", sent_messages
        assert "be brief" in str(object_value(sent_messages[0])["content"]), sent_messages
        assert object_value(sent_messages[-1])["role"] == "user", sent_messages
        assert body["stop_reason"] == "end_turn", body
        assert object_value(list_value(body["content"])[0])["text"] == REPLY_TEXT, body


@pytest.mark.parametrize(("model_path", "upstream"), _CHAT_BRIDGED)
def test_each_bridged_call_is_billed_once_as_anthropic_messages(
    gateway: Gateway, model_path: str, upstream: str
) -> None:
    """One bridged call with 11 prompt and 7 completion tokens at 0.001 and 0.002 writes one SpendLogs row with
    `call_type = 'anthropic_messages'`, `spend = 0.025` and the provider prefix in `custom_llm_provider`.
    UNVERIFIED: the call type string for bridged calls, taken from the msgprov suite's Fireworks row."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, model_path)
        served(messages(gateway, model))
        row: Final = spend_row_by_group(gateway, model)
        assert row["call_type"] == "anthropic_messages", row
        assert float(str(row["spend"])) == pytest.approx(CALL_COST), row
        assert row["custom_llm_provider"] == model_path.split("/")[0], row


def test_mistral_bridge_sends_tool_choice_any_and_strips_nothing_else(gateway: Gateway) -> None:
    """`tool_choice: {"type": "any"}` with one Anthropic tool on a Mistral deployment.

    The adapter turns `any` into OpenAI `required`, and Mistral's chat mapping turns that into `"any"`, so the
    provider body has `tool_choice: "any"` and one function tool named `get_weather` whose `parameters` is the
    `input_schema`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "mistral/mistral-large-latest")
        served(messages(gateway, model, tools=[_TOOL], tool_choice={"type": "any"}))
        sent: Final = provider.only().body
        assert sent["tool_choice"] == "any", sent
        function: Final = object_value(object_value(list_value(sent["tools"])[0])["function"])
        assert (function["name"], function["parameters"]) == ("get_weather", _TOOL["input_schema"]), sent


def test_groq_bridge_tool_choice_any_is_required(gateway: Gateway) -> None:
    """On Groq, which has no tool_choice override, Anthropic `tool_choice: {"type": "any"}` reaches the provider
    as the OpenAI `"required"`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "groq/llama-3.3-70b-versatile")
        served(messages(gateway, model, tools=[_TOOL], tool_choice={"type": "any"}))
        assert provider.only().body["tool_choice"] == "required"


def test_xai_bridge_sends_max_tokens_and_no_anthropic_fields(gateway: Gateway) -> None:
    """A bridged xAI call with `max_tokens: 64` and `metadata: {"user_id": "dec1-user"}`.

    The provider body has `max_tokens: 64`, `user: "dec1-user"` (xAI lists `user`), and none of the Anthropic-only
    members `system`, `metadata`, `stop_sequences` or `anthropic_version` at the top level.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "xai/grok-3")
        served(messages(gateway, model, metadata={"user_id": "dec1-user"}))
        sent: Final = provider.only().body
        assert (sent["max_tokens"], sent.get("user")) == (64, "dec1-user"), sent
        assert not {"system", "metadata", "stop_sequences", "anthropic_version"} & set(sent), sent


def test_a_bridged_tool_call_becomes_a_tool_use_block(gateway: Gateway) -> None:
    """Groq answers one tool call `call_dec1bridge` to `get_weather` with arguments `{"city": "Oslo"}` and finish
    `tool_calls`.

    The caller gets `stop_reason: "tool_use"` and a `tool_use` block with `id: "call_dec1bridge"`, `name:
    "get_weather"` and `input: {"city": "Oslo"}` parsed into an object.
    """
    tool_call: Final[dict[str, JsonValue]] = {
        "id": "call_dec1bridge",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
    }

    def respond(call: Call) -> Reply:
        return json_reply(
            chat_json(call.body, content=None, finish_reason="tool_calls", extra_message={"tool_calls": [tool_call]})
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "groq/llama-3.3-70b-versatile")
        body: Final = served(messages(gateway, model, tools=[_TOOL]))
        assert body["stop_reason"] == "tool_use", body
        block: Final = next(
            object_value(b) for b in list_value(body["content"]) if object_value(b)["type"] == "tool_use"
        )
        assert (block["id"], block["name"], block["input"]) == ("call_dec1bridge", "get_weather", {"city": "Oslo"}), (
            block
        )


def test_a_length_finish_becomes_max_tokens(gateway: Gateway) -> None:
    """The bridged provider answers `finish_reason: "length"`; the caller's `stop_reason` is `max_tokens`."""

    def respond(call: Call) -> Reply:
        return json_reply(chat_json(call.body, finish_reason="length"))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")
        assert served(messages(gateway, model))["stop_reason"] == "max_tokens"


def test_a_bridged_provider_429_is_an_anthropic_rate_limit_error(gateway: Gateway) -> None:
    """The bridged Groq provider answers 429 `dec1 groq throttled`.

    The caller gets HTTP 429 with the Anthropic error envelope `{"type": "error", "error": {"type":
    "rate_limit_error", "message": ...}}` containing the provider text, not the OpenAI envelope.
    """
    with peer(lambda call: openai_error(429, "dec1 groq throttled")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "groq/llama-3.3-70b-versatile")
        response: Final = messages(gateway, model)
        assert response.status_code == 429, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["type"] == "error", body
        assert object_value(body["error"])["type"] == "rate_limit_error", body
        assert "dec1 groq throttled" in string_value(object_value(body["error"])["message"]), body


def test_the_bridged_chat_stream_is_anthropic_sse(gateway: Gateway) -> None:
    """A streamed `/v1/messages` call on hosted vLLM: the provider streams chat chunks with a usage chunk.

    The caller receives `message_start` first and `message_stop` last, text deltas joining to `dec1 openai
    reply`, and a `message_delta` whose `usage.output_tokens` is 7. The provider body asked for
    `stream_options.include_usage: true`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "hosted_vllm/dec1-served-model")
        _, lines = stream_lines(
            gateway,
            "/v1/messages",
            {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": marker()}], "stream": True},
        )
        events: Final = data_events(lines)
        assert (events[0]["type"], events[-1]["type"]) == ("message_start", "message_stop"), events
        text: Final = "".join(
            string_value(object_value(e["delta"]).get("text") or "")
            for e in events
            if e["type"] == "content_block_delta"
        )
        assert text == REPLY_TEXT
        delta: Final = next(e for e in events if e["type"] == "message_delta")
        assert object_value(delta["usage"])["output_tokens"] == 7, delta
        assert object_value(provider.only().body["stream_options"]) == {"include_usage": True}


@pytest.mark.parametrize("suffix", ["/v1", "/beta", "/anthropic", ""])
def test_deepseek_is_served_natively_at_its_anthropic_endpoint(gateway: Gateway, suffix: str) -> None:
    """A `deepseek/deepseek-chat` deployment with an `api_base` of `<peer><suffix>`.

    `deepseek_anthropic_base` strips `/v1` or `/beta`, keeps an existing `/anthropic`, and appends `anthropic`
    otherwise, so in every case the provider receives `POST /anthropic/v1/messages`. The body is native Anthropic
    (`messages` and `max_tokens: 64` as sent, `model: "deepseek-chat"`), not a chat completion.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "deepseek/deepseek-chat", api_base_suffix=suffix)
        served(messages(gateway, model))
        call: Final = provider.only()
        assert call.path == "/anthropic/v1/messages", call.path
        assert (call.body["model"], call.body["max_tokens"]) == ("deepseek-chat", 64), call.body
        assert "choices" not in call.body and isinstance(call.body["messages"], list)


def test_deepseek_native_messages_use_x_api_key_and_the_default_version(gateway: Gateway) -> None:
    """The DeepSeek Anthropic endpoint authenticates with `x-api-key: <deployment key>` and gets
    `anthropic-version: 2023-06-01` (filled only when absent), and the caller's proxy key never appears in
    `authorization` upstream. UNVERIFIED: whether the caller's own `anthropic-version` header (sent by the helper
    as 2023-06-01) is forwarded; both give the same value here."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "deepseek/deepseek-chat")
        served(messages(gateway, model))
        headers: Final = provider.only().headers
        assert headers["x-api-key"] == PROVIDER_KEY, dict(headers)
        assert headers["anthropic-version"] == "2023-06-01", dict(headers)
        assert gateway.key not in headers.get("authorization", ""), dict(headers)


def test_deepseek_native_messages_drop_the_custom_tool_type(gateway: Gateway) -> None:
    """A tool sent as `{"type": "custom", "name": "get_weather", ...}` reaches DeepSeek without the `type` key, since
    DeepSeek rejects Anthropic's explicit custom discriminator (`sanitize_tools_for_deepseek`); name and
    `input_schema` stay."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "deepseek/deepseek-chat")
        served(messages(gateway, model, tools=[{"type": "custom", **_TOOL}]))
        tool: Final = object_value(list_value(provider.only().body["tools"])[0])
        assert "type" not in tool and tool["name"] == "get_weather", tool
