"""`POST /v1/messages` served by Fireworks AI and Baseten through the chat completions bridge.

Neither provider has a native Messages config, so `litellm/llms/anthropic/messages/adapters/transformation.py`
turns the Anthropic request into an OpenAI chat completion and the answer back into an Anthropic
message. The request mapping: `system` becomes the first `system` message, `stop_sequences` becomes
`stop`, `metadata.user_id` becomes `user`, Anthropic tools become OpenAI functions, `tool_choice`
`any`/`auto`/`tool`/`none` becomes `required`/`auto`/a named function/`none` (Fireworks then sends
`required` as `any`), and `thinking` on a non-Claude model becomes `reasoning_effort`. The answer
mapping: finish `stop` to `end_turn`, `length` to `max_tokens`, `tool_calls` to `tool_use`, and
`usage.input_tokens = prompt_tokens - cached_tokens` with the cached part reported as
`cache_read_input_tokens`.

Fireworks also resolves a short model name to `accounts/fireworks/models/<name>`, asks for a usage
chunk on every stream (`stream_options.include_usage`), sends `x-session-affinity` only for a session
id the caller gave, can put the authenticated key's user in `user` (`fireworks_forward_user_id`), and
refuses images for a model whose cost map row says `supports_vision: false`. Both providers are
Python only (`litellm/messages/dispatch.py` marks CHAT_COMPLETIONS targets PYTHON_ONLY).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, Rig, anthropic_error, sse_events
from integration.messages_endpoint.dec1_msgprov_support import (
    BASETEN_BACKEND,
    BASETEN_KEY,
    CACHE_READ_COST,
    FIREWORKS_BACKEND,
    FIREWORKS_KEY,
    INPUT_COST,
    OUTPUT_COST,
    PNG_BASE64,
    PRICES,
    TOOL,
    at,
    chat_echo,
    chat_reply,
    chat_stream_reply,
    only_spend_row,
    openai_error_reply,
    path_of,
    provider_deployment,
    spend_of,
    unique_text,
)
from pydantic import JsonValue

_FIREWORKS_ECHO: Final = chat_echo(FIREWORKS_BACKEND)
_BASETEN_ECHO: Final = chat_echo(BASETEN_BACKEND)
_TOOL_CALL: Final[dict[str, JsonValue]] = {
    "id": "call_msgprov01",
    "type": "function",
    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})},
}


def _fireworks(
    gateway: Gateway,
    respond: Callable[[Request], Reply] = _FIREWORKS_ECHO,
    *,
    model: str = f"fireworks_ai/{FIREWORKS_BACKEND}",
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    return provider_deployment(gateway, respond, model=model, api_key=FIREWORKS_KEY, **params)


def _baseten(
    gateway: Gateway, respond: Callable[[Request], Reply] = _BASETEN_ECHO, **params: JsonValue
) -> AbstractContextManager[Rig]:
    return provider_deployment(gateway, respond, model=f"baseten/{BASETEN_BACKEND}", api_key=BASETEN_KEY, **params)


# ---------------------------------------------------------------- Fireworks: URL, auth and body


def test_fireworks_posts_chat_completions_with_the_bearer_key(gateway: Gateway) -> None:
    """A Fireworks deployment receives one `POST /chat/completions` under its `api_base` with
    `Authorization: Bearer <api_key>` and the resolved model id in the body."""
    with _fireworks(gateway) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        call: Final = rig.one_call()
        assert (call.method, path_of(call)) == ("POST", "/chat/completions")
        assert call.headers["authorization"] == f"Bearer {FIREWORKS_KEY}"
        assert JSON_OBJECT.validate_json(call.body)["model"] == FIREWORKS_BACKEND


def test_a_short_fireworks_model_name_is_resolved_to_the_account_path(gateway: Gateway) -> None:
    """`fireworks_ai/glm-5p3` is sent as `model: "accounts/fireworks/models/glm-5p3"`
    (`resolve_fireworks_resource_name`); Fireworks rejects the bare short name."""
    with _fireworks(gateway, model="fireworks_ai/glm-5p3") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert rig.one_body()["model"] == FIREWORKS_BACKEND


def test_the_fireworks_body_is_an_openai_chat_request(gateway: Gateway) -> None:
    """A request with `system`, `max_tokens: 64` and `stop_sequences` reaches Fireworks with the system
    message first, the user turn after it, `max_tokens: 64` and `stop: ["###"]`, and none of the
    Anthropic members (`system`, `stop_sequences`, `anthropic_version`) at the top level."""
    prompt: Final = unique_text()
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(prompt, system="You are terse.", stop_sequences=["###"])).status_code == 200
        body: Final = rig.one_body()
        assert body["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": prompt},
        ]
        assert (body["max_tokens"], body["stop"]) == (64, ["###"])
        assert not {"system", "stop_sequences", "anthropic_version"} & set(body), sorted(body)


def test_the_anthropic_user_id_becomes_the_openai_user(gateway: Gateway) -> None:
    """`metadata: {"user_id": "synthetic-user-01"}` is sent as `user: "synthetic-user-01"` and no
    `metadata` member reaches Fireworks."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(metadata={"user_id": "synthetic-user-01"})).status_code == 200
        body: Final = rig.one_body()
        assert body["user"] == "synthetic-user-01" and "metadata" not in body


def test_anthropic_tools_become_openai_functions_on_fireworks(gateway: Gateway) -> None:
    """The Anthropic tool becomes `{"type": "function", "function": {"name", "description",
    "parameters": <input_schema>}}`."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(tools=[TOOL])).status_code == 200
        tool: Final = object_value(at(rig.one_body(), "tools", 0))
        function: Final = object_value(tool["function"])
        assert (tool["type"], function["name"], function["description"], function["parameters"]) == (
            "function",
            TOOL["name"],
            TOOL["description"],
            TOOL["input_schema"],
        )


@pytest.mark.parametrize(
    ("tool_choice", "sent"),
    [
        ({"type": "any"}, "any"),
        ({"type": "auto"}, "auto"),
        ({"type": "tool", "name": "get_weather"}, {"type": "function", "function": {"name": "get_weather"}}),
    ],
    ids=["any", "auto", "tool"],
)
def test_fireworks_tool_choice_mapping(gateway: Gateway, tool_choice: dict[str, JsonValue], sent: JsonValue) -> None:
    """Anthropic `any` becomes OpenAI `required`, which Fireworks sends as `any`
    (`FireworksAIConfig.map_openai_params`); `auto` stays `auto`; a named tool becomes a named
    function. glm-5p3 has `supports_tool_choice: true` in the cost map."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(tools=[TOOL], tool_choice=tool_choice)).status_code == 200
        assert rig.one_body()["tool_choice"] == sent


def test_a_streaming_fireworks_request_asks_for_a_usage_chunk(gateway: Gateway) -> None:
    """With `stream: true` the Fireworks body carries `stream: true` and
    `stream_options: {"include_usage": true}`, without which the stream has no usage and the call is
    billed at zero."""
    with _fireworks(gateway) as rig:
        rig.stream(rig.body(stream=True))
        body: Final = rig.one_body()
        assert (body["stream"], body["stream_options"]) == (True, {"include_usage": True})


def test_disabled_thinking_is_sent_as_reasoning_effort_none(gateway: Gateway) -> None:
    """`thinking: {"type": "disabled"}` on a reasoning model becomes `reasoning_effort: "none"`."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(thinking={"type": "disabled"})).status_code == 200
        assert rig.one_body()["reasoning_effort"] == "none"


def test_an_enabled_thinking_budget_is_sent_as_a_reasoning_effort(gateway: Gateway) -> None:
    """`thinking: {"type": "enabled", "budget_tokens": 4096}` becomes `reasoning_effort: "high"` (the
    adapter's budget thresholds are 1024 low, 2048 medium, 4096 high) and no `thinking` member reaches
    Fireworks. UNVERIFIED: whether the threshold comparison puts exactly 4096 at `high`."""
    with _fireworks(gateway) as rig:
        thinking: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 4096}
        assert rig.post(rig.body(max_tokens=8192, thinking=thinking)).status_code == 200
        body: Final = rig.one_body()
        assert body["reasoning_effort"] == "high" and "thinking" not in body, body


def test_an_image_is_refused_for_a_fireworks_model_without_vision(gateway: Gateway) -> None:
    """glm-5p3 has `supports_vision: false` in the cost map, so an image block is answered with HTTP 400
    `invalid_request_error` saying the model does not support image inputs, and Fireworks is never
    called."""
    content: Final[list[JsonValue]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}},
        {"type": "text", "text": unique_text()},
    ]
    with _fireworks(gateway) as rig:
        response: Final = rig.post(rig.body(messages=[{"role": "user", "content": content}]))
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert "does not support image inputs" in str(error["message"])
        assert rig.calls() == ()


def test_cache_control_never_reaches_fireworks(gateway: Gateway) -> None:
    """A text block with `cache_control` reaches Fireworks without any `cache_control` member
    (`without_cache_control`), which an OpenAI-compatible server would reject."""
    content: Final[list[JsonValue]] = [{"type": "text", "text": unique_text(), "cache_control": {"type": "ephemeral"}}]
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(messages=[{"role": "user", "content": content}])).status_code == 200
        assert "cache_control" not in json.dumps(rig.one_body())


def test_a_caller_session_id_is_sent_as_x_session_affinity(gateway: Gateway) -> None:
    """`x-litellm-session-id: <id>` on the caller request is sent to Fireworks as
    `x-session-affinity: <id>`, so a session's turns hit one node and its prompt cache.
    UNVERIFIED: that the Messages route copies the header into `litellm_session_id`."""
    session: Final = f"session-{uuid.uuid4().hex}"
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body(), headers={"x-litellm-session-id": session}).status_code == 200
        assert rig.one_call().headers["x-session-affinity"] == session


def test_no_session_affinity_header_without_a_caller_session(gateway: Gateway) -> None:
    """Without a caller session id there is no `x-session-affinity` header. A per-request id the proxy
    generated would pin each request to a different node and defeat prompt caching
    (`get_fireworks_session_id`)."""
    with _fireworks(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert "x-session-affinity" not in rig.one_call().headers


def test_fireworks_forward_user_id_sends_the_key_owner_as_user(gateway: Gateway) -> None:
    """With `fireworks_forward_user_id: true` on the deployment, a call made with a key owned by a proxy
    user sends that user id as `user`, replacing the caller's `metadata.user_id`."""
    with _fireworks(gateway, fireworks_forward_user_id=True) as rig:
        owner: Final = rig.scenario.user()
        key: Final = rig.scenario.key(user_id=owner)
        assert rig.post(rig.body(metadata={"user_id": "caller-claimed"}), key=key).status_code == 200
        assert rig.one_body()["user"] == owner


# ---------------------------------------------------------------- Fireworks: response, stream, errors, spend


def test_a_fireworks_tool_call_becomes_an_anthropic_tool_use_block(gateway: Gateway) -> None:
    """A chat answer with one `tool_calls` entry and finish `tool_calls` becomes
    `content: [{"type": "tool_use", "id": "call_msgprov01", "name": "get_weather", "input": {"city":
    "Lisbon"}}]` with `stop_reason: tool_use`; the JSON string arguments are parsed into `input`."""
    reply: Final = chat_reply(
        "chatcmpl-msgprov", model=FIREWORKS_BACKEND, content=None, finish_reason="tool_calls", tool_calls=[_TOOL_CALL]
    )
    with _fireworks(gateway, lambda _r: reply) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body(tools=[TOOL])).content)
        assert payload["stop_reason"] == "tool_use"
        assert payload["content"] == [
            {"type": "tool_use", "id": "call_msgprov01", "name": "get_weather", "input": {"city": "Lisbon"}}
        ]


@pytest.mark.parametrize(("finish", "stop_reason"), [("stop", "end_turn"), ("length", "max_tokens")])
def test_the_fireworks_finish_reason_maps_to_the_anthropic_stop_reason(
    gateway: Gateway, finish: str, stop_reason: str
) -> None:
    """Finish `stop` becomes `end_turn` and `length` becomes `max_tokens`."""
    reply: Final = chat_reply("chatcmpl-msgprov", model=FIREWORKS_BACKEND, finish_reason=finish)
    with _fireworks(gateway, lambda _r: reply) as rig:
        assert JSON_OBJECT.validate_json(rig.post(rig.body()).content)["stop_reason"] == stop_reason


def test_fireworks_cached_tokens_are_reported_as_cache_reads(gateway: Gateway) -> None:
    """Chat usage `prompt_tokens: 20` with `prompt_tokens_details.cached_tokens: 8` becomes Anthropic
    `input_tokens: 12` and `cache_read_input_tokens: 8`, so input is not double counted."""
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_tokens_details": {"cached_tokens": 8},
    }
    reply: Final = chat_reply("chatcmpl-msgprov", model=FIREWORKS_BACKEND, usage=usage)
    with _fireworks(gateway, lambda _r: reply) as rig:
        answer: Final = object_value(JSON_OBJECT.validate_json(rig.post(rig.body()).content)["usage"])
        assert (answer["input_tokens"], answer["cache_read_input_tokens"], answer["output_tokens"]) == (12, 8, 5)


def test_the_fireworks_chat_stream_becomes_anthropic_sse(gateway: Gateway) -> None:
    """Fireworks chat chunks become an Anthropic stream: `message_start` first, `message_stop` last,
    `hello stream` across the text deltas, and `stop_reason: end_turn` with `output_tokens: 7` on
    `message_delta`."""
    with _fireworks(gateway) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        assert (events[0][0], events[-1][0]) == ("message_start", "message_stop"), text
        deltas: Final = [object_value(data["delta"]) for name, data in events if name == "content_block_delta"]
        assert "".join(str(delta.get("text", "")) for delta in deltas) == "hello stream"
        message_delta: Final = next(data for name, data in events if name == "message_delta")
        assert object_value(message_delta["delta"])["stop_reason"] == "end_turn"
        assert object_value(message_delta["usage"])["output_tokens"] == 7


@pytest.mark.parametrize(
    ("status", "anthropic_type"),
    [(400, "invalid_request_error"), (401, "authentication_error"), (429, "rate_limit_error")],
)
def test_a_fireworks_error_is_answered_in_the_anthropic_envelope(
    gateway: Gateway, status: int, anthropic_type: str
) -> None:
    """An OpenAI-shaped Fireworks error keeps its status and is answered in the Anthropic envelope with
    the matching type and the provider's message."""
    message: Final = f"synthetic fireworks {status}"
    with _fireworks(gateway, lambda _r: openai_error_reply(status, message)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == status, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == anthropic_type
        assert message in str(error["message"])


def test_the_fireworks_spend_row_prices_cached_tokens(gateway: Gateway) -> None:
    """20 prompt tokens with 8 cached and 5 completion tokens are billed as
    `12 * 0.001 + 8 * 0.0001 + 5 * 0.002 = 0.0228` on a row with `custom_llm_provider = "fireworks_ai"`
    and `call_type = "anthropic_messages"`."""
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_tokens_details": {"cached_tokens": 8},
    }
    reply: Final = chat_reply("chatcmpl-msgprov", model=FIREWORKS_BACKEND, usage=usage)
    with _fireworks(gateway, lambda _r: reply, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"]) == ("anthropic_messages", "fireworks_ai")
        assert spend_of(row) == pytest.approx(12 * INPUT_COST + 8 * CACHE_READ_COST + 5 * OUTPUT_COST)


def test_a_streamed_fireworks_call_is_billed_from_the_usage_chunk(gateway: Gateway) -> None:
    """A streamed call whose usage chunk reports 11 and 7 tokens writes one row with spend 0.025."""
    with _fireworks(gateway, **PRICES) as rig:
        rig.stream(rig.body(stream=True))
        assert spend_of(only_spend_row(rig)) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_fireworks_count_tokens_is_answered_without_a_provider_call(gateway: Gateway) -> None:
    """Fireworks has no count tokens API, so `POST /v1/messages/count_tokens` is answered from a local
    estimate: HTTP 200 with a positive integer `input_tokens`, and Fireworks is never called.
    UNVERIFIED: that no provider count is attempted for `fireworks_ai`."""
    with _fireworks(gateway) as rig:
        body: Final[dict[str, JsonValue]] = {"model": rig.model, "messages": [{"role": "user", "content": unique_text()}]}
        response: Final = rig.post(body, path="/v1/messages/count_tokens")
        assert response.status_code == 200, response.text
        count: Final = JSON_OBJECT.validate_json(response.content)["input_tokens"]
        assert isinstance(count, int) and count > 0
        assert rig.calls() == ()


# ---------------------------------------------------------------- Baseten


def test_baseten_posts_chat_completions_with_the_bearer_key(gateway: Gateway) -> None:
    """A Baseten deployment with an `api_base` receives `POST /chat/completions` with
    `Authorization: Bearer <api_key>` and `model: "zai-org/GLM-4.6"`."""
    with _baseten(gateway) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        call: Final = rig.one_call()
        assert (call.method, path_of(call)) == ("POST", "/chat/completions")
        assert call.headers["authorization"] == f"Bearer {BASETEN_KEY}"
        assert JSON_OBJECT.validate_json(call.body)["model"] == BASETEN_BACKEND


def test_the_baseten_body_is_an_openai_chat_request(gateway: Gateway) -> None:
    """System first, then the user turn, `max_tokens: 64`, and `stop_sequences` sent as `stop`."""
    prompt: Final = unique_text()
    with _baseten(gateway) as rig:
        assert rig.post(rig.body(prompt, system="You are terse.", stop_sequences=["###"])).status_code == 200
        body: Final = rig.one_body()
        assert body["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": prompt},
        ]
        assert (body["max_tokens"], body["stop"]) == (64, ["###"])
        assert "stop_sequences" not in body


def test_baseten_tool_choice_any_is_sent_as_required(gateway: Gateway) -> None:
    """Without a Fireworks-style override, Anthropic `tool_choice: any` reaches Baseten as the OpenAI
    `required`. UNVERIFIED: that the Baseten config passes `tool_choice` through for GLM-4.6."""
    with _baseten(gateway) as rig:
        assert rig.post(rig.body(tools=[TOOL], tool_choice={"type": "any"})).status_code == 200
        assert rig.one_body()["tool_choice"] == "required"


def test_a_baseten_tool_call_becomes_an_anthropic_tool_use_block(gateway: Gateway) -> None:
    """A Baseten `tool_calls` answer becomes one `tool_use` block with parsed `input` and
    `stop_reason: tool_use`."""
    reply: Final = chat_reply(
        "chatcmpl-msgprov", model=BASETEN_BACKEND, content=None, finish_reason="tool_calls", tool_calls=[_TOOL_CALL]
    )
    with _baseten(gateway, lambda _r: reply) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body(tools=[TOOL])).content)
        assert payload["stop_reason"] == "tool_use"
        assert at(payload, "content", 0, "input") == {"city": "Lisbon"}


def test_the_baseten_stream_becomes_anthropic_sse(gateway: Gateway) -> None:
    """Baseten chat chunks become an Anthropic stream that starts with `message_start`, ends with
    `message_stop` and carries `hello stream`."""
    reply: Final = chat_stream_reply("chatcmpl-msgprov", model=BASETEN_BACKEND)
    with _baseten(gateway, lambda _r: reply) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        assert (events[0][0], events[-1][0]) == ("message_start", "message_stop"), text
        deltas: Final = [object_value(data["delta"]) for name, data in events if name == "content_block_delta"]
        assert "".join(str(delta.get("text", "")) for delta in deltas) == "hello stream"


def test_a_baseten_503_is_answered_as_an_anthropic_error(gateway: Gateway) -> None:
    """A Baseten 503 is answered with HTTP 503 and an Anthropic envelope whose type is `api_error`
    (Anthropic has no service unavailable type). UNVERIFIED: the type chosen for 503."""
    with _baseten(gateway, lambda _r: openai_error_reply(503, "synthetic baseten unavailable", "server_error")) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 503, response.text
        assert anthropic_error(response)["type"] == "api_error"


def test_the_baseten_spend_row_names_baseten(gateway: Gateway) -> None:
    """One Baseten call writes one `anthropic_messages` row with `custom_llm_provider = "baseten"` and
    spend 0.025 from the deployment prices."""
    with _baseten(gateway, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"]) == ("anthropic_messages", "baseten")
        assert spend_of(row) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)

