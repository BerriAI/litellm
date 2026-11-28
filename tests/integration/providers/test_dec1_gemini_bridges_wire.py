"""`/v1/messages` and `/v1/responses` served by a `gemini/` (Google AI Studio) deployment.

Neither route has a native Gemini implementation. `/v1/messages` goes through the Anthropic to chat
adapter (`litellm/llms/anthropic/messages/adapters`) and then the AI Studio chat path; `/v1/responses`
goes through the Responses to chat bridge. So a Claude Code or Codex style client pointed at a
`gemini/` model must still produce one `POST {api_base}/models/{model}:generateContent` (or
`:streamGenerateContent?alt=sse`) with the key header, and must get its own wire format back: an
Anthropic `message` with `stop_reason` mapped from the OpenAI finish reason (`stop` -> `end_turn`,
`length` -> `max_tokens`, `tool_calls` -> `tool_use`, `content_filter` -> `refusal`), or a Responses
`response` object. Anthropic's `web_search_*` server tool becomes `web_search_options` and so Google
Search grounding. `/v1/messages` is one of the routes being moved to Rust next, so these are the
bridge contracts the port has to keep for AI Studio. The Vertex suite pins the Responses field mapping
on Vertex; only the AI Studio specific parts are repeated here.
"""

from __future__ import annotations

import json
from typing import Final

from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_gemini_support import (
    FLASH,
    GEMINI_KEY,
    at,
    gemini,
    gemini_echo,
    gemini_reply,
    generate_path,
    headers_of,
    json_of,
    marker,
    path_of,
    query_of,
    sse_payloads,
    usage_metadata,
)
from pydantic import JsonValue


def _messages_body(text: str | None = None, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "max_tokens": 256,
        "messages": [{"role": "user", "content": text if text is not None else f"hi {marker()}"}],
        **fields,
    }


def _blocks(response_body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [object_value(block) for block in list_value(response_body["content"])]


# ---------------------------------------------------------------- /v1/messages


def test_messages_reaches_generate_content_with_the_key_header(gateway: Gateway) -> None:
    """`POST /v1/messages` on a gemini deployment sends exactly one
    `POST /v1beta/models/gemini-2.5-flash:generateContent` with `x-goog-api-key` and no `x-api-key` or
    `anthropic-version` header (those are Anthropic's, never Google's)."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.post("/v1/messages", _messages_body())
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert path_of(call) == generate_path(FLASH)
        headers: Final = headers_of(call)
        assert headers.get("x-goog-api-key") == GEMINI_KEY
        assert "x-api-key" not in headers and "anthropic-version" not in headers


def test_messages_answer_is_an_anthropic_message(gateway: Gateway) -> None:
    """The Gemini text "pong" with `STOP` comes back as `{"type": "message", "role": "assistant", "content":
    [{"type": "text", "text": "pong"}], "stop_reason": "end_turn"}` with an `id` and the model name."""
    with gemini(gateway, gemini_echo) as rig:
        body: Final = json_of(rig.post("/v1/messages", _messages_body()))
        assert (body["type"], body["role"], body["stop_reason"]) == ("message", "assistant", "end_turn")
        assert [(block["type"], block.get("text")) for block in _blocks(body)] == [("text", "pong")]
        assert body.get("id")


def test_messages_system_and_max_tokens_reach_gemini(gateway: Gateway) -> None:
    """Anthropic's top-level `system: "Be terse"` becomes `system_instruction` with that text and
    `max_tokens: 256` becomes `generationConfig.max_output_tokens: 256`."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.post("/v1/messages", _messages_body(system="Be terse")).status_code == 200
        body: Final = rig.peer.one_body()
        assert "Be terse" in json.dumps(body["system_instruction"])
        assert at(body, "generationConfig", "max_output_tokens") == 256


def test_messages_max_tokens_finish_is_max_tokens(gateway: Gateway) -> None:
    """Gemini `finishReason: MAX_TOKENS` -> OpenAI `length` -> Anthropic `stop_reason: "max_tokens"`."""
    with gemini(gateway, lambda _r: gemini_reply(finish="MAX_TOKENS")) as rig:
        assert json_of(rig.post("/v1/messages", _messages_body()))["stop_reason"] == "max_tokens"


def test_messages_safety_finish_is_refusal(gateway: Gateway) -> None:
    """Gemini `finishReason: SAFETY` -> OpenAI `content_filter` -> Anthropic `stop_reason: "refusal"`."""
    with gemini(gateway, lambda _r: gemini_reply(finish="SAFETY", text="")) as rig:
        assert json_of(rig.post("/v1/messages", _messages_body()))["stop_reason"] == "refusal"


def test_messages_tools_round_trip_as_tool_use(gateway: Gateway) -> None:
    """An Anthropic tool `get_weather` reaches Gemini as a function declaration; Gemini's `functionCall`
    `{"name": "get_weather", "args": {"city": "Paris"}}` comes back as a `tool_use` block with that name and
    `input: {"city": "Paris"}`, and `stop_reason: "tool_use"`."""
    tool: Final[dict[str, JsonValue]] = {
        "name": "get_weather",
        "description": "Weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    call_part: Final[list[JsonValue]] = [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]
    with gemini(gateway, lambda _r: gemini_reply(parts=call_part)) as rig:
        body: Final = json_of(rig.post("/v1/messages", _messages_body(tools=[tool])))
        assert "get_weather" in json.dumps(rig.peer.one_body()["tools"])
        tool_use: Final = [block for block in _blocks(body) if block["type"] == "tool_use"]
        assert len(tool_use) == 1
        assert (tool_use[0]["name"], tool_use[0]["input"]) == ("get_weather", {"city": "Paris"})
        assert body["stop_reason"] == "tool_use"


def test_messages_usage_uses_input_and_output_tokens(gateway: Gateway) -> None:
    """`usageMetadata` 11 prompt and 7 candidate tokens come back as `usage.input_tokens: 11` and
    `usage.output_tokens: 7`."""
    with gemini(gateway, gemini_echo) as rig:
        usage: Final = object_value(json_of(rig.post("/v1/messages", _messages_body()))["usage"])
        assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def test_messages_thinking_budget_reaches_gemini_2_5(gateway: Gateway) -> None:
    """Anthropic `thinking: {"type": "enabled", "budget_tokens": 2048}` reaches `gemini-2.5-flash` as
    `generationConfig.thinkingConfig.thinkingBudget: 2048`."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.post(
            "/v1/messages", _messages_body(max_tokens=4096, thinking={"type": "enabled", "budget_tokens": 2048})
        )
        assert response.status_code == 200, response.text
        assert at(rig.peer.one_body(), "generationConfig", "thinkingConfig", "thinkingBudget") == 2048


def test_messages_thought_parts_come_back_as_thinking_blocks(gateway: Gateway) -> None:
    """A `thought: true` part "weighing options" before the text "done" comes back as a `thinking` block
    holding "weighing options" followed by a `text` block "done"."""
    parts: Final[list[JsonValue]] = [{"text": "weighing options", "thought": True}, {"text": "done"}]
    with gemini(gateway, lambda _r: gemini_reply(parts=parts)) as rig:
        blocks: Final = _blocks(json_of(rig.post("/v1/messages", _messages_body())))
        assert [block["type"] for block in blocks] == ["thinking", "text"]
        assert blocks[0]["thinking"] == "weighing options"


def test_messages_web_search_tool_becomes_google_search(gateway: Gateway) -> None:
    """Anthropic's server tool `{"type": "web_search_20250305", "name": "web_search"}` is turned into
    `web_search_options` by the adapter and so reaches Gemini as `tools: [{"googleSearch": {}}]`."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.post(
            "/v1/messages", _messages_body(tools=[{"type": "web_search_20250305", "name": "web_search"}])
        )
        assert response.status_code == 200, response.text
        assert rig.peer.one_body()["tools"] == [{"googleSearch": {}}]


def test_messages_stream_uses_alt_sse_and_answers_anthropic_events(gateway: Gateway) -> None:
    """`stream: true` on `/v1/messages` posts to `:streamGenerateContent?alt=sse` and the caller receives
    Anthropic SSE: the first event is `message_start`, text arrives in `content_block_delta` events
    joining to "hello stream", and the last is `message_stop`."""
    with gemini(gateway, gemini_echo) as rig:
        response, text = rig.stream("/v1/messages", _messages_body())
        assert response.status_code == 200, text
        call: Final = rig.peer.one_call()
        assert path_of(call) == generate_path(FLASH, method="streamGenerateContent")
        assert query_of(call) == "alt=sse"
        events: Final = sse_payloads(text)
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
        joined: Final = "".join(
            str(object_value(event["delta"]).get("text", ""))
            for event in events
            if event["type"] == "content_block_delta"
        )
        assert joined == "hello stream"


def test_messages_google_429_is_an_anthropic_rate_limit_error(gateway: Gateway) -> None:
    """A Google 429 on the bridged route answers 429 with Anthropic's error envelope
    `{"type": "error", "error": {"type": "rate_limit_error", ...}}`. UNVERIFIED: the exact error `type`
    string the proxy writes for a bridged provider error."""
    from integration.providers.dec1_gemini_support import google_error

    with gemini(gateway, lambda _r: google_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded for metric")) as rig:
        response: Final = rig.post("/v1/messages", _messages_body())
        assert response.status_code == 429, response.text
        assert "rate_limit" in response.text


# ---------------------------------------------------------------- /v1/responses


def test_responses_reaches_generate_content_with_the_key_header(gateway: Gateway) -> None:
    """`POST /v1/responses` with `input: "<text>"` on the gemini deployment sends one
    `POST /v1beta/models/gemini-2.5-flash:generateContent` with `x-goog-api-key`; there is no native
    Responses call to Google."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.post("/v1/responses", {"input": f"hi {marker()}"})
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert path_of(call) == generate_path(FLASH)
        assert headers_of(call).get("x-goog-api-key") == GEMINI_KEY


def test_responses_answer_is_a_completed_response_object(gateway: Gateway) -> None:
    """The answer is `{"object": "response", "status": "completed"}` whose output holds one `message` item
    with an `output_text` part "pong", and `usage` with `input_tokens: 11`, `output_tokens: 7`."""
    with gemini(gateway, gemini_echo) as rig:
        body: Final = json_of(rig.post("/v1/responses", {"input": f"hi {marker()}"}))
        assert (body["object"], body["status"]) == ("response", "completed")
        messages: Final = [
            object_value(item) for item in list_value(body["output"]) if object_value(item)["type"] == "message"
        ]
        assert "pong" in json.dumps(messages[0]["content"])
        assert (at(body, "usage", "input_tokens"), at(body, "usage", "output_tokens")) == (11, 7)


def test_responses_web_search_tool_becomes_google_search(gateway: Gateway) -> None:
    """A Responses tool `{"type": "web_search_preview"}` reaches Gemini as `{"googleSearch": {}}`."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.post(
            "/v1/responses", {"input": f"news {marker()}", "tools": [{"type": "web_search_preview"}]}
        )
        assert response.status_code == 200, response.text
        assert {"googleSearch": {}} in list_value(rig.peer.one_body()["tools"])


def test_responses_reasoning_effort_is_a_thinking_level_on_gemini_3(gateway: Gateway) -> None:
    """`reasoning: {"effort": "high"}` on a Gemini 3 model reaches Google as
    `thinkingConfig.thinkingLevel: "high"` (the Vertex suite pins the 2.5 budget form)."""
    with gemini(gateway, gemini_echo, backend="gemini-3-flash-preview") as rig:
        response: Final = rig.post("/v1/responses", {"input": f"hi {marker()}", "reasoning": {"effort": "high"}})
        assert response.status_code == 200, response.text
        assert at(rig.peer.one_body(), "generationConfig", "thinkingConfig", "thinkingLevel") == "high"


def test_responses_stream_uses_alt_sse_and_ends_completed(gateway: Gateway) -> None:
    """A streamed Responses call posts to `:streamGenerateContent?alt=sse` and the last event the caller sees
    is `response.completed`."""
    with gemini(gateway, gemini_echo) as rig:
        response, text = rig.stream("/v1/responses", {"input": f"hi {marker()}"})
        assert response.status_code == 200, text
        assert query_of(rig.peer.one_call()) == "alt=sse"
        assert sse_payloads(text)[-1]["type"] == "response.completed"


def test_responses_cached_tokens_are_reported_as_input_token_details(gateway: Gateway) -> None:
    """A Gemini usage with `cachedContentTokenCount: 6` inside 11 prompt tokens comes back as
    `usage.input_tokens_details.cached_tokens: 6`."""
    usage: Final = usage_metadata(11, 7, cachedContentTokenCount=6)
    with gemini(gateway, lambda _r: gemini_reply(usage=usage)) as rig:
        body: Final = json_of(rig.post("/v1/responses", {"input": f"hi {marker()}"}))
        assert at(body, "usage", "input_tokens_details", "cached_tokens") == 6
