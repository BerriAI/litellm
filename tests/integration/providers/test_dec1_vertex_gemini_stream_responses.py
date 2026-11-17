"""Streaming chat and `/v1/responses` served by a Vertex AI Gemini deployment.

Streaming chat goes to `:streamGenerateContent` with the query `alt=sse` (without it Vertex answers
one JSON array instead of server-sent events), and each Gemini frame becomes one or more OpenAI
`chat.completion.chunk` frames ending with `data: [DONE]`; usage arrives on the final chunk when the
caller asks for it with `stream_options.include_usage`. `/v1/responses` has no native Vertex config
(`ProviderConfigManager.get_provider_responses_api_config` returns nothing for `vertex_ai`), so it
runs through the chat bridge: `instructions` becomes `system_instruction`, `max_output_tokens`
becomes `generationConfig.max_output_tokens`, and the Gemini answer is rebuilt as a Responses
object with `output[].content[].output_text` and Responses-style usage.

Both paths are Python today. They are pinned here because a Rust HTTP front door has to keep the
exact SSE framing, the `alt=sse` query and the usage mapping, and because the responses bridge is
one of the first things a port tends to shortcut.
"""

from __future__ import annotations

import json
from typing import Final

from integration._support.client import Gateway, list_value, object_value, string_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_vertex_support import (
    at,
    gemini_echo,
    gemini_reply,
    gemini_sse,
    json_of,
    marker,
    models_path,
    path_of,
    query_of,
    sse_payloads,
    streamed_text,
    usage_metadata,
    vertex,
)
from pydantic import JsonValue

_STREAM: Final = models_path() + ":streamGenerateContent"
_GENERATE: Final = models_path() + ":generateContent"


def _chunks(text: str) -> tuple[dict[str, JsonValue], ...]:
    return sse_payloads(text)


# ---------------------------------------------------------------- streaming chat


def test_a_streaming_chat_posts_to_stream_generate_content_with_alt_sse(gateway: Gateway) -> None:
    """`stream: true` sends one `POST .../models/gemini-2.5-flash:streamGenerateContent` whose query
    is exactly `alt=sse` and whose `Authorization` carries the minted token."""
    with vertex(gateway, gemini_echo) as rig:
        response, _text = rig.stream(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}]}
        )
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (_STREAM, "alt=sse")
        assert call.headers["authorization"].startswith("Bearer token-")


def test_the_streaming_body_is_the_same_gemini_body_without_a_stream_key(gateway: Gateway) -> None:
    """Streaming is selected by the URL alone: the body has `contents` and `generationConfig` exactly
    as a non-streamed call would and no `stream` member, which Vertex rejects."""
    prompt: Final = f"stream body {marker()}"
    with vertex(gateway, gemini_echo) as rig:
        rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": 20})
        body: Final = rig.peer.one_body()
        assert body["contents"] == [{"role": "user", "parts": [{"text": prompt}]}]
        assert object_value(body["generationConfig"])["max_output_tokens"] == 20
        assert "stream" not in body


def test_gemini_frames_are_relayed_as_chat_chunks_ending_with_done(gateway: Gateway) -> None:
    """Two Gemini frames carrying "hello " and "stream" reach the caller as `chat.completion.chunk`
    frames whose deltas join to "hello stream", and the body ends with `data: [DONE]`."""
    with vertex(gateway, gemini_echo) as rig:
        response, text = rig.stream(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}]}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert streamed_text(text) == "hello stream"
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in _chunks(text))
        assert text.rstrip().endswith("data: [DONE]")


def test_the_last_chunk_carries_the_mapped_finish_reason(gateway: Gateway) -> None:
    """A stream whose last Gemini frame says `finishReason: MAX_TOKENS` ends with a chunk whose
    `finish_reason` is `length`, and no earlier chunk carries a finish reason."""
    with vertex(gateway, lambda _r: gemini_sse(finish="MAX_TOKENS")) as rig:
        _response, text = rig.stream(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}]}
        )
        reasons: Final = [
            object_value(list_value(chunk["choices"])[0]).get("finish_reason")
            for chunk in _chunks(text)
            if chunk.get("choices")
        ]
        assert [reason for reason in reasons if reason is not None] == ["length"]


def test_include_usage_adds_the_gemini_usage_on_the_final_chunk(gateway: Gateway) -> None:
    """With `stream_options: {"include_usage": true}` the stream carries a chunk whose `usage` is
    `{prompt_tokens: 11, completion_tokens: 7, total_tokens: 18}` from the last frame's
    `usageMetadata`, and it is the last chunk before `[DONE]`."""
    with vertex(gateway, gemini_echo) as rig:
        _response, text = rig.stream(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": f"s {marker()}"}], "stream_options": {"include_usage": True}},
        )
        with_usage: Final = [chunk for chunk in _chunks(text) if chunk.get("usage")]
        assert with_usage, text
        usage: Final = object_value(with_usage[-1]["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 7, 18)
        assert _chunks(text)[-1] == with_usage[-1]


def test_streamed_thoughts_tokens_are_reported_as_reasoning_tokens(gateway: Gateway) -> None:
    """A stream whose final `usageMetadata` has `thoughtsTokenCount: 30` outside the candidate count
    reports `completion_tokens = 7 + 30 = 37` and `completion_tokens_details.reasoning_tokens = 30`
    on the usage chunk, the same arithmetic as a non-streamed call."""
    usage_in: Final = usage_metadata(11, 7, thoughtsTokenCount=30, totalTokenCount=48)
    with vertex(gateway, lambda _r: gemini_sse(usage=usage_in)) as rig:
        _response, text = rig.stream(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": f"s {marker()}"}], "stream_options": {"include_usage": True}},
        )
        usage: Final = object_value([chunk for chunk in _chunks(text) if chunk.get("usage")][-1]["usage"])
        assert usage["completion_tokens"] == 37
        assert object_value(usage["completion_tokens_details"])["reasoning_tokens"] == 30


def test_a_streamed_thought_part_arrives_as_reasoning_content(gateway: Gateway) -> None:
    """A Gemini frame whose part has `thought: true` is relayed as `delta.reasoning_content`, never as
    `delta.content`, so the joined content is only the answer text."""

    def respond(_request: Request) -> Reply:
        frames: Final[list[dict[str, JsonValue]]] = [
            {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "pondering", "thought": True}]}, "index": 0}
                ]
            },
            {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "done"}]}, "finishReason": "STOP", "index": 0}
                ],
                "usageMetadata": usage_metadata(),
            },
        ]
        return Reply(
            content_type="text/event-stream", chunks=tuple(f"data: {json.dumps(f)}\r\n\r\n".encode() for f in frames)
        )

    with vertex(gateway, respond) as rig:
        _response, text = rig.stream(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}]}
        )
        deltas: Final = [
            object_value(object_value(list_value(chunk["choices"])[0]).get("delta") or {})
            for chunk in _chunks(text)
            if chunk.get("choices")
        ]
        assert (
            "".join(string_value(d["reasoning_content"]) for d in deltas if d.get("reasoning_content")) == "pondering"
        )
        assert streamed_text(text) == "done"


def test_a_streamed_function_call_arrives_as_a_tool_call_delta(gateway: Gateway) -> None:
    """A Gemini frame with a `functionCall` part is relayed as a `delta.tool_calls` entry naming
    `get_weather` with the JSON arguments, and the stream finishes with `finish_reason: "tool_calls"`."""

    def respond(_request: Request) -> Reply:
        frame: Final[dict[str, JsonValue]] = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Oslo"}}}],
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": usage_metadata(),
        }
        return Reply(content_type="text/event-stream", chunks=(f"data: {json.dumps(frame)}\r\n\r\n".encode(),))

    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    with vertex(gateway, respond) as rig:
        _response, text = rig.stream(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}], "tools": [tool]}
        )
        choices: Final = [
            object_value(list_value(chunk["choices"])[0]) for chunk in _chunks(text) if chunk.get("choices")
        ]
        calls: Final = [
            object_value(call)
            for item in choices
            for call in list_value(object_value(item.get("delta") or {}).get("tool_calls") or [])
        ]
        assert calls and object_value(calls[0]["function"])["name"] == "get_weather"
        arguments: Final = "".join(str(object_value(call["function"]).get("arguments") or "") for call in calls)
        assert json.loads(arguments) == {"city": "Oslo"}
        assert "tool_calls" in [item.get("finish_reason") for item in choices]


def test_a_streamed_gemini_chat_writes_one_spend_row(gateway: Gateway) -> None:
    """A streamed call is billed once: exactly one SpendLogs row for the model group with 11 prompt and
    7 completion tokens taken from the last frame's `usageMetadata`."""
    from integration._support.client import eventually
    from integration._support.database import read_rows

    with vertex(gateway, gemini_echo) as rig:
        rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": f"s {marker()}"}]})
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (rig.model,)
            ),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert [(row["prompt_tokens"], row["completion_tokens"]) for row in rows] == [(11, 7)]


# ---------------------------------------------------------------- /v1/responses


def _output_text(body: dict[str, JsonValue]) -> str:
    texts: Final = [
        string_value(object_value(part)["text"])
        for item in list_value(body["output"])
        if object_value(item)["type"] == "message"
        for part in list_value(object_value(item)["content"])
        if object_value(part)["type"] == "output_text"
    ]
    return "".join(texts)


def test_responses_input_reaches_generate_content_as_a_user_turn(gateway: Gateway) -> None:
    """`POST /v1/responses` with a string `input` sends one non-streamed `:generateContent` call whose
    `contents` is the single user turn holding that text."""
    prompt: Final = f"responses {marker()}"
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.post("/v1/responses", {"input": prompt})
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert path_of(call) == _GENERATE
        assert object_value(json.loads(call.body))["contents"] == [{"role": "user", "parts": [{"text": prompt}]}]


def test_responses_instructions_become_the_system_instruction(gateway: Gateway) -> None:
    """`instructions` is sent as `system_instruction: {"parts": [{"text": <instructions>}]}` and not as
    a user turn in `contents`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.post("/v1/responses", {"input": f"q {marker()}", "instructions": "reply tersely"}).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["system_instruction"] == {"parts": [{"text": "reply tersely"}]}
        assert len(list_value(body["contents"])) == 1


def test_responses_max_output_tokens_becomes_max_output_tokens(gateway: Gateway) -> None:
    """`max_output_tokens: 64` on a Responses request reaches Vertex as
    `generationConfig.max_output_tokens: 64`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.post("/v1/responses", {"input": f"q {marker()}", "max_output_tokens": 64}).status_code == 200
        assert object_value(rig.peer.one_body()["generationConfig"])["max_output_tokens"] == 64


def test_responses_reasoning_effort_becomes_the_gemini_thinking_budget(gateway: Gateway) -> None:
    """`reasoning: {"effort": "low"}` reaches Gemini 2.5 as `thinkingConfig.thinkingBudget: 1024`.
    UNVERIFIED: assumes the bridge forwards the Responses `reasoning` object as `reasoning_effort`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.post("/v1/responses", {"input": f"q {marker()}", "reasoning": {"effort": "low"}}).status_code == 200
        assert at(rig.peer.one_body(), "generationConfig", "thinkingConfig", "thinkingBudget") == 1024


def test_the_gemini_answer_comes_back_as_a_completed_responses_object(gateway: Gateway) -> None:
    """The caller gets `object: "response"`, `status: "completed"` and one `message` output item whose
    `output_text` is the candidate text."""
    with vertex(gateway, lambda _r: gemini_reply(text="from vertex")) as rig:
        response: Final = rig.post("/v1/responses", {"input": f"q {marker()}"})
        assert response.status_code == 200, response.text
        body: Final = json_of(response)
        assert (body["object"], body["status"]) == ("response", "completed")
        assert _output_text(body) == "from vertex"


def test_responses_usage_uses_input_and_output_token_names(gateway: Gateway) -> None:
    """Gemini's 11 prompt and 7 candidate tokens are reported as `usage {input_tokens: 11,
    output_tokens: 7, total_tokens: 18}`, the Responses names rather than the chat ones."""
    with vertex(gateway, gemini_echo) as rig:
        usage: Final = object_value(json_of(rig.post("/v1/responses", {"input": f"q {marker()}"}))["usage"])
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 7, 18)


def test_responses_usage_reports_cached_and_reasoning_details(gateway: Gateway) -> None:
    """`cachedContentTokenCount: 40` and `thoughtsTokenCount: 30` (outside the candidate count) come
    back as `input_tokens_details.cached_tokens: 40` and `output_tokens_details.reasoning_tokens: 30`.
    UNVERIFIED: the field transfer is in the chat to responses usage bridge."""
    usage_in: Final = usage_metadata(100, 20, cachedContentTokenCount=40, thoughtsTokenCount=30, totalTokenCount=150)
    with vertex(gateway, lambda _r: gemini_reply(usage=usage_in)) as rig:
        usage: Final = object_value(json_of(rig.post("/v1/responses", {"input": f"q {marker()}"}))["usage"])
        assert at(usage, "input_tokens_details", "cached_tokens") == 40
        assert at(usage, "output_tokens_details", "reasoning_tokens") == 30


def test_a_streamed_response_goes_to_stream_generate_content_and_ends_completed(gateway: Gateway) -> None:
    """`stream: true` on `/v1/responses` posts to `:streamGenerateContent?alt=sse` and the caller gets
    Responses events that include `response.output_text.delta` frames joining to "hello stream" and a
    final `response.completed` event."""
    with vertex(gateway, gemini_echo) as rig:
        response, text = rig.stream("/v1/responses", {"input": f"q {marker()}"})
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert (path_of(call), query_of(call)) == (_STREAM, "alt=sse")
        events: Final = sse_payloads(text)
        deltas: Final = "".join(
            str(event.get("delta", "")) for event in events if event.get("type") == "response.output_text.delta"
        )
        assert deltas == "hello stream"
        assert events[-1]["type"] == "response.completed"
