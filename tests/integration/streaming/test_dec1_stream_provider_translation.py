"""Provider stream dialects translated to OpenAI chat completion chunks.

A caller of `/v1/chat/completions` always gets OpenAI chunks, whatever the deployment speaks. These
contracts feed the proxy the three most used non OpenAI stream dialects from an owned peer and pin the
translation: Anthropic Messages SSE events, Gemini `streamGenerateContent` SSE objects, and Bedrock
Converse binary event stream frames. For each: text deltas, the finish reason mapping, tool calls,
reasoning, and the usage that reaches the usage chunk and the SpendLogs row.

Assumed harness shapes: `anthropic/<model>` with `api_base` at the peer origin is called on
`/v1/messages`; `gemini/<model>` with `api_base` at the peer origin is called on a path containing
`:streamGenerateContent`; `bedrock/converse/<model>` with `aws_bedrock_runtime_endpoint` at the peer is
called on `/model/<id>/converse-stream`. All three are used that way by existing tests in this directory.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.anthropic_sse import message_start, text_delta
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.streaming.dec1_stream_support import (
    CLAUDE,
    EVENT_STREAM,
    GEMINI,
    bedrock_frame,
    bedrock_scripted,
    chunks_of,
    content_of,
    data,
    deltas_of,
    event,
    finish_reasons,
    joined_arguments,
    marker,
    payloads,
    scripted,
    sse,
    tool_call_fragments,
    usage_chunks,
)

_CHAT_TOOLS: Final = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def _anthropic_tail(stop_reason: str, output_tokens: int = 3, index: int = 0) -> tuple[bytes, ...]:
    return (
        event("content_block_stop", {"type": "content_block_stop", "index": index}),
        event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        event("message_stop", {"type": "message_stop"}),
    )


def _anthropic_text(identity: str, pieces: tuple[str, ...], stop_reason: str = "end_turn") -> tuple[bytes, ...]:
    return (
        message_start(identity, CLAUDE),
        event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        event("ping", {"type": "ping"}),
        *map(text_delta, pieces),
        *_anthropic_tail(stop_reason),
    )


def _anthropic(gateway: Gateway, frames: tuple[bytes, ...]):  # type: ignore[no-untyped-def]
    return scripted(gateway, lambda request: sse(frames), model=f"anthropic/{CLAUDE}", base_suffix="")


def test_anthropic_text_events_become_openai_content_chunks_ending_in_stop_and_done(gateway: Gateway) -> None:
    """An Anthropic deployment streams `message_start`, a text block with two deltas, a ping, and
    `end_turn`. The chat caller receives `chat.completion.chunk` frames whose content concatenates to the
    two pieces, `role: assistant` on the first chunk, one `finish_reason: stop`, and `[DONE]`. No
    Anthropic event name (`message_start`, `ping`) appears in the caller's body."""
    with _anthropic(gateway, _anthropic_text(f"msg_{marker()}", ("Bon", "jour"))) as rig:
        body: Final = rig.chat().text
        chunks: Final = chunks_of(body)
        assert {item["object"] for item in chunks} == {"chat.completion.chunk"}
        assert content_of(chunks) == "Bonjour"
        assert deltas_of(chunks)[0].get("role") == "assistant"
        assert finish_reasons(chunks) == ["stop"]
        assert payloads(body)[-1] == "[DONE]"
        assert "message_start" not in body and "ping" not in body and "event:" not in body


def test_anthropic_stream_chunks_share_one_id_and_the_public_model_name(gateway: Gateway) -> None:
    """Chunks translated from an Anthropic stream all carry the same `id`, the same `created`, and
    `model` equal to the public name the caller sent, never `claude-...` or `anthropic/claude-...`."""
    with _anthropic(gateway, _anthropic_text(f"msg_{marker()}", ("same", " id"))) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        assert len({item["id"] for item in chunks}) == 1
        assert len({item["created"] for item in chunks}) == 1
        assert {item["model"] for item in chunks} == {rig.model}


@pytest.mark.parametrize(
    ("stop_reason", "finish"),
    [("end_turn", "stop"), ("stop_sequence", "stop"), ("max_tokens", "length"), ("refusal", "content_filter")],
)
def test_anthropic_stop_reason_maps_to_the_openai_finish_reason(
    gateway: Gateway, stop_reason: str, finish: str
) -> None:
    """An Anthropic stream ends with each stop reason in turn. The chat caller's finish reason is the
    OpenAI equivalent from `_FINISH_REASON_MAP`: `end_turn` and `stop_sequence` give `stop`, `max_tokens`
    gives `length`, `refusal` gives `content_filter`."""
    with _anthropic(gateway, _anthropic_text(f"msg_{marker()}", ("map",), stop_reason)) as rig:
        assert finish_reasons(chunks_of(rig.chat().text)) == [finish]


def test_anthropic_usage_events_become_one_openai_usage_chunk_and_the_spend_row(gateway: Gateway) -> None:
    """An Anthropic stream reports `input_tokens: 5` in `message_start` and `output_tokens: 3` in
    `message_delta`. With `include_usage` the chat caller's final chunk has `choices: []` and usage
    5, 3 and 8, and the SpendLogs row records 5 prompt and 3 completion tokens."""
    with _anthropic(gateway, _anthropic_text(f"msg_{marker()}", ("count",))) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        usage: Final = object_value(usage_chunks(chunks)[-1]["usage"])
        assert chunks[-1]["choices"] == []
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (5, 3, 8)
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (5, 3)


def test_anthropic_cache_read_tokens_are_added_to_prompt_tokens_and_reported_as_cached(gateway: Gateway) -> None:
    """Anthropic reports uncached input separately from cache reads: `input_tokens: 5`,
    `cache_read_input_tokens: 7`. In OpenAI usage the chat caller sees `prompt_tokens: 12` (the sum) and
    `prompt_tokens_details.cached_tokens: 7`. Getting the sum wrong under or over bills every cached
    Claude call."""
    identity: Final = f"msg_{marker()}"
    start: Final = event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": CLAUDE,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 5,
                    "cache_read_input_tokens": 7,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 1,
                },
            },
        },
    )
    frames: Final = (start, *_anthropic_text(identity, ("cached",))[1:])
    with _anthropic(gateway, frames) as rig:
        usage: Final = object_value(
            usage_chunks(chunks_of(rig.chat(stream_options={"include_usage": True}).text))[-1]["usage"]
        )
        assert usage["prompt_tokens"] == 12
        assert object_value(usage["prompt_tokens_details"])["cached_tokens"] == 7
        assert usage["completion_tokens"] == 3


def test_anthropic_tool_use_block_becomes_openai_tool_call_fragments(gateway: Gateway) -> None:
    """An Anthropic stream opens a `tool_use` block (`toolu_dec1`, `get_weather`) and sends two
    `input_json_delta` pieces, ending with `stop_reason: tool_use`. The chat caller gets tool call
    fragments at index 0 with that id and name on the first fragment, arguments concatenating to
    `{"city":"Paris"}`, and `finish_reason: tool_calls`."""
    identity: Final = f"msg_{marker()}"
    frames: Final = (
        message_start(identity, CLAUDE),
        event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_dec1", "name": "get_weather", "input": {}},
            },
        ),
        event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
            },
        ),
        event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'},
            },
        ),
        *_anthropic_tail("tool_use"),
    )
    with _anthropic(gateway, frames) as rig:
        chunks: Final = chunks_of(rig.chat(tools=_CHAT_TOOLS).text)
        fragments: Final = tool_call_fragments(chunks, 0)
        assert [item["id"] for item in fragments if item.get("id")] == ["toolu_dec1"]
        assert object_value(fragments[0]["function"])["name"] == "get_weather"
        assert json.loads(joined_arguments(fragments)) == {"city": "Paris"}
        assert finish_reasons(chunks) == ["tool_calls"]


def test_anthropic_thinking_block_becomes_reasoning_content_before_the_answer(gateway: Gateway) -> None:
    """An Anthropic stream sends a `thinking` block (two `thinking_delta` pieces and a
    `signature_delta`) and then a text block. The chat caller receives the thinking text under
    `delta.reasoning_content`, then the answer under `delta.content`; the thinking text is never mixed
    into `content`."""
    identity: Final = f"msg_{marker()}"
    frames: Final = (
        message_start(identity, CLAUDE),
        event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "consider "}},
        ),
        event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "carefully"}},
        ),
        event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "c2lnbmF0dXJl"},
            },
        ),
        event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        event(
            "content_block_start",
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        ),
        event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "answer"}},
        ),
        *_anthropic_tail("end_turn", index=1),
    )
    with _anthropic(gateway, frames) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert content_of(chunks, field="reasoning_content") == "consider carefully"
        assert content_of(chunks) == "answer"


GEMINI_USAGE: Final = {"promptTokenCount": 9, "candidatesTokenCount": 7, "totalTokenCount": 16}


def _gemini_text(parts: tuple[str, ...], finish: str = "STOP") -> tuple[bytes, ...]:
    """One Gemini object per text part; the last carries `finishReason` and `usageMetadata` together.

    That is the shape Gemini sends: usage rides on the finishing object rather than on a separate
    trailing object. The proxy reads a Gemini finish reason as provisional (`chunk_parser` marks every
    chunk `is_finished: False`) and keeps the latest one it saw, so a trailing candidate free object
    would reset any non STOP reason to the default `stop`."""

    def frame(part: str) -> bytes:
        last: Final = part == parts[-1]
        candidate: Final = {
            "content": {"parts": [{"text": part}], "role": "model"},
            "index": 0,
            **({"finishReason": finish} if last else {}),
        }
        return data({"candidates": [candidate], **({"usageMetadata": GEMINI_USAGE} if last else {})})

    return tuple(frame(part) for part in parts)


def _gemini(gateway: Gateway, frames: tuple[bytes, ...]):  # type: ignore[no-untyped-def]
    return scripted(gateway, lambda request: sse(frames), model=f"gemini/{GEMINI}", base_suffix="")


def test_gemini_candidate_parts_become_openai_content_chunks(gateway: Gateway) -> None:
    """A Gemini deployment streams two `candidates[0].content.parts[0].text` objects, the second with
    `finishReason: STOP` and the usage. The chat caller receives OpenAI chunks whose content
    is the two parts, one `finish_reason: stop`, `[DONE]`, and no `candidates` key anywhere. The peer was
    called on a `:streamGenerateContent` path."""
    with _gemini(gateway, _gemini_text(("Guten ", "Tag"))) as rig:
        body: Final = rig.chat().text
        chunks: Final = chunks_of(body)
        assert content_of(chunks) == "Guten Tag"
        assert finish_reasons(chunks) == ["stop"]
        assert payloads(body)[-1] == "[DONE]"
        assert "candidates" not in body
        assert all(
            ":streamGenerateContent" in request.target for request in rig.wire.drain() if request.method == "POST"
        )


@pytest.mark.parametrize(
    ("gemini_reason", "finish"),
    [("MAX_TOKENS", "length"), ("SAFETY", "content_filter"), ("RECITATION", "content_filter")],
)
def test_gemini_finish_reason_maps_to_the_openai_finish_reason(
    gateway: Gateway, gemini_reason: str, finish: str
) -> None:
    """A Gemini stream ends with `MAX_TOKENS`, `SAFETY` or `RECITATION`. The chat caller's finish reason
    is `length`, `content_filter` and `content_filter` respectively (`_FINISH_REASON_MAP`)."""
    with _gemini(gateway, _gemini_text(("mapped ", "reason"), gemini_reason)) as rig:
        assert finish_reasons(chunks_of(rig.chat().text)) == [finish]


def test_gemini_usage_metadata_becomes_the_usage_chunk_and_the_spend_row(gateway: Gateway) -> None:
    """A Gemini stream ends with `usageMetadata` of 9 prompt, 7 candidate and 16 total tokens. With
    `include_usage` the chat caller's usage chunk reports 9, 7 and 16, and the SpendLogs row records 9
    prompt and 7 completion tokens with `custom_llm_provider == "gemini"`."""
    with _gemini(gateway, _gemini_text(("count ", "me"))) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        usage: Final = object_value(usage_chunks(chunks)[-1]["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (9, 7, 16)
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["custom_llm_provider"]) == (9, 7, "gemini")


def test_gemini_function_call_part_becomes_a_complete_openai_tool_call(gateway: Gateway) -> None:
    """A Gemini stream answers with one `functionCall` part (`get_weather`, args `{"city": "Paris"}`)
    and `finishReason: STOP`. Gemini sends arguments as a whole object, so the chat caller gets one tool
    call with a generated non empty `id`, the name, arguments that parse to the same object, and
    `finish_reason: tool_calls` even though Gemini said STOP."""
    part: Final = {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
    frames: Final = (
        data({"candidates": [{"content": {"parts": [part], "role": "model"}, "index": 0, "finishReason": "STOP"}]}),
        data({"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 7, "totalTokenCount": 16}}),
    )
    with _gemini(gateway, frames) as rig:
        chunks: Final = chunks_of(rig.chat(tools=_CHAT_TOOLS).text)
        fragments: Final = tool_call_fragments(chunks, 0)
        assert fragments and fragments[0].get("id")
        assert object_value(fragments[0]["function"])["name"] == "get_weather"
        assert json.loads(joined_arguments(fragments)) == {"city": "Paris"}
        assert finish_reasons(chunks) == ["tool_calls"]


def test_gemini_thought_parts_become_reasoning_content(gateway: Gateway) -> None:
    """A Gemini stream sends a part flagged `thought: true` and then a normal text part. The chat caller
    receives the thought text under `delta.reasoning_content` and the answer under `delta.content`.
    UNVERIFIED: read from the `thought` handling in
    `litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py`."""
    frames: Final = (
        data(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "pondering", "thought": True}], "role": "model"}, "index": 0}
                ]
            }
        ),
        data(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "answer"}], "role": "model"}, "index": 0, "finishReason": "STOP"}
                ]
            }
        ),
        data(
            {
                "usageMetadata": {
                    "promptTokenCount": 9,
                    "candidatesTokenCount": 7,
                    "totalTokenCount": 16,
                    "thoughtsTokenCount": 2,
                }
            }
        ),
    )
    with _gemini(gateway, frames) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert content_of(chunks, field="reasoning_content") == "pondering"
        assert content_of(chunks) == "answer"


def _converse(*events: tuple[str, dict]) -> Reply:
    return Reply(content_type=EVENT_STREAM, chunks=tuple(bedrock_frame(kind, payload) for kind, payload in events))


_CONVERSE_USAGE: Final = (
    "metadata",
    {"usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14}, "metrics": {"latencyMs": 1}},
)


def _converse_text(stop_reason: str = "end_turn") -> Reply:
    return _converse(
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"delta": {"text": "Hej "}, "contentBlockIndex": 0}),
        ("contentBlockDelta", {"delta": {"text": "världen"}, "contentBlockIndex": 0}),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": stop_reason}),
        _CONVERSE_USAGE,
    )


def _is_converse_stream(request: Request) -> bool:
    return request.target.endswith("/converse-stream")


def test_bedrock_binary_event_stream_becomes_openai_sse_chunks(gateway: Gateway) -> None:
    """A Bedrock Converse deployment answers with binary `application/vnd.amazon.eventstream` frames:
    messageStart, two text deltas, contentBlockStop, messageStop `end_turn`, metadata. The chat caller
    receives ordinary `text/event-stream` chunks whose content is `Hej världen`, one
    `finish_reason: stop`, and `[DONE]`. The peer was called on `/model/<id>/converse-stream`."""
    with bedrock_scripted(gateway, lambda request: _converse_text()) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].split(";")[0] == "text/event-stream"
        chunks: Final = chunks_of(response.text)
        assert content_of(chunks) == "Hej världen"
        assert finish_reasons(chunks) == ["stop"]
        assert payloads(response.text)[-1] == "[DONE]"
        assert all(_is_converse_stream(request) for request in rig.wire.drain() if request.method == "POST")


@pytest.mark.parametrize(
    ("stop_reason", "finish"),
    [
        ("max_tokens", "length"),
        ("stop_sequence", "stop"),
        ("content_filtered", "content_filter"),
        ("guardrail_intervened", "content_filter"),
    ],
)
def test_bedrock_stop_reason_maps_to_the_openai_finish_reason(gateway: Gateway, stop_reason: str, finish: str) -> None:
    """A Bedrock Converse stream ends with each `stopReason`. The chat caller's finish reason is
    `length` for `max_tokens`, `stop` for `stop_sequence`, and `content_filter` for `content_filtered`
    and `guardrail_intervened` (`_FINISH_REASON_MAP` in `litellm/litellm_core_utils/core_helpers.py`)."""
    with bedrock_scripted(gateway, lambda request: _converse_text(stop_reason)) as rig:
        assert finish_reasons(chunks_of(rig.chat().text)) == [finish]


def test_bedrock_metadata_usage_becomes_the_usage_chunk_and_the_spend_row(gateway: Gateway) -> None:
    """The Bedrock `metadata` event reports 9 input, 5 output and 14 total tokens. With `include_usage`
    the chat caller's last chunk has `choices: []` and usage 9, 5 and 14; the SpendLogs row records the
    same counts with `custom_llm_provider == "bedrock"`."""
    with bedrock_scripted(gateway, lambda request: _converse_text()) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        usage: Final = object_value(usage_chunks(chunks)[-1]["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (9, 5, 14)
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["custom_llm_provider"]) == (9, 5, "bedrock")


def test_bedrock_tool_use_events_become_openai_tool_call_fragments(gateway: Gateway) -> None:
    """A Bedrock Converse stream opens a tool use block (`tooluse_dec1`, `get_weather`) with
    contentBlockStart, sends the input JSON in two `toolUse.input` string pieces, and stops with
    `tool_use`. The chat caller gets tool call fragments with that id and name, arguments concatenating
    to `{"city":"Paris"}`, and `finish_reason: tool_calls`."""
    reply: Final = _converse(
        ("messageStart", {"role": "assistant"}),
        (
            "contentBlockStart",
            {"start": {"toolUse": {"toolUseId": "tooluse_dec1", "name": "get_weather"}}, "contentBlockIndex": 0},
        ),
        ("contentBlockDelta", {"delta": {"toolUse": {"input": '{"city":'}}, "contentBlockIndex": 0}),
        ("contentBlockDelta", {"delta": {"toolUse": {"input": '"Paris"}'}}, "contentBlockIndex": 0}),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "tool_use"}),
        _CONVERSE_USAGE,
    )
    with bedrock_scripted(gateway, lambda request: reply) as rig:
        chunks: Final = chunks_of(rig.chat(tools=_CHAT_TOOLS).text)
        fragments: Final = tool_call_fragments(chunks, 0)
        assert [item["id"] for item in fragments if item.get("id")] == ["tooluse_dec1"]
        assert "".join(str(object_value(item["function"]).get("name") or "") for item in fragments) == "get_weather"
        assert json.loads(joined_arguments(fragments)) == {"city": "Paris"}
        assert finish_reasons(chunks) == ["tool_calls"]


def test_bedrock_exception_frame_mid_stream_ends_the_callers_stream_with_an_error_frame(gateway: Gateway) -> None:
    """After one text delta Bedrock sends an event stream frame with `:message-type: exception` and
    `:exception-type: throttlingException`. The chat caller receives the text, then an in-band
    `data: {"error": ...}` frame, then `[DONE]`; the stream does not end as if it had completed.
    UNVERIFIED on the error wording; this asserts only that an error frame precedes `[DONE]` and that
    no `finish_reason` was sent."""
    from integration._support.upstream import aws_event_stream_frame

    failure: Final = aws_event_stream_frame(
        {":exception-type": "throttlingException", ":content-type": "application/json", ":message-type": "exception"},
        b'{"message":"Too many requests"}',
    )
    frames: Final = (
        bedrock_frame("messageStart", {"role": "assistant"}),
        bedrock_frame("contentBlockDelta", {"delta": {"text": "partial"}, "contentBlockIndex": 0}),
        failure,
    )
    with bedrock_scripted(gateway, lambda request: Reply(content_type=EVENT_STREAM, chunks=frames)) as rig:
        seen: Final = payloads(rig.chat().text)
        assert seen[-1] == "[DONE]"
        assert "error" in json.loads(seen[-2])
        assert finish_reasons([json.loads(item) for item in seen[:-2]]) == []
