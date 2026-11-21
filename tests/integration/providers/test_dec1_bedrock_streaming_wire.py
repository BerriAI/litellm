"""Streaming `/v1/chat/completions` from Bedrock: AWS event-stream framing in, OpenAI SSE chunks out.

Bedrock streams never use SSE. Converse streams (`/converse-stream`) send binary
`application/vnd.amazon.eventstream` frames whose `:event-type` header names a Converse event
(`messageStart`, `contentBlockStart`, `contentBlockDelta`, `contentBlockStop`, `messageStop`, `metadata`),
and InvokeModel streams (`/invoke-with-response-stream`) send `chunk` frames whose JSON payload holds the
model family's own event base64 encoded under `bytes`. The proxy must decode both, emit OpenAI
`chat.completion.chunk` SSE events in order, carry reasoning deltas as `reasoning_content`, put the usage
(including cache counts that only arrive in the trailing `metadata` frame) on the final chunk when the
caller asks for `stream_options.include_usage`, end with `data: [DONE]`, and bill the call from that usage.

The plain Converse text, stop reason, metadata usage, tool fragments and mid-stream exception paths are
pinned in `streaming/test_dec1_stream_provider_translation.py` and `test_bedrock_converse_stream_event_frames_wire.py`;
these tests add reasoning, cache usage, InvokeModel families and stream-open errors.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_msgprov_support import invoke_stream_frames
from integration.providers.dec1_bedrock_support import (
    CLAUDE_SONNET_45,
    EVENT_STREAM,
    INPUT_COST,
    LLAMA2_13B,
    MISTRAL_7B,
    OUTPUT_COST,
    PRICES,
    aws_error,
    bedrock,
    converse_frames,
    error_message,
    invoke_frames,
    path_of,
    spend_row,
    sse_events,
    streamed_text,
)
from pydantic import JsonValue

MODEL: Final = f"bedrock/{CLAUDE_SONNET_45}"
USAGE_OPTIONS: Final[dict[str, JsonValue]] = {"include_usage": True}


def _converse_events(
    *blocks: tuple[str, dict[str, JsonValue]], usage: dict[str, JsonValue] | None = None, stop: str = "end_turn"
) -> Reply:
    return converse_frames(
        (
            ("messageStart", {"role": "assistant"}),
            *blocks,
            ("messageStop", {"stopReason": stop}),
            (
                "metadata",
                {
                    "usage": usage if usage is not None else {"inputTokens": 13, "outputTokens": 6, "totalTokens": 19},
                    "metrics": {"latencyMs": 1},
                },
            ),
        )
    )


def test_a_streamed_chat_call_posts_to_converse_stream_and_answers_sse_ending_in_done(gateway: Gateway) -> None:
    """`stream: true` on a Converse deployment must hit `/model/<id>/converse-stream` once.

    The caller gets HTTP 200 `text/event-stream`, every `data:` payload is a `chat.completion.chunk`, the
    text deltas join to `hello stream`, the last choice chunk has `finish_reason: stop`, and the body ends
    with `data: [DONE]`.
    """

    def respond(_: Request) -> Reply:
        return _converse_events(
            ("contentBlockDelta", {"delta": {"text": "hello "}, "contentBlockIndex": 0}),
            ("contentBlockDelta", {"delta": {"text": "stream"}, "contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert path_of(rig.one_post()) == f"/model/{CLAUDE_SONNET_45}/converse-stream"
        events: Final = sse_events(response)
        assert {event["object"] for event in events} == {"chat.completion.chunk"}
        assert streamed_text(events) == "hello stream"
        finishes: Final = [c["finish_reason"] for e in events for c in e.get("choices", []) if c.get("finish_reason")]  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue, reportUnknownMemberType]
        assert finishes == ["stop"], finishes
        assert response.text.rstrip().endswith("data: [DONE]")


def test_converse_reasoning_deltas_stream_as_reasoning_content_before_the_text(gateway: Gateway) -> None:
    """Bedrock streams two `reasoningContent.text` deltas, a `reasoningContent.signature` delta, then text.

    The caller's chunks must carry `delta.reasoning_content` pieces that join to `think hard`, followed by
    content deltas that join to `done`, and no reasoning text may leak into `content`.
    """

    def respond(_: Request) -> Reply:
        return _converse_events(
            ("contentBlockDelta", {"delta": {"reasoningContent": {"text": "think "}}, "contentBlockIndex": 0}),
            ("contentBlockDelta", {"delta": {"reasoningContent": {"text": "hard"}}, "contentBlockIndex": 0}),
            ("contentBlockDelta", {"delta": {"reasoningContent": {"signature": "sig-stream"}}, "contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
            ("contentBlockDelta", {"delta": {"text": "done"}, "contentBlockIndex": 1}),
            ("contentBlockStop", {"contentBlockIndex": 1}),
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        events: Final = sse_events(response)
        reasoning: Final = "".join(
            str(choice["delta"].get("reasoning_content") or "")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            for event in events
            for choice in event.get("choices", [])  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue, reportUnknownMemberType]
        )
        assert reasoning == "think hard"
        assert streamed_text(events) == "done"


def test_converse_stream_cache_counts_from_the_metadata_frame_reach_the_final_usage_chunk(gateway: Gateway) -> None:
    """The trailing `metadata` frame reports `inputTokens: 10`, `cacheReadInputTokens: 5`, `cacheWriteInputTokens: 9`, `outputTokens: 4`.

    With `stream_options.include_usage` the last chunk's `usage` must report `prompt_tokens` 24,
    `completion_tokens` 4 and `prompt_tokens_details.cached_tokens` 5, the same totals as the
    non-streaming Converse answer.
    """

    def respond(_: Request) -> Reply:
        return _converse_events(
            ("contentBlockDelta", {"delta": {"text": "cached"}, "contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
            usage={
                "inputTokens": 10,
                "outputTokens": 4,
                "totalTokens": 28,
                "cacheReadInputTokens": 5,
                "cacheWriteInputTokens": 9,
            },
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(stream=True, stream_options=USAGE_OPTIONS)
        assert response.status_code == 200, response.text
        usage_chunks: Final = [event["usage"] for event in sse_events(response) if event.get("usage")]
        assert usage_chunks, response.text
        usage: Final = usage_chunks[-1]
        assert usage["prompt_tokens"] == 24  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert usage["completion_tokens"] == 4  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert usage["prompt_tokens_details"]["cached_tokens"] == 5  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_a_streamed_converse_call_is_billed_from_the_trailing_metadata_usage(gateway: Gateway) -> None:
    """A streamed Converse call whose `metadata` frame reports 13 input and 6 output tokens, with deployment prices.

    The SpendLogs row for the stream's id must record `prompt_tokens` 13, `completion_tokens` 6 and
    `spend = 13 * 0.001 + 6 * 0.002 = 0.025`, so usage that arrives after `messageStop` is not lost to billing.
    """

    def respond(_: Request) -> Reply:
        return _converse_events(
            ("contentBlockDelta", {"delta": {"text": "billed"}, "contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
        )

    with bedrock(gateway, respond, model=MODEL, **PRICES) as rig:
        response: Final = rig.chat(stream=True, stream_options=USAGE_OPTIONS)
        assert response.status_code == 200, response.text
        stream_id: Final = str(sse_events(response)[0]["id"])
        row: Final = spend_row(stream_id)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (13, 6)
        assert float(str(row["spend"])) == pytest.approx(13 * INPUT_COST + 6 * OUTPUT_COST)


def test_claude_on_invoke_streams_through_invoke_with_response_stream(gateway: Gateway) -> None:
    """A Claude `bedrock/invoke/` deployment with `stream: true` must call `/model/<id>/invoke-with-response-stream`.

    Bedrock answers six `chunk` frames, each a base64 Anthropic stream event (message_start, text deltas
    `hello ` and `stream`, message_delta with 11 in and 7 out, message_stop). The caller must get SSE text
    `hello stream` and, with `include_usage`, a final usage of 11 prompt and 7 completion tokens.
    """

    def respond(_: Request) -> Reply:
        return Reply(content_type=EVENT_STREAM, chunks=invoke_stream_frames("msg_dec1_stream", model=CLAUDE_SONNET_45))

    with bedrock(gateway, respond, model=f"bedrock/invoke/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(stream=True, stream_options=USAGE_OPTIONS, max_tokens=64)
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == f"/model/{CLAUDE_SONNET_45}/invoke-with-response-stream"
        events: Final = sse_events(response)
        assert streamed_text(events) == "hello stream"
        usage: Final = [event["usage"] for event in events if event.get("usage")][-1]
        assert (usage["prompt_tokens"], usage["completion_tokens"]) == (11, 7)  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_mistral_invoke_stream_chunks_are_decoded_from_outputs_text(gateway: Gateway) -> None:
    """A Mistral InvokeModel stream sends `chunk` frames `{"outputs": [{"text": "bon"}]}` and `{"outputs": [{"text": "jour"}]}`.

    The proxy must call `/model/<id>/invoke-with-response-stream` and the caller's SSE text must be
    `bonjour` (`_chunk_parser`, `invoke_handler.py:820`).
    """

    def respond(_: Request) -> Reply:
        return invoke_frames(
            [
                {"outputs": [{"text": "bon", "stop_reason": None}]},
                {"outputs": [{"text": "jour", "stop_reason": "stop"}]},
            ]
        )

    with bedrock(gateway, respond, model=f"bedrock/{MISTRAL_7B}") as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == f"/model/{MISTRAL_7B}/invoke-with-response-stream"
        assert streamed_text(sse_events(response)) == "bonjour"


def test_llama_invoke_stream_chunks_are_decoded_from_generation(gateway: Gateway) -> None:
    """A Llama InvokeModel stream sends `chunk` frames `{"generation": "hel"}` and `{"generation": "lo", "stop_reason": "stop"}`.

    The last one also carries `amazon-bedrock-invocationMetrics`, which must never surface as text. The
    caller's SSE text must be exactly `hello`.
    """

    def respond(_: Request) -> Reply:
        return invoke_frames(
            [
                {"generation": "hel", "stop_reason": None},
                {
                    "generation": "lo",
                    "stop_reason": "stop",
                    "amazon-bedrock-invocationMetrics": {"inputTokenCount": 9, "outputTokenCount": 2},
                },
            ]
        )

    with bedrock(gateway, respond, model=f"bedrock/{LLAMA2_13B}") as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        assert streamed_text(sse_events(response)) == "hello"


def test_a_validation_error_on_stream_open_is_an_http_400_not_a_200_stream(gateway: Gateway) -> None:
    """`/converse-stream` itself answers HTTP 400 `{"message": "Malformed input request"}` before any frame.

    Nothing was streamed yet, so the caller must get HTTP 400 JSON (not a 200 event stream with an error
    event) whose message carries `Malformed input request`, mapped by `_map_bedrock_exception`.
    """

    def respond(_: Request) -> Reply:
        return aws_error(400, "ValidationException", "Malformed input request: extraneous key [foo]")

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(stream=True, num_retries=0)
        assert response.status_code == 400, response.text
        assert "Malformed input request" in error_message(response), response.text


def test_a_throttled_stream_open_is_an_http_429(gateway: Gateway) -> None:
    """`/converse-stream` answers HTTP 429 ThrottlingException `Too many requests, please wait before trying again.`.

    With `num_retries: 0` the caller must get HTTP 429 JSON whose message contains that text, and the
    peer must have been called exactly once.
    """

    def respond(_: Request) -> Reply:
        return aws_error(429, "ThrottlingException", "Too many requests, please wait before trying again.")

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(stream=True, num_retries=0)
        assert response.status_code == 429, response.text
        assert "Too many requests" in error_message(response), response.text
        assert len(rig.posts()) == 1
