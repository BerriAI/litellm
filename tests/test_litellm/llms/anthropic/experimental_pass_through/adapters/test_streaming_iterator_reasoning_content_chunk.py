"""
Regression test for a vLLM reasoning-parser boundary chunk that carries BOTH
`reasoning_content` and `content` in the same delta.

Captured live from a Qwen3.6 vLLM backend (see litellm.yaml `complex` tier):
the reasoning parser flushed the tail of its reasoning buffer bundled with
the first character of the answer in one SSE chunk. `AnthropicStreamWrapper`
picked `text` for the new content-block type (`_should_start_new_content_block`
checks `content` before `reasoning_content`) but emitted a `thinking_delta`
for it (`translate_streaming_openai_response_to_anthropic` checks
`reasoning_content` before falling back to text) — a `thinking_delta` landing
on a block the client was told is `text`. Claude Code (a strict Anthropic SSE
client) rejects that with "Content block is not a thinking block", and
because the corrupted assistant turn gets replayed as conversation history,
every subsequent turn fails the same way until the context is cleared.
"""

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, StreamingChoices, ModelResponseStream


def _reasoning_then_boundary_then_stop_chunks() -> list:
    return [
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(reasoning_content="Let me check"),
                    finish_reason=None,
                )
            ],
        ),
        # The boundary chunk: reasoning tail + first token of the answer,
        # bundled together by the vLLM reasoning parser.
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(reasoning_content=".", content="\n"),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="The answer is 42."),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(index=0, delta=Delta(), finish_reason="stop")
            ],
        ),
    ]


def _assert_no_block_type_mismatch(events: list) -> None:
    open_blocks: dict = {}
    for event in events:
        event_type = event.get("type")
        if event_type == "content_block_start":
            open_blocks[event["index"]] = event["content_block"]["type"]
        elif event_type == "content_block_delta":
            idx = event["index"]
            delta_type = event["delta"]["type"]
            declared_type = open_blocks.get(idx)
            expected_block_type = {
                "text_delta": "text",
                "thinking_delta": "thinking",
                "signature_delta": "thinking",
                "input_json_delta": "tool_use",
            }[delta_type]
            assert declared_type == expected_block_type, (
                f"index {idx} was opened as {declared_type!r} but received a "
                f"{delta_type!r} (expects a {expected_block_type!r} block)"
            )


def test_reasoning_and_content_boundary_chunk_does_not_corrupt_block_types():
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_reasoning_then_boundary_then_stop_chunks()),
        model="art",
    )
    events = list(wrapper)
    _assert_no_block_type_mismatch(events)


def test_reasoning_and_content_boundary_chunk_preserves_both_payloads():
    """The boundary chunk's reasoning tail and content must both survive as
    deltas — neither payload gets silently dropped by the split."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_reasoning_then_boundary_then_stop_chunks()),
        model="art",
    )
    events = list(wrapper)

    thinking_text = "".join(
        e["delta"]["thinking"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"]["type"] == "thinking_delta"
    )
    answer_text = "".join(
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
    )

    assert thinking_text == "Let me check."
    assert answer_text == "\nThe answer is 42."
