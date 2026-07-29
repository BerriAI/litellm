"""
Regression tests for fake-streamed providers routed through `/v1/messages`.

A fake-streaming provider (e.g. Vertex AI Gemma `:predict`) collapses its whole
response into a single `MockResponseIterator` chunk that carries content text AND a
`finish_reason` together. `AnthropicStreamWrapper` previously dropped all content in
this case — `translate_streaming_openai_response_to_anthropic` sees the finish_reason
and emits only a `message_delta`. `_CombinedChunkSplitter` splits such chunks so the
content survives.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import AsyncIterator

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
    _CombinedChunkSplitter,
)
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    PromptTokensDetailsWrapper,
    StreamingChoices,
    Usage,
)


def _build_fake_stream(
    content: str, finish_reason: str = "stop"
) -> MockResponseIterator:
    """Mimic a Vertex Gemma `:predict` fake stream: one collapsed chunk."""
    model_response = ModelResponse()
    model_response.choices = [
        Choices(
            index=0,
            message=Message(role="assistant", content=content),
            finish_reason=finish_reason,
        )
    ]
    model_response.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    model_response.model = "gemma4"
    return MockResponseIterator(model_response=model_response)


def _collect_async(wrapper: AnthropicStreamWrapper) -> str:
    async def _run() -> str:
        out = []
        async for raw in wrapper.async_anthropic_sse_wrapper():
            out.append(raw.decode() if isinstance(raw, bytes) else raw)
        return "".join(out)

    return asyncio.run(_run())


def test_fake_stream_content_reaches_anthropic_sse():
    """Content from a collapsed fake-stream chunk must be emitted as a delta."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=_build_fake_stream("Hello, the answer is 2."),
        model="gemma4",
    )
    sse = _collect_async(wrapper)

    assert "content_block_delta" in sse
    assert "Hello, the answer is 2." in sse
    assert "message_delta" in sse
    assert "message_stop" in sse


def test_fake_stream_usage_preserved():
    """The finish chunk keeps usage so output_tokens is non-zero."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=_build_fake_stream("Two."),
        model="gemma4",
    )
    sse = _collect_async(wrapper)

    message_delta = next(
        json.loads(line[len("data: ") :])
        for block in sse.split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ") and '"message_delta"' in line
    )
    assert message_delta["usage"]["output_tokens"] == 5
    assert message_delta["usage"]["input_tokens"] == 10


def test_delayed_usage_chunk_preserves_cache_tokens():
    usage = Usage(
        prompt_tokens=120,
        completion_tokens=5,
        total_tokens=125,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=30,
            cache_creation_tokens=20,
        ),
    )
    chunks = [
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Two."),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(),
                    finish_reason="stop",
                )
            ],
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(),
                    finish_reason=None,
                )
            ],
            usage=usage,
        ),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="gpt-4o")
    events = list(wrapper)

    message_delta = next(
        event for event in events if event.get("type") == "message_delta"
    )

    assert message_delta["usage"]["input_tokens"] == 70
    assert message_delta["usage"]["output_tokens"] == 5
    assert message_delta["usage"]["cache_read_input_tokens"] == 30
    assert message_delta["usage"]["cache_creation_input_tokens"] == 20


def test_splitter_passes_through_non_combined_chunks():
    """A chunk with content but no finish_reason is not split."""
    chunk = ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0, delta=Delta(content="partial"), finish_reason=None
            )
        ]
    )
    chunks = list(_CombinedChunkSplitter(iter([chunk])))
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "partial"


def test_splitter_splits_combined_chunk_into_content_then_finish():
    """A chunk with both content and finish_reason becomes two chunks."""
    chunk = ModelResponseStream(
        choices=[
            StreamingChoices(index=0, delta=Delta(content="done"), finish_reason="stop")
        ]
    )
    content_chunk, finish_chunk = list(_CombinedChunkSplitter(iter([chunk])))

    assert content_chunk.choices[0].delta.content == "done"
    assert content_chunk.choices[0].finish_reason is None

    assert finish_chunk.choices[0].finish_reason == "stop"
    assert finish_chunk.choices[0].delta.content is None


def test_is_combined_false_when_choices_empty():
    """A metadata-only chunk with no choices is never treated as combined."""
    assert _CombinedChunkSplitter._is_combined(SimpleNamespace(choices=[])) is False


def test_is_combined_false_when_delta_missing():
    """A finish chunk whose choice has no delta is not combined."""
    chunk = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", delta=None)])
    assert _CombinedChunkSplitter._is_combined(chunk) is False


def test_split_clears_reasoning_and_thinking_on_finish_chunk():
    """When the combined delta carries reasoning/thinking, only the content
    chunk keeps them — the finish chunk is cleared."""
    delta = SimpleNamespace(
        content="hi",
        tool_calls=None,
        reasoning_content="some reasoning",
        thinking_blocks=[{"type": "thinking"}],
    )
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", delta=delta)]
    )

    content_chunk, finish_chunk = _CombinedChunkSplitter._split(chunk)

    assert content_chunk.choices[0].delta.reasoning_content == "some reasoning"
    assert content_chunk.choices[0].delta.thinking_blocks == [{"type": "thinking"}]
    assert finish_chunk.choices[0].delta.reasoning_content is None
    assert finish_chunk.choices[0].delta.thinking_blocks is None


def _thinking_delta_chunk(thinking: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    reasoning_content=thinking,
                    thinking_blocks=[{"type": "thinking", "thinking": thinking, "signature": None}],
                    provider_specific_fields={
                        "thinking_blocks": [{"type": "thinking", "thinking": thinking, "signature": None}]
                    },
                ),
                finish_reason=None,
            )
        ],
    )


def _signature_chunk(recap: str, signature: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    reasoning_content=recap,
                    thinking_blocks=[{"type": "thinking", "thinking": recap, "signature": signature}],
                    provider_specific_fields={
                        "thinking_blocks": [{"type": "thinking", "thinking": recap, "signature": signature}]
                    },
                ),
                finish_reason=None,
            )
        ],
    )


def test_thinking_then_signature_chunk_does_not_crash_stream():
    """Regression for the /v1/messages streaming crash reported on autoroute.

    Anthropic streams extended thinking as incremental ``thinking_delta`` chunks, then a
    closing chunk that recaps the full accumulated thinking AND carries the signature. The
    adapter used to raise ``ValueError`` on that closing chunk, killing the whole stream. It
    must instead emit a single ``signature_delta`` for the recap chunk and never re-emit the
    recap thinking, so the incremental thinking text is not duplicated.
    """
    chunks = [
        _thinking_delta_chunk("First, "),
        _thinking_delta_chunk("reason."),
        _signature_chunk("First, reason.", "sig-abc"),
        ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content="Done"), finish_reason=None)],
        ),
        ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="stop")],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        ),
    ]

    async def _aiter() -> "AsyncIterator[ModelResponseStream]":
        for chunk in chunks:
            yield chunk

    wrapper = AnthropicStreamWrapper(completion_stream=_aiter(), model="claude-haiku-4-5")
    sse = _collect_async(wrapper)

    signature_deltas = [
        json.loads(line[len("data: ") :])
        for block in sse.split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ") and '"signature_delta"' in line
    ]
    assert len(signature_deltas) == 1
    assert signature_deltas[0]["delta"]["signature"] == "sig-abc"

    thinking_text = "".join(
        json.loads(line[len("data: ") :])["delta"]["thinking"]
        for block in sse.split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ") and '"thinking_delta"' in line
    )
    assert thinking_text == "First, reason."

    assert "message_stop" in sse
    assert "Done" in sse
