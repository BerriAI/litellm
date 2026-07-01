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


def test_split_separates_reasoning_content_and_finish():
    """A delta carrying reasoning + content + finish_reason splits three ways:
    a reasoning-only chunk, then a content-only chunk, then a finish-only chunk.

    Reasoning must come first (Anthropic requires the thinking block before the
    text block), and reasoning must never share a chunk with content — otherwise
    the wrapper opens a text block and emits a thinking_delta into it, which
    Anthropic clients reject with "Content block is not a thinking block"."""
    delta = SimpleNamespace(
        content="hi",
        tool_calls=None,
        reasoning_content="some reasoning",
        thinking_blocks=[{"type": "thinking"}],
    )
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", delta=delta)]
    )

    reasoning_chunk, content_chunk, finish_chunk = _CombinedChunkSplitter._split(chunk)

    assert reasoning_chunk.choices[0].delta.reasoning_content == "some reasoning"
    assert reasoning_chunk.choices[0].delta.thinking_blocks == [{"type": "thinking"}]
    assert reasoning_chunk.choices[0].delta.content is None
    assert reasoning_chunk.choices[0].finish_reason is None

    assert content_chunk.choices[0].delta.content == "hi"
    assert content_chunk.choices[0].delta.reasoning_content is None
    assert content_chunk.choices[0].delta.thinking_blocks is None
    assert content_chunk.choices[0].finish_reason is None

    assert finish_chunk.choices[0].finish_reason == "stop"
    assert finish_chunk.choices[0].delta.content is None
    assert finish_chunk.choices[0].delta.reasoning_content is None


def _open_block_type_by_index(events):
    """Map content-block index -> the type it was opened with."""
    return {
        event["index"]: event["content_block"]["type"]
        for event in events
        if event.get("type") == "content_block_start"
    }


def _assert_no_delta_block_mismatch(events):
    """Every thinking/signature delta must land in a thinking block and every
    text delta in a text block. A mismatch is exactly what makes Anthropic
    clients (e.g. Claude Code) raise "Content block is not a thinking block"."""
    open_types = _open_block_type_by_index(events)
    for event in events:
        if event.get("type") != "content_block_delta":
            continue
        delta_type = event["delta"]["type"]
        block_type = open_types.get(event["index"])
        if delta_type in ("thinking_delta", "signature_delta"):
            assert block_type == "thinking", (
                f"{delta_type} emitted into a {block_type!r} block: {event}"
            )
        elif delta_type == "text_delta":
            assert block_type == "text", (
                f"text_delta emitted into a {block_type!r} block: {event}"
            )


def _reasoning_then_answer_chunks():
    """Mimic a vLLM reasoning model whose boundary chunk carries the last
    reasoning token together with the first answer token (verified live with
    hosted_vllm/qwen)."""
    return [
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0, delta=Delta(reasoning_content="Let me think."), finish_reason=None
                )
            ]
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(reasoning_content="\n", content="Hi there!"),
                    finish_reason=None,
                )
            ]
        ),
        ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="stop")]
        ),
    ]


def _text_deltas(events):
    return [
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event["delta"].get("type") == "text_delta"
    ]


def _thinking_deltas(events):
    return [
        event["delta"]["thinking"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event["delta"].get("type") == "thinking_delta"
    ]


def test_boundary_reasoning_content_chunk_sync():
    """Sync: a boundary chunk with reasoning+content must not emit a
    thinking_delta into a text block, and must not drop the answer text."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_reasoning_then_answer_chunks()), model="qwen"
    )
    events = [event for event in wrapper if isinstance(event, dict)]

    _assert_no_delta_block_mismatch(events)
    assert "".join(_thinking_deltas(events)) == "Let me think.\n"
    assert "".join(_text_deltas(events)) == "Hi there!"


def test_boundary_reasoning_content_chunk_async():
    """Async: same guarantees as the sync path."""

    class _AsyncStream:
        def __init__(self, chunks):
            self._it = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    async def _run():
        wrapper = AnthropicStreamWrapper(
            completion_stream=_AsyncStream(_reasoning_then_answer_chunks()), model="qwen"
        )
        return [event async for event in wrapper if isinstance(event, dict)]

    events = asyncio.run(_run())

    _assert_no_delta_block_mismatch(events)
    assert "".join(_thinking_deltas(events)) == "Let me think.\n"
    assert "".join(_text_deltas(events)) == "Hi there!"
