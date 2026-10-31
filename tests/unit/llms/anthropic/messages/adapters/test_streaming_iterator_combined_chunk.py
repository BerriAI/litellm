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
from typing import AsyncIterator, Final

import anthropic
import httpx
import pytest

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_streaming_response,
    convert_to_streaming_response_async,
)
from litellm.llms.anthropic.messages.adapters.streaming_iterator import (
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


def _build_fake_stream(content: str, finish_reason: str = "stop") -> MockResponseIterator:
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

    message_delta = next(event for event in events if event.get("type") == "message_delta")

    assert message_delta["usage"]["input_tokens"] == 70
    assert message_delta["usage"]["output_tokens"] == 5
    assert message_delta["usage"]["cache_read_input_tokens"] == 30
    assert message_delta["usage"]["cache_creation_input_tokens"] == 20


def _cached_bedrock_response(cache_read: int, cache_write: int) -> dict:
    return {
        "id": "chatcmpl-cached",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "us.anthropic.claude-opus-4-7",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
        "usage": Usage(
            prompt_tokens=42145,
            completion_tokens=6,
            total_tokens=42151,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ).model_dump(),
    }


async def _replay_async(cached: dict) -> list[ModelResponseStream]:
    return [chunk async for chunk in convert_to_streaming_response_async(cached)]


@pytest.mark.parametrize("cache_read, cache_write", [(42139, 0), (0, 42139)], ids=["read", "write"])
@pytest.mark.parametrize("replay", ["async", "sync"])
def test_a_cache_hit_replayed_as_a_messages_stream_keeps_the_prompt_cache_tokens(replay, cache_read, cache_write):
    cached: Final = _cached_bedrock_response(cache_read, cache_write)
    chunks: Final = (
        asyncio.run(_replay_async(cached)) if replay == "async" else list(convert_to_streaming_response(cached))
    )

    events: Final = list(AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-opus-4-7"))
    message_delta: Final = next(event for event in events if event.get("type") == "message_delta")

    assert message_delta["usage"]["input_tokens"] == 6
    assert message_delta["usage"].get("cache_read_input_tokens", 0) == cache_read
    assert message_delta["usage"].get("cache_creation_input_tokens", 0) == cache_write


def test_splitter_passes_through_non_combined_chunks():
    """A chunk with content but no finish_reason is not split."""
    chunk = ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content="partial"), finish_reason=None)])
    chunks = list(_CombinedChunkSplitter(iter([chunk])))
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "partial"


def test_splitter_splits_combined_chunk_into_content_then_finish():
    """A chunk with both content and finish_reason becomes two chunks."""
    chunk = ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content="done"), finish_reason="stop")])
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
    chunk = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", delta=delta)])

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


def _fake_stream_with_thinking(thinking_blocks: list[dict[str, str]]) -> MockResponseIterator:
    model_response: Final = ModelResponse()
    model_response.choices = [
        Choices(
            index=0,
            message=Message(
                role="assistant",
                content='{"a": 1}',
                reasoning_content="".join(block.get("thinking", "") for block in thinking_blocks),
                thinking_blocks=thinking_blocks,
            ),
            finish_reason="stop",
        )
    ]
    model_response.usage = Usage(prompt_tokens=5, completion_tokens=7, total_tokens=12)
    return MockResponseIterator(model_response=model_response)


def _anthropic_sdk_final_content(
    completion_stream: "MockResponseIterator | AsyncIterator[ModelResponseStream]",
) -> list[dict[str, object]]:
    sse: Final = _collect_async(AnthropicStreamWrapper(completion_stream=completion_stream, model="claude"))
    transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
    )
    client: Final = anthropic.Anthropic(api_key="transport-only", http_client=httpx.Client(transport=transport))
    with client.messages.stream(
        model="claude", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        return [block.model_dump(exclude_none=True) for block in stream.get_final_message().content]


@pytest.mark.parametrize(
    "thinking_blocks",
    [
        [
            {"type": "thinking", "thinking": "first", "signature": "SIG_A"},
            {"type": "thinking", "thinking": "second", "signature": "SIG_B"},
        ],
        [
            {"type": "redacted_thinking", "data": "OPAQUE"},
            {"type": "thinking", "thinking": "visible", "signature": "SIG_C"},
        ],
        [
            {"type": "thinking", "thinking": "unsigned", "signature": ""},
            {"type": "thinking", "thinking": "signed", "signature": "SIG_D"},
        ],
    ],
    ids=["two_signed_thinking_blocks", "redacted_then_signed_thinking", "unsigned_then_signed_thinking"],
)
def test_fake_stream_gives_each_thinking_block_its_own_content_block(thinking_blocks: list[dict[str, str]]):
    content: Final = _anthropic_sdk_final_content(_fake_stream_with_thinking(thinking_blocks))

    assert content == [*thinking_blocks, {"type": "text", "text": '{"a": 1}'}]


def _native_thinking_chunk(thinking: str, index: int, signature: str | None = None) -> ModelResponseStream:
    block: Final = {"type": "thinking", "thinking": thinking, "index": index, **({"signature": signature} if signature else {})}
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(reasoning_content=thinking or None, thinking_blocks=[block]))]
    )


async def _native_stream_with_two_thinking_blocks() -> "AsyncIterator[ModelResponseStream]":
    for chunk in (
        _native_thinking_chunk("ab", 0),
        _native_thinking_chunk("cd", 0),
        _native_thinking_chunk("", 0, "SIG_A"),
        _native_thinking_chunk("ef", 1),
        _native_thinking_chunk("gh", 1),
        _native_thinking_chunk("", 1, "SIG_B"),
        ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content="hi"))]),
        ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
    ):
        yield chunk


def test_native_stream_thinking_blocks_keep_their_text_once():
    content: Final = _anthropic_sdk_final_content(_native_stream_with_two_thinking_blocks())

    assert content == [
        {"type": "thinking", "thinking": "abcd", "signature": "SIG_A"},
        {"type": "thinking", "thinking": "efgh", "signature": "SIG_B"},
        {"type": "text", "text": "hi"},
    ]
