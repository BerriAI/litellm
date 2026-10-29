from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Final

import pytest

from litellm.completion_extras.litellm_interactions_transformation import (
    InteractionsChatStreamingIterator,
    transform_chat_to_interactions_request,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
    stream_chunk_builder,  # pyright: ignore[reportUnknownVariableType]  # upstream signature uses bare list
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import CallTypes, Choices, Usage


def _logging_stub(model: str) -> Logging:
    return Logging(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type=CallTypes.completion.value,
        start_time=datetime(2026, 10, 22, tzinfo=UTC),
        litellm_call_id="call-stream-test",
        function_id="fn-stream-test",
    )


def test_sync_stream_preserves_initial_step_start_text_and_keeps_signature_out_of_reasoning() -> None:
    sse_lines: Final = (
        "event: interaction.created",
        'data: {"event_type": "interaction.created", "interaction": {"id": "v1_stream_99", "status": "in_progress"}}',
        'data: {"event_type": "step.start", "index": 0, "step": {"type": "thought"}}',
        'data: {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary", "content": {"type": "text", "text": " step 1 "}}}',
        'data: {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "CRYPTO_SIG_SECRET"}}',
        'data: {"event_type": "step.stop", "index": 0}',
        'data: {"event_type": "step.start", "index": 1, "step": {"type": "model_output", "content": [{"type": "text", "text": "Hello"}]}}',
        'data: {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": " [DONE] world"}}',
        'data: {"event_type": "interaction.completed", "interaction": {"id": "v1_stream_99", "status": "completed", "usage": {"total_input_tokens": 15, "total_output_tokens": 10, "total_thought_tokens": 25, "total_tokens": 50}}}',
        "data: [DONE]",
    )

    iterator: Final = InteractionsChatStreamingIterator(
        model="gemini-3-flash-preview",
        sync_lines=iter(sse_lines),
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=iterator,
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        logging_obj=_logging_stub("gemini-3-flash-preview"),
        stream_options={"include_usage": True},
    )
    chunks: Final = list(wrapper)
    rebuilt: Final = stream_chunk_builder(chunks)  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]  # stream_chunk_builder parameter lacks element type
    assert rebuilt is not None
    assert rebuilt.id == "v1_stream_99"
    choice: Final = rebuilt.choices[0]
    assert isinstance(choice, Choices)
    assert choice.finish_reason == "stop"
    assert choice.message.content == "Hello [DONE] world"
    assert choice.message.reasoning_content == " step 1 "
    assert "CRYPTO_SIG_SECRET" not in (choice.message.reasoning_content or "")
    assert choice.message.thinking_blocks == [
        {"type": "thinking", "thinking": " step 1 ", "signature": "CRYPTO_SIG_SECRET"}
    ]
    usage: Final = rebuilt.model_extra.get("usage") if rebuilt.model_extra else None
    assert isinstance(usage, Usage)
    assert usage.prompt_tokens == 15
    assert usage.completion_tokens == 35
    assert usage.total_tokens == 50


@pytest.mark.asyncio
async def test_async_parallel_tool_call_stream_rebuilds_and_roundtrips_turn2() -> None:
    sse_lines: Final = (
        'data: {"event_type": "interaction.created", "interaction": {"object": "interaction", "model": "gemini-3-flash-preview"}}',
        'data: {"event_type": "step.start", "index": 0, "step": {"type": "thought"}}',
        'data: {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "STREAM_THOUGHT_SIG"}}',
        'data: {"event_type": "step.stop", "index": 0}',
        'data: {"event_type": "step.start", "index": 1, "step": {"type": "model_output", "content": [{"type": "text", "text": "Checking"}]}}',
        'data: {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": " weather."}}',
        'data: {"event_type": "step.stop", "index": 1}',
        'data: {"event_type": "step.start", "index": 2, "step": {"id": "call_1", "type": "function_call", "name": "get_weather"}}',
        'data: {"event_type": "step.delta", "index": 2, "delta": {"arguments": "{\\"city\\":\\"Tok", "type": "arguments_delta"}}',
        'data: {"event_type": "step.delta", "index": 2, "delta": {"arguments": "yo\\"}", "type": "arguments_delta"}}',
        'data: {"event_type": "step.stop", "index": 2}',
        'data: {"event_type": "step.start", "index": 3, "step": {"id": "call_2", "type": "function_call", "name": "get_weather"}}',
        'data: {"event_type": "step.delta", "index": 3, "delta": {"arguments": "{\\"city\\":\\"Paris\\"}", "type": "arguments_delta"}}',
        'data: {"event_type": "step.stop", "index": 3}',
        'data: {"event_type": "interaction.completed", "interaction": {"status": "requires_action", "usage": {"total_tokens": 63, "total_input_tokens": 47, "total_output_tokens": 16}}}',
        "data: [DONE]",
    )

    async def line_gen() -> AsyncIterator[str]:
        for item in sse_lines:
            yield item

    iterator: Final = InteractionsChatStreamingIterator(
        model="gemini-3-flash-preview",
        async_lines=line_gen(),
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=iterator,
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        logging_obj=_logging_stub("gemini-3-flash-preview"),
    )
    chunks: Final = [chunk async for chunk in wrapper]
    rebuilt: Final = stream_chunk_builder(chunks)  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]  # stream_chunk_builder parameter lacks element type
    assert rebuilt is not None
    choice: Final = rebuilt.choices[0]
    assert isinstance(choice, Choices)
    assert choice.finish_reason == "tool_calls"
    tool_calls: Final = choice.message.tool_calls or []
    assert len(tool_calls) == 2

    turn2: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Weather in Tokyo and Paris?"},
            choice.message,
            {"role": "tool", "tool_call_id": tool_calls[0].id, "content": '{"temp": 20}'},
            {"role": "tool", "tool_call_id": tool_calls[1].id, "content": '{"temp": 15}'},
        ],
        model="gemini-3-flash-preview",
        optional_params={},
        litellm_params={},
    )
    assert turn2.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Weather in Tokyo and Paris?"}]},
        {"type": "thought", "signature": "STREAM_THOUGHT_SIG"},
        {"type": "model_output", "content": [{"type": "text", "text": "Checking weather."}]},
        {
            "type": "function_call",
            "id": "call_1",
            "name": "get_weather",
            "arguments": {"city": "Tokyo"},
        },
        {
            "type": "function_call",
            "id": "call_2",
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        },
        {
            "type": "function_result",
            "call_id": "call_1",
            "name": "get_weather",
            "result": [{"type": "text", "text": '{"temp": 20}'}],
        },
        {
            "type": "function_result",
            "call_id": "call_2",
            "name": "get_weather",
            "result": [{"type": "text", "text": '{"temp": 15}'}],
        },
    ]
