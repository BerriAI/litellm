import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Final

import httpx
import pytest

from litellm.completion_extras.litellm_interactions_transformation import (
    InteractionsChatStreamingIterator,
    transform_chat_to_interactions_request,
)
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    MidStreamFallbackError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
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


@pytest.mark.parametrize(
    ("error_line", "expected_substr"),
    (
        (
            'data: {"event_type": "error", "error": {"code": "internal_error", "message": "Backend stream aborted"}}',
            "[internal_error] Backend stream aborted",
        ),
        (
            'data: {"event_type": "interaction.failed", "interaction": {"id": "v1_fail", "status": "failed", "error": {"code": "deadline_exceeded", "message": "Upstream timed out"}}}',
            "[deadline_exceeded] Upstream timed out",
        ),
        (
            'data: {"event_type": "interaction.completed", "interaction": {"id": "v1_fail2", "status": "failed", "error": {"message": "Completed in failed state"}}}',
            "Completed in failed state",
        ),
        (
            'data: {"event_type": "interaction.status_update", "status": "failed", "error": {"code": "internal", "message": "Status update failed"}}',
            "[internal] Status update failed",
        ),
        (
            'data: {"event_type": "interaction.completed", "interaction": {"id": "v1_cancel", "status": "cancelled"}}',
            "[cancelled] Interaction stream was cancelled",
        ),
    ),
)
def test_sync_mid_stream_error_events_close_response_and_raise(
    error_line: str,
    expected_substr: str,
) -> None:
    sse_lines: Final = (
        'data: {"event_type": "interaction.created", "interaction": {"id": "v1_err_sync", "status": "in_progress"}}',
        'data: {"event_type": "step.start", "index": 0, "step": {"type": "model_output", "content": [{"type": "text", "text": "Partial"}]}}',
        error_line,
    )
    sync_resp: Final = httpx.Response(200, stream=httpx.ByteStream(b""))
    iterator: Final = InteractionsChatStreamingIterator(
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        sync_lines=iter(sse_lines),
        http_response=sync_resp,
    )

    first_chunk: Final = next(iterator)
    assert first_chunk.choices[0].delta.content == "Partial"
    with pytest.raises(APIConnectionError, match=re.escape(expected_substr)) as exc_info:
        next(iterator)
    assert getattr(exc_info.value, "llm_provider", None) == "gemini"
    assert getattr(exc_info.value, "model", None) == "gemini-3-flash-preview"
    assert sync_resp.is_stream_consumed or sync_resp.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_line", "expected_substr"),
    (
        (
            'data: {"event_type": "error", "error": {"code": "unavailable", "message": "Worker reset"}}',
            "Worker reset",
        ),
        (
            'data: {"event_type": "interaction.failed", "error": {"code": "aborted", "message": "Generation failed"}}',
            "Generation failed",
        ),
    ),
)
async def test_async_mid_stream_error_events_surface_through_stream_wrapper(
    error_line: str,
    expected_substr: str,
) -> None:
    sse_lines: Final = (
        'data: {"event_type": "interaction.created", "interaction": {"id": "v1_err_async", "status": "in_progress"}}',
        'data: {"event_type": "step.start", "index": 0, "step": {"type": "model_output", "content": [{"type": "text", "text": "Start"}]}}',
        error_line,
    )

    async def line_gen() -> AsyncIterator[str]:
        for item in sse_lines:
            yield item

    async_resp: Final = httpx.Response(200, stream=httpx.AsyncByteStream())
    iterator: Final = InteractionsChatStreamingIterator(
        model="gemini-3-flash-preview",
        custom_llm_provider="vertex_ai",
        async_lines=line_gen(),
        http_response=async_resp,
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=iterator,
        model="gemini-3-flash-preview",
        custom_llm_provider="vertex_ai",
        logging_obj=_logging_stub("gemini-3-flash-preview"),
    )

    first_chunk: Final = await anext(wrapper)
    assert first_chunk.choices[0].delta.content == "Start"
    with pytest.raises(MidStreamFallbackError, match=re.escape(expected_substr)):
        await anext(wrapper)
    assert async_resp.is_closed


def test_late_thought_signature_delta_backfills_tool_call_id_and_unsigned_thought_resets_state() -> None:
    late_sig_lines: Final = (
        'data: {"event_type": "interaction.created", "interaction": {"id": "v1_late_sig", "status": "in_progress"}}',
        'data: {"event_type": "step.start", "index": 0, "step": {"type": "thought", "signature": ""}}',
        'data: {"event_type": "step.start", "index": 1, "step": {"id": "call_late", "type": "function_call", "name": "bash"}}',
        'data: {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": "{\\"cmd\\":\\"ls\\"}"}}',
        'data: {"event_type": "step.start", "index": 2, "step": {"id": "call_second", "type": "function_call", "name": "bash"}}',
        'data: {"event_type": "step.delta", "index": 2, "delta": {"type": "arguments_delta", "arguments": "{\\"cmd\\":\\"pwd\\"}"}}',
        'data: {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "LATE_SIG_XYZ"}}',
        'data: {"event_type": "interaction.completed", "interaction": {"id": "v1_late_sig", "status": "requires_action"}}',
        "data: [DONE]",
    )
    late_iter: Final = InteractionsChatStreamingIterator(
        model="gemini-3-flash-preview",
        sync_lines=iter(late_sig_lines),
    )
    late_wrapper: Final = CustomStreamWrapper(
        completion_stream=late_iter,
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        logging_obj=_logging_stub("gemini-3-flash-preview"),
    )
    rebuilt_late: Final = stream_chunk_builder(list(late_wrapper))  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]  # stream_chunk_builder parameter lacks element type
    assert rebuilt_late is not None
    late_choice: Final = rebuilt_late.choices[0]
    assert isinstance(late_choice, Choices)
    late_calls: Final = late_choice.message.tool_calls or []
    assert [tc.id for tc in late_calls] == ["call_late__thought__LATE_SIG_XYZ", "call_second"]

    non_leak_lines: Final = (
        'data: {"event_type": "interaction.created", "interaction": {"id": "v1_noleak", "status": "in_progress"}}',
        'data: {"event_type": "step.start", "index": 0, "step": {"type": "thought", "signature": "STALE_SIG_1"}}',
        'data: {"event_type": "step.start", "index": 1, "step": {"id": "call_1", "type": "function_call", "name": "bash"}}',
        'data: {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": "{}"}}',
        'data: {"event_type": "step.start", "index": 2, "step": {"type": "function_result", "call_id": "call_1", "name": "bash"}}',
        'data: {"event_type": "step.start", "index": 3, "step": {"type": "thought", "signature": "", "summary": [{"type": "text", "text": "step two"}]}}',
        'data: {"event_type": "step.start", "index": 4, "step": {"id": "call_2", "type": "function_call", "name": "bash"}}',
        'data: {"event_type": "step.delta", "index": 4, "delta": {"type": "arguments_delta", "arguments": "{}"}}',
        'data: {"event_type": "interaction.completed", "interaction": {"id": "v1_noleak", "status": "requires_action"}}',
        "data: [DONE]",
    )
    noleak_iter: Final = InteractionsChatStreamingIterator(
        model="gemini-3-flash-preview",
        sync_lines=iter(non_leak_lines),
    )
    noleak_wrapper: Final = CustomStreamWrapper(
        completion_stream=noleak_iter,
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        logging_obj=_logging_stub("gemini-3-flash-preview"),
    )
    rebuilt_noleak: Final = stream_chunk_builder(list(noleak_wrapper))  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]  # stream_chunk_builder parameter lacks element type
    assert rebuilt_noleak is not None
    noleak_choice: Final = rebuilt_noleak.choices[0]
    assert isinstance(noleak_choice, Choices)
    noleak_calls: Final = noleak_choice.message.tool_calls or []
    assert [tc.id for tc in noleak_calls] == ["call_1__thought__STALE_SIG_1", "call_2"]


@pytest.mark.parametrize(
    ("code", "expected_cls"),
    (
        ("invalid_argument", BadRequestError),
        ("unauthenticated", AuthenticationError),
        ("permission_denied", PermissionDeniedError),
        ("not_found", NotFoundError),
    ),
)
def test_sse_4xx_error_codes_surface_directly_through_stream_wrapper(
    code: str,
    expected_cls: type[Exception],
) -> None:
    sse_lines: Final = (
        'data: {"event_type": "interaction.created", "interaction": {"id": "v1_code_map"}}',
        f'data: {{"event_type": "error", "error": {{"code": "{code}", "message": "Fault {code}"}}}}',
    )
    iterator: Final = InteractionsChatStreamingIterator(
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        sync_lines=iter(sse_lines),
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=iterator,
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        logging_obj=_logging_stub("gemini-3-flash-preview"),
    )

    with pytest.raises(expected_cls) as direct_exc:
        next(wrapper)
    assert not isinstance(direct_exc.value, MidStreamFallbackError)


def test_sse_resource_exhausted_wraps_rate_limit_error_for_router_fallback() -> None:
    sse_lines: Final = (
        'data: {"event_type": "interaction.created", "interaction": {"id": "v1_429_map"}}',
        'data: {"event_type": "error", "error": {"code": "resource_exhausted", "message": "Quota hit"}}',
    )
    iterator: Final = InteractionsChatStreamingIterator(
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        sync_lines=iter(sse_lines),
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=iterator,
        model="gemini-3-flash-preview",
        custom_llm_provider="gemini",
        logging_obj=_logging_stub("gemini-3-flash-preview"),
    )

    with pytest.raises(MidStreamFallbackError) as wrapped_exc:
        next(wrapper)
    assert isinstance(wrapped_exc.value.original_exception, RateLimitError)


def test_multisegment_late_thought_signature_deltas_backfill_matching_tool_slots() -> None:
    sse_lines: Final = (
        'data: {"event_type": "interaction.created", "interaction": {"id": "v1_multiseg"}}',
        'data: {"event_type": "step.start", "index": 0, "step": {"type": "thought", "signature": ""}}',
        'data: {"event_type": "step.start", "index": 1, "step": {"id": "call_a", "type": "function_call", "name": "bash"}}',
        'data: {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": "{\\"cmd\\":\\"ls a\\"}"}}',
        'data: {"event_type": "step.start", "index": 2, "step": {"type": "thought", "signature": ""}}',
        'data: {"event_type": "step.start", "index": 3, "step": {"id": "call_b", "type": "function_call", "name": "bash"}}',
        'data: {"event_type": "step.delta", "index": 3, "delta": {"type": "arguments_delta", "arguments": "{\\"cmd\\":\\"ls b\\"}"}}',
        'data: {"event_type": "step.delta", "index": 2, "delta": {"type": "thought_signature", "signature": "SIG_SEG_B"}}',
        'data: {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "SIG_SEG_A"}}',
        'data: {"event_type": "interaction.completed", "interaction": {"id": "v1_multiseg", "status": "requires_action"}}',
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
    )
    rebuilt: Final = stream_chunk_builder(list(wrapper))  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]  # stream_chunk_builder parameter lacks element type
    assert rebuilt is not None
    choice: Final = rebuilt.choices[0]
    assert isinstance(choice, Choices)
    tool_calls: Final = choice.message.tool_calls or []
    assert [tc.id for tc in tool_calls] == ["call_a__thought__SIG_SEG_A", "call_b__thought__SIG_SEG_B"]
