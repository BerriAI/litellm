"""
Tests for ``streaming_buffer_until_moderated`` on the unified guardrail
post-call streaming iterator hook.

With this flag set, the hook must withhold every upstream chunk until
end-of-stream moderation has run. The decisive guarantee versus the
detect-only ``streaming_end_of_stream_only`` behavior: when the guardrail
blocks, the original (objectionable) content is NEVER yielded to the client --
only the block message is. On a clean response, all original chunks are
released unchanged after moderation passes.
"""

import json
from typing import Any, AsyncGenerator, List, Literal, Optional

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.llms.base_llm.guardrail_translation.base_translation import StreamingScanKey
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
    _is_redundant_scan,
)
from litellm.proxy.utils import _pipelines_stream_live
from litellm.types.proxy.policy_engine.pipeline_types import GuardrailPipeline, PipelineStep
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    ChatCompletionToolCallChunk,
    Delta,
    Function,
    FunctionCall,
    GenericGuardrailAPIInputs,
    ModelResponseStream,
    StreamingChoices,
)

BLOCK_MESSAGE = "Blocked by policy: this response was withheld."
ORIGINAL_MARKER = "ORIGINAL-SECRET-ANSWER"
TOOL_ARGUMENTS_MARKER = "TOOL-ARGS-SECRET"


class _BlockingGuardrail(CustomGuardrail):
    """Always blocks at moderation time."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        raise ModifyResponseException(
            message=BLOCK_MESSAGE,
            model="claude-3-5-sonnet",
            request_data=request_data,
            guardrail_name=self.guardrail_name,
        )


class _PassingGuardrail(CustomGuardrail):
    """Never blocks; returns inputs unchanged."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        return inputs


class _CountingPassingGuardrail(_PassingGuardrail):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_count = 0

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.scan_count += 1
        return inputs


class _ToolCallRecordingGuardrail(_CountingPassingGuardrail):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_call_scan_indexes: List[int] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.scan_count += 1
        if inputs.get("tool_calls"):
            self.tool_call_scan_indexes.append(self.scan_count)
        return inputs


class _SecondScanBlockingGuardrail(_CountingPassingGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.scan_count += 1
        if self.scan_count == 2:
            raise ModifyResponseException(
                message=BLOCK_MESSAGE,
                model="gpt-4",
                request_data=request_data,
                guardrail_name=self.guardrail_name,
            )
        return inputs


class _MarkerBlockingGuardrail(_CountingPassingGuardrail):
    """Blocks as soon as the inspected input field (texts or tool_calls) carries the marker."""

    def __init__(self, *args, marker: str, field: Literal["texts", "tool_calls"] = "texts", **kwargs):
        super().__init__(*args, **kwargs)
        self.marker = marker
        self.field = field

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.scan_count += 1
        if self.marker in json.dumps(inputs.get(self.field, [])):
            raise ModifyResponseException(
                message=BLOCK_MESSAGE,
                model="gpt-4o",
                request_data=request_data,
                guardrail_name=self.guardrail_name,
            )
        return inputs


def _sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def _anthropic_stream():
    """A complete Anthropic /v1/messages SSE stream whose assistant text
    contains ORIGINAL_MARKER so leakage is unambiguous to assert."""
    yield _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_orig",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    yield _sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    for text in ["Here is ", "the ", ORIGINAL_MARKER, " for you."]:
        yield _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


def _decode(chunks: List[Any]) -> str:
    return "".join(c.decode() if isinstance(c, bytes) else str(c) for c in chunks)


def _chat_chunk(content: str = "", finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-windowed",
        created=1724900000,
        model="gpt-4",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
    )


def _tool_call_chunk(
    arguments: str, finish_reason: str | None = None, legacy_function_call: bool = False
) -> ModelResponseStream:
    delta = (
        Delta(role="assistant", content=None, function_call=FunctionCall(name="run_shell", arguments=arguments))
        if legacy_function_call
        else Delta(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionDeltaToolCall(
                    id="call_1",
                    type="function",
                    index=0,
                    function=Function(name="run_shell", arguments=arguments),
                )
            ],
        )
    )
    return ModelResponseStream(
        id="chatcmpl-windowed",
        created=1724900000,
        model="gpt-4",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)],
    )


async def _windowed_chat_stream(
    yielded_count: List[int],
    collected: List[Any],
    content_chunks: List[str],
    tool_argument_chunks: List[str] | None = None,
    legacy_function_call: bool = False,
) -> AsyncGenerator[ModelResponseStream, None]:
    for content in content_chunks:
        yielded_count.append(len(collected))
        yield _chat_chunk(content)
    for arguments in tool_argument_chunks or []:
        yielded_count.append(len(collected))
        yield _tool_call_chunk(arguments, legacy_function_call=legacy_function_call)
    yielded_count.append(len(collected))
    yield _chat_chunk(finish_reason="tool_calls" if tool_argument_chunks else "stop")


def _tool_argument_text(chunks: List[Any]) -> str:
    return "".join(
        tool_call.function.arguments or ""
        for chunk in chunks
        if isinstance(chunk, ModelResponseStream)
        for choice in chunk.choices
        for tool_call in choice.delta.tool_calls or []
    )


def _function_call_argument_text(chunks: list[Any]) -> str:
    return "".join(
        choice.delta.function_call.arguments or ""
        for chunk in chunks
        if isinstance(chunk, ModelResponseStream)
        for choice in chunk.choices
        if choice.delta.function_call is not None
    )


async def _run_windowed(
    guardrail: CustomGuardrail,
    content_chunks: List[str],
    end_of_stream_only: bool = False,
    tool_argument_chunks: List[str] | None = None,
    legacy_function_call: bool = False,
) -> tuple[List[Any], List[int]]:
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = True
    guardrail.streaming_end_of_stream_only = end_of_stream_only
    guardrail.streaming_sampling_rate = 2
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/chat/completions")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: List[Any] = []
    yielded_count: List[int] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_windowed_chat_stream(
            yielded_count, collected, content_chunks, tool_argument_chunks, legacy_function_call
        ),
        request_data=request_data,
    ):
        collected.append(chunk)
    return collected, yielded_count


def _responses_message_stream_events(text_chunks: List[str]) -> List[dict]:
    message = {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant"}
    content = [{"type": "output_text", "text": "".join(text_chunks), "annotations": []}]
    return [
        {"type": "response.output_item.added", "output_index": 0, "item": {**message, "content": []}},
        *(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            }
            for text in text_chunks
        ),
        {"type": "response.output_item.done", "output_index": 0, "item": {**message, "content": content}},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-4o",
                "status": "completed",
                "output": [{**message, "content": content}],
            },
        },
    ]


def _responses_truncated_function_call_events(text: str, argument_chunks: List[str]) -> List[dict]:
    message = {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant"}
    content = [{"type": "output_text", "text": text, "annotations": []}]
    function_call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "run_shell"}
    return [
        {"type": "response.output_item.added", "output_index": 0, "item": {**message, "content": []}},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.output_item.added", "output_index": 1, "item": {**function_call, "arguments": ""}},
        *(
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 1, "delta": arguments}
            for arguments in argument_chunks
        ),
        {
            "type": "response.incomplete",
            "response": {
                "id": "resp_1",
                "model": "gpt-4o",
                "status": "incomplete",
                "output": [
                    {**message, "content": content},
                    {**function_call, "arguments": "".join(argument_chunks), "status": "incomplete"},
                ],
            },
        },
    ]


async def _replay(events: List[dict]) -> AsyncGenerator[dict, None]:
    for event in events:
        yield event


async def _run_windowed_responses(guardrail: CustomGuardrail, events: List[dict]) -> str:
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = True
    guardrail.streaming_sampling_rate = 2
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/responses")
    request_data = {
        "input": "hi",
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: List[Any] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_replay(events),
        request_data=request_data,
    ):
        collected.append(chunk)
    return json.dumps([chunk if isinstance(chunk, dict) else str(chunk) for chunk in collected])


def _chat_text(chunks: List[Any]) -> str:
    return "".join(
        choice.delta.content or ""
        for chunk in chunks
        if isinstance(chunk, ModelResponseStream)
        for choice in chunk.choices
    )


async def _run(guardrail: CustomGuardrail) -> str:
    # Rubrik's real config: end-of-stream-only moderation. Without buffering
    # this releases every chunk before moderation runs (content leaks on
    # block); the buffer flag must change that to moderate-then-release.
    guardrail.streaming_end_of_stream_only = True
    guardrail.streaming_buffer_until_moderated = True
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/messages")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: List[Any] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_anthropic_stream(),
        request_data=request_data,
    ):
        collected.append(chunk)
    return _decode(collected)


@pytest.mark.asyncio
async def test_buffered_block_withholds_original_content():
    raw = await _run(_BlockingGuardrail(guardrail_name="blk", event_hook="post_call"))
    # The original content must never reach the client...
    assert ORIGINAL_MARKER not in raw, f"original content leaked: {raw!r}"
    # ...only the block message, in a clean terminating stream.
    assert BLOCK_MESSAGE in raw
    assert '"error"' not in raw


@pytest.mark.asyncio
async def test_buffered_clean_releases_all_content():
    raw = await _run(_PassingGuardrail(guardrail_name="pass", event_hook="post_call"))
    # A clean response is released in full after moderation passes.
    assert ORIGINAL_MARKER in raw
    assert (
        raw.rstrip().endswith('event: message_stop\ndata: {"type": "message_stop"}'.rstrip()) or "message_stop" in raw
    )
    assert BLOCK_MESSAGE not in raw


@pytest.mark.asyncio
async def test_windowed_buffer_releases_after_each_passing_scan():
    guardrail = _CountingPassingGuardrail(guardrail_name="windowed-pass", event_hook="post_call")
    content_chunks = ["one ", "two ", "three ", "four ", "five ", "six "]

    collected, yielded_count = await _run_windowed(guardrail, content_chunks)

    assert yielded_count[2] >= 2
    assert yielded_count == [0, 0, 2, 2, 4, 4, 6]
    assert _chat_text(collected) == "".join(content_chunks)
    assert guardrail.scan_count > 1


@pytest.mark.asyncio
async def test_windowed_buffer_drops_blocked_window():
    guardrail = _SecondScanBlockingGuardrail(guardrail_name="windowed-block", event_hook="post_call")
    content_chunks = ["one ", "two ", "MARKER ", "four ", "five ", "six "]

    collected, _ = await _run_windowed(guardrail, content_chunks)
    raw = _decode(collected)

    assert _chat_text(collected) == "one two "
    assert "MARKER" not in raw
    assert BLOCK_MESSAGE in raw
    assert '"error"' not in raw


@pytest.mark.asyncio
async def test_windowed_buffer_holds_tool_call_windows_until_end_of_stream_scan():
    guardrail = _ToolCallRecordingGuardrail(guardrail_name="windowed-tools", event_hook="post_call")
    content_chunks = ["one ", "two ", "three "]
    tool_argument_chunks = ['{"cmd": "', TOOL_ARGUMENTS_MARKER, '"}']

    collected, yielded_count = await _run_windowed(guardrail, content_chunks, tool_argument_chunks=tool_argument_chunks)

    assert yielded_count == [0, 0, 2, 2, 2, 2, 2]
    assert _chat_text(collected) == "".join(content_chunks)
    assert _tool_argument_text(collected) == "".join(tool_argument_chunks)
    assert guardrail.tool_call_scan_indexes == [guardrail.scan_count]


@pytest.mark.asyncio
async def test_windowed_buffer_holds_legacy_function_call_windows_until_end_of_stream():
    guardrail = _PassingGuardrail(guardrail_name="windowed-functions", event_hook="post_call")
    content_chunks = ["one ", "two ", "three "]
    function_argument_chunks = ['{"cmd": "', TOOL_ARGUMENTS_MARKER, '"}']

    collected, yielded_count = await _run_windowed(
        guardrail, content_chunks, tool_argument_chunks=function_argument_chunks, legacy_function_call=True
    )

    assert yielded_count == [0, 0, 2, 2, 2, 2, 2]
    assert _chat_text(collected) == "".join(content_chunks)
    assert _function_call_argument_text(collected) == "".join(function_argument_chunks)


def test_tool_call_only_scan_key_is_not_skipped_as_empty():
    assert _is_redundant_scan(StreamingScanKey(texts=("",)), None) is True
    assert _is_redundant_scan(StreamingScanKey(texts=("",), tool_calls=("run_shell:{}",)), None) is False


@pytest.mark.asyncio
async def test_windowed_responses_output_item_done_round_keeps_text_window_withheld():
    guardrail = _MarkerBlockingGuardrail(
        guardrail_name="windowed-responses", event_hook="post_call", marker=ORIGINAL_MARKER
    )
    events = _responses_message_stream_events(["one ", f"{ORIGINAL_MARKER} "])

    raw = await _run_windowed_responses(guardrail, events)

    assert ORIGINAL_MARKER not in raw, f"unscanned window leaked: {raw!r}"
    assert BLOCK_MESSAGE in raw
    assert guardrail.scan_count >= 1


@pytest.mark.asyncio
async def test_windowed_responses_incomplete_stream_scans_tool_call_before_release():
    guardrail = _MarkerBlockingGuardrail(
        guardrail_name="windowed-responses-tools",
        event_hook="post_call",
        marker=TOOL_ARGUMENTS_MARKER,
        field="tool_calls",
    )
    events = _responses_truncated_function_call_events("hi ", ['{"cmd": "', TOOL_ARGUMENTS_MARKER, '"}'])

    raw = await _run_windowed_responses(guardrail, events)

    assert '"hi "' in raw
    assert TOOL_ARGUMENTS_MARKER not in raw, f"unscanned tool call leaked: {raw!r}"
    assert BLOCK_MESSAGE in raw


@pytest.mark.asyncio
async def test_windowed_buffer_with_explicit_end_of_stream_only_stays_fully_buffered():
    guardrail = _CountingPassingGuardrail(guardrail_name="windowed-eos", event_hook="post_call")
    content_chunks = ["one ", "two ", "three ", "four ", "five ", "six "]

    collected, yielded_count = await _run_windowed(guardrail, content_chunks, end_of_stream_only=True)

    assert yielded_count == [0, 0, 0, 0, 0, 0, 0]
    assert _chat_text(collected) == "".join(content_chunks)
    assert guardrail.scan_count == 1


MASKED_MARKER = "[MASKED]"


class _MaskingGuardrail(_CountingPassingGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.scan_count += 1
        masked: GenericGuardrailAPIInputs = {
            "texts": [text.replace(ORIGINAL_MARKER, MASKED_MARKER) for text in inputs.get("texts") or []]
        }
        tool_calls = inputs.get("tool_calls")
        if tool_calls:
            masked["tool_calls"] = [_mask_tool_call(tool_call) for tool_call in tool_calls]
        return masked


def _mask_tool_call(
    tool_call: ChatCompletionToolCallChunk | ChatCompletionMessageToolCall,
) -> ChatCompletionToolCallChunk | ChatCompletionMessageToolCall:
    if isinstance(tool_call, dict):
        function = tool_call["function"]
        arguments = function["arguments"].replace(TOOL_ARGUMENTS_MARKER, MASKED_MARKER)
        return {**tool_call, "function": {**function, "arguments": arguments}}
    tool_call.function.arguments = tool_call.function.arguments.replace(TOOL_ARGUMENTS_MARKER, MASKED_MARKER)
    return tool_call


class _ToolDroppingGuardrail(_MaskingGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        masked = await super().apply_guardrail(inputs, request_data, input_type, logging_obj)
        return {**masked, "tool_calls": []}


class _ToolClearingGuardrail(_MaskingGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        masked = await super().apply_guardrail(inputs, request_data, input_type, logging_obj)
        tool_calls = inputs.get("tool_calls")
        if tool_calls is not None:
            tool_calls.clear()
        return {**masked, "tool_calls": tool_calls or []}


class _ToolManglingGuardrail(_MaskingGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        masked = await super().apply_guardrail(inputs, request_data, input_type, logging_obj)
        return {**masked, "tool_calls": [{"type": "function"} for _ in inputs.get("tool_calls") or []]}


async def _anthropic_tool_use_stream():
    yield _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_tool",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    yield _sse_event(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )
    yield _sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"Sending {ORIGINAL_MARKER}"},
        },
    )
    yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "send_email", "input": {}},
        },
    )
    for partial in ['{"to": "', TOOL_ARGUMENTS_MARKER, '"}']:
        yield _sse_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": partial}},
        )
    yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 1})
    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


async def _run_messages_stream(guardrail: CustomGuardrail, stream: AsyncGenerator[bytes, None]) -> str:
    guardrail.streaming_end_of_stream_only = True
    guardrail.streaming_buffer_until_moderated = True
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/messages")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: list[Any] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=stream,
        request_data=request_data,
    ):
        collected.append(chunk)
    return _decode(collected)


async def _run_buffered_chat(
    guardrail: CustomGuardrail, content_chunks: list[str], tool_argument_chunks: list[str] | None = None
) -> list[Any]:
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = False
    guardrail.streaming_end_of_stream_only = False
    guardrail.streaming_sampling_rate = 2
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/chat/completions")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: list[Any] = []
    yielded_count: list[int] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_windowed_chat_stream(yielded_count, collected, content_chunks, tool_argument_chunks, False),
        request_data=request_data,
    ):
        collected.append(chunk)
    assert yielded_count == [0] * len(yielded_count), "buffered stream released a chunk before the end scan"
    return collected


@pytest.mark.asyncio
async def test_buffered_messages_stream_delivers_masked_text_to_the_client():
    guardrail = _MaskingGuardrail(guardrail_name="masker", event_hook="post_call", mask_response_content=True)
    raw = await _run(guardrail)
    assert guardrail.scan_count == 1
    assert MASKED_MARKER in raw, raw
    assert ORIGINAL_MARKER not in raw, raw
    assert BLOCK_MESSAGE not in raw


@pytest.mark.asyncio
async def test_buffered_messages_stream_delivers_masked_tool_call_input():
    guardrail = _MaskingGuardrail(guardrail_name="masker", event_hook="post_call")
    raw = await _run_messages_stream(guardrail, _anthropic_tool_use_stream())
    assert guardrail.scan_count == 1
    assert TOOL_ARGUMENTS_MARKER not in raw, raw
    assert ORIGINAL_MARKER not in raw, raw
    assert raw.count(MASKED_MARKER) == 2, raw
    assert "message_stop" in raw


@pytest.mark.asyncio
async def test_buffered_chat_stream_delivers_masked_text_and_tool_arguments():
    guardrail = _MaskingGuardrail(guardrail_name="masker", event_hook="post_call")
    collected = await _run_buffered_chat(
        guardrail, [f"call {ORIGINAL_MARKER} now"], ['{"cmd": "', TOOL_ARGUMENTS_MARKER, '"}']
    )
    assert _chat_text(collected) == f"call {MASKED_MARKER} now"
    assert _tool_argument_text(collected) == f'{{"cmd": "{MASKED_MARKER}"}}'
    assert collected[-1].choices[0].finish_reason == "tool_calls"


def _two_choice_tool_call_chunk(finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-two-choices",
        created=1724900000,
        model="gpt-4",
        choices=[
            StreamingChoices(
                index=index,
                finish_reason=finish_reason,
                delta=Delta(
                    role="assistant",
                    content=f"call {ORIGINAL_MARKER} now",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=f"call_{index}",
                            type="function",
                            index=0,
                            function=Function(name="run_shell", arguments=f'{{"cmd": "{TOOL_ARGUMENTS_MARKER}"}}'),
                        )
                    ],
                ),
            )
            for index in (0, 1)
        ],
    )


async def _two_choice_tool_call_stream() -> AsyncGenerator[ModelResponseStream, None]:
    yield _two_choice_tool_call_chunk()
    yield _two_choice_tool_call_chunk(finish_reason="tool_calls")


@pytest.mark.asyncio
async def test_buffered_messages_stream_withholds_everything_when_the_guardrail_drops_a_tool_call():
    guardrail = _ToolDroppingGuardrail(guardrail_name="dropper", event_hook="post_call")
    with pytest.raises(HTTPException) as raised:
        await _run_messages_stream(guardrail, _anthropic_tool_use_stream())
    assert raised.value.status_code == 500
    assert "dropper" in str(raised.value.detail)
    assert guardrail.scan_count == 1


@pytest.mark.asyncio
async def test_buffered_chat_stream_withholds_everything_when_the_tool_call_rewrite_cannot_be_placed():
    guardrail = _MaskingGuardrail(guardrail_name="masker", event_hook="post_call")
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = False
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/chat/completions")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: list[Any] = []

    async def drain() -> None:
        async for chunk in unified.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_api_key_dict, response=_two_choice_tool_call_stream(), request_data=request_data
        ):
            collected.append(chunk)

    with pytest.raises(HTTPException) as raised:
        await drain()
    assert raised.value.status_code == 500
    assert "masker" in str(raised.value.detail)
    assert guardrail.scan_count == 1
    assert collected == []


@pytest.mark.asyncio
async def test_buffered_responses_stream_delivers_masked_text_to_the_client():
    guardrail = _MaskingGuardrail(guardrail_name="masker", event_hook="post_call")
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = False
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/responses")
    request_data = {
        "input": "hi",
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: list[Any] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_replay(_responses_message_stream_events(["Here is ", ORIGINAL_MARKER])),
        request_data=request_data,
    ):
        collected.append(chunk)
    raw = json.dumps([chunk if isinstance(chunk, dict) else str(chunk) for chunk in collected])
    assert guardrail.scan_count == 1
    assert MASKED_MARKER in raw, raw
    assert ORIGINAL_MARKER not in raw, raw
    assert "response.completed" in raw


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "guardrail_cls", [_ToolDroppingGuardrail, _ToolManglingGuardrail], ids=["returns fewer", "returns a nameless call"]
)
async def test_buffered_responses_stream_withholds_everything_when_the_guardrail_drops_a_tool_call(guardrail_cls: type):
    guardrail = guardrail_cls(guardrail_name="dropper", event_hook="post_call")
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = False
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/responses")
    request_data = {
        "input": "hi",
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: list[Any] = []

    async def drain() -> None:
        async for chunk in unified.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_api_key_dict,
            response=_replay(
                _responses_truncated_function_call_events("Running", ['{"cmd": "', TOOL_ARGUMENTS_MARKER, '"}'])
            ),
            request_data=request_data,
        ):
            collected.append(chunk)

    with pytest.raises(HTTPException) as raised:
        await drain()

    assert raised.value.status_code == 500
    assert "dropper" in str(raised.value.detail)
    assert guardrail.scan_count == 1
    assert collected == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "guardrail_cls", [_ToolDroppingGuardrail, _ToolClearingGuardrail], ids=["returns fewer", "clears in place"]
)
async def test_buffered_chat_stream_withholds_everything_when_the_guardrail_drops_a_tool_call(guardrail_cls: type):
    guardrail = guardrail_cls(guardrail_name="dropper", event_hook="post_call")
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = False
    with pytest.raises(HTTPException) as raised:
        await _run_buffered_chat(guardrail, ["Running"], ['{"cmd": "', TOOL_ARGUMENTS_MARKER, '"}'])
    assert raised.value.status_code == 500
    assert "dropper" in str(raised.value.detail)
    assert guardrail.scan_count == 1


@pytest.mark.asyncio
async def test_live_messages_stream_keeps_block_only_for_a_masking_guardrail():
    guardrail = _MaskingGuardrail(guardrail_name="masker", event_hook="post_call")
    guardrail.streaming_buffer_until_moderated = False
    guardrail.streaming_end_of_stream_only = True
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/messages")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: list[Any] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict, response=_anthropic_stream(), request_data=request_data
    ):
        collected.append(chunk)
    raw = _decode(collected)
    assert ORIGINAL_MARKER in raw
    assert MASKED_MARKER not in raw


def _detect_only_pipeline(guardrail: CustomGuardrail) -> tuple[tuple[str, GuardrailPipeline], ...]:
    step = PipelineStep(guardrail=guardrail.guardrail_name, on_pass="allow", on_fail="next")
    return (("live-policy", GuardrailPipeline(mode="post_call", steps=[step])),)


@pytest.mark.parametrize(
    ("attribute", "guardrail_config", "streams_live"),
    (
        pytest.param(False, None, True, id="attribute-false-streams-live"),
        pytest.param(None, None, False, id="unset-keeps-buffering"),
        pytest.param(True, None, False, id="attribute-true-keeps-buffering"),
        pytest.param(False, {"streaming_buffer_until_moderated": True}, False, id="config-true-wins"),
        pytest.param(True, {"streaming_buffer_until_moderated": False}, True, id="config-false-wins"),
    ),
)
def test_pipeline_live_streaming_resolves_the_flag_like_the_flat_streaming_path(
    monkeypatch: pytest.MonkeyPatch,
    attribute: bool | None,
    guardrail_config: dict | None,
    streams_live: bool,
) -> None:
    guardrail = _PassingGuardrail(guardrail_name="pipeline-scanner", event_hook="post_call")
    if attribute is not None:
        guardrail.streaming_buffer_until_moderated = attribute
    if guardrail_config is not None:
        guardrail.guardrail_config = guardrail_config
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    assert _pipelines_stream_live(_detect_only_pipeline(guardrail)) is streams_live


async def _a2a_ndjson_stream(yielded_count: list[int], collected: list[Any]) -> AsyncGenerator[str, None]:
    for text in ["Here is ", ORIGINAL_MARKER, " for you."]:
        yielded_count.append(len(collected))
        yield (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "1",
                    "result": {"kind": "artifact-update", "artifact": {"parts": [{"kind": "text", "text": text}]}},
                }
            )
            + "\n"
        )


@pytest.mark.asyncio
async def test_buffered_masking_guardrail_streams_live_on_a_translation_that_cannot_deliver_rewrites():
    guardrail = _MaskingGuardrail(guardrail_name="masker", event_hook="post_call", mask_response_content=True)
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = False
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/a2a/agent-1/message/stream")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: list[Any] = []
    yielded_count: list[int] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_a2a_ndjson_stream(yielded_count, collected),
        request_data=request_data,
    ):
        collected.append(chunk)
    assert len(collected) == 3
    assert yielded_count == [0, 1, 2], "the stream was withheld although the rewrite cannot be delivered"


@pytest.mark.asyncio
async def test_buffered_chat_stream_withholds_everything_when_the_guardrail_returns_a_malformed_tool_call():
    guardrail = _ToolManglingGuardrail(guardrail_name="mangler", event_hook="post_call")
    with pytest.raises(HTTPException) as raised:
        await _run_buffered_chat(guardrail, [f"call {ORIGINAL_MARKER}"], ['{"cmd": "', TOOL_ARGUMENTS_MARKER, '"}'])
    assert raised.value.status_code == 500
    assert "mangler" in str(raised.value.detail)


@pytest.mark.asyncio
async def test_buffered_messages_stream_withholds_everything_when_the_guardrail_returns_a_malformed_tool_call():
    guardrail = _ToolManglingGuardrail(guardrail_name="mangler", event_hook="post_call")
    with pytest.raises(HTTPException) as raised:
        await _run_messages_stream(guardrail, _anthropic_tool_use_stream())
    assert raised.value.status_code == 500
    assert "mangler" in str(raised.value.detail)
