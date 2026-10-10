"""Regression tests: providers that deliver output items only via
``response.output_item.done`` events and leave ``response.output`` empty on the
terminal ``response.completed`` (e.g. the chatgpt/Codex backend) must not
surface an empty output list to the client — the iterator backfills the
terminal response from the items accumulated during the stream, and strips
provider-internal fields (``phase``/``logprobs``) from the reconstructed items.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _completed_response(output: list) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_backfill",
        created_at=0,
        status="completed",
        model="gpt-5.4-codex",
        object="response",
        output=output,
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


def _mock_config(completed_output: list) -> Mock:
    completed = _completed_response(completed_output)

    def _transform(model, parsed_chunk, logging_obj):
        evt_type = parsed_chunk.get("type")
        if evt_type == "response.completed":
            return ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=completed,
            )
        if evt_type in ("response.output_item.done", "response.output_text.done"):
            return SimpleNamespace(
                type=evt_type,
                output_index=parsed_chunk.get("output_index"),
                item=parsed_chunk.get("item"),
                text=parsed_chunk.get("text"),
            )
        stub = Mock()
        stub.type = evt_type
        return stub

    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.side_effect = _transform
    return mock_config


def _make_iterator(*, sse_events: list[bytes], logging_obj, config: Mock) -> ResponsesAPIStreamingIterator:
    async def aiter_bytes():
        for evt in sse_events:
            yield evt

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = aiter_bytes

    return ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5.4-codex",
        responses_api_provider_config=config,
        logging_obj=logging_obj,
        litellm_metadata={},
        custom_llm_provider="chatgpt",
    )


def _logging_obj_stub() -> Mock:
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.completion_start_time = None
    logging_obj.model_call_details = {"litellm_params": {}}
    return logging_obj


async def _terminal_chunk(iterator: ResponsesAPIStreamingIterator):
    async for chunk in iterator:
        if getattr(chunk, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
            return chunk
    return None


@pytest.mark.asyncio
async def test_empty_terminal_output_backfilled_from_item_done_events():
    """output_item.done items (plus a text-only fallback for an item that never
    got one) must be backfilled into the empty terminal response.output, with
    provider-internal phase/logprobs fields stripped."""
    logging_obj = _logging_obj_stub()
    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.created"}),
            _sse_event(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": "msg_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "phase": "thinking",
                        "content": [{"type": "output_text", "text": "hello", "logprobs": []}],
                    },
                }
            ),
            _sse_event({"type": "response.output_text.done", "output_index": 1, "text": "world"}),
            _sse_event({"type": "response.completed"}),
        ],
        logging_obj=logging_obj,
        config=_mock_config(completed_output=[]),
    )

    completed = await _terminal_chunk(iterator)

    assert completed is not None
    output = completed.response.output
    assert output, "terminal response.output must be backfilled from streamed items"
    assert output[0]["id"] == "msg_1"
    assert "phase" not in output[0], "provider-internal field must be stripped"
    assert "logprobs" not in output[0]["content"][0], "provider-internal field must be stripped"
    assert output[0]["content"][0]["text"] == "hello"
    assert output[1]["role"] == "assistant"
    assert output[1]["content"][0]["text"] == "world", "text-only item reconstructed without output_item.done"


@pytest.mark.asyncio
async def test_populated_terminal_output_is_not_overwritten():
    """When the provider fills response.output on the terminal event, the stream
    stays authoritative and nothing is backfilled."""
    provider_output = [
        {"id": "msg_provider", "type": "message", "role": "assistant", "content": []}
    ]
    logging_obj = _logging_obj_stub()
    iterator = _make_iterator(
        sse_events=[
            _sse_event(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {"id": "msg_streamed", "type": "message", "role": "assistant", "content": []},
                }
            ),
            _sse_event({"type": "response.completed"}),
        ],
        logging_obj=logging_obj,
        config=_mock_config(completed_output=provider_output),
    )

    completed = await _terminal_chunk(iterator)

    assert completed is not None
    assert completed.response.output == provider_output


@pytest.mark.asyncio
async def test_empty_terminal_output_without_streamed_items_stays_empty():
    """No accumulated items and an empty terminal output is a no-op — the stream
    must still complete normally instead of crashing."""
    logging_obj = _logging_obj_stub()
    iterator = _make_iterator(
        sse_events=[
            _sse_event({"type": "response.created"}),
            _sse_event({"type": "response.completed"}),
        ],
        logging_obj=logging_obj,
        config=_mock_config(completed_output=[]),
    )

    completed = await _terminal_chunk(iterator)

    assert completed is not None
    assert completed.response.output == []
