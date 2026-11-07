"""Regression tests for Issue #50530: Responses-to-Chat streaming finish_reason when response.completed has empty output."""

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    OpenAiResponsesToChatCompletionStreamIterator,
)


def _stream_events(completed_output):
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "echo",
        "arguments": "",
    }
    done = {**item, "arguments": '{"text":"hi"}'}
    return [
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_1",
            "delta": '{"text":"hi"}',
        },
        {"type": "response.output_item.done", "output_index": 0, "item": done},
        {
            "type": "response.completed",
            "response": {"id": "r_1", "status": "completed", "output": completed_output},
        },
    ]


def test_streamed_function_call_with_empty_completed_output_finishes_with_tool_calls():
    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=iter(()),
        sync_stream=True,
    )
    chunks = [iterator.chunk_parser(event) for event in _stream_events([])]
    tool_call_chunks = sum(bool(c.choices[0].delta.tool_calls) for c in chunks if c.choices)
    finish_reasons = [c.choices[0].finish_reason for c in chunks if c.choices]

    assert tool_call_chunks == 2
    assert finish_reasons[-1] == "tool_calls"


def test_streamed_custom_tool_call_with_empty_completed_output_finishes_with_tool_calls():
    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=iter(()),
        sync_stream=True,
    )
    custom_item = {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_custom_1",
        "name": "code_exec",
        "input": "",
    }
    events = [
        {"type": "response.output_item.added", "output_index": 1, "item": custom_item},
        {
            "type": "response.custom_tool_call_input.delta",
            "output_index": 1,
            "item_id": "ctc_1",
            "delta": "print(1)",
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {**custom_item, "input": "print(1)"},
        },
        {
            "type": "response.completed",
            "response": {"id": "r_2", "status": "completed", "output": []},
        },
    ]
    chunks = [iterator.chunk_parser(event) for event in events]
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_streamed_text_only_with_empty_completed_output_still_finishes_with_stop():
    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=iter(()),
        sync_stream=True,
    )
    events = [
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Hello world"},
        {
            "type": "response.completed",
            "response": {"id": "r_3", "status": "completed", "output": []},
        },
    ]
    chunks = [iterator.chunk_parser(event) for event in events]
    assert chunks[-1].choices[0].finish_reason == "stop"
