"""Contract tests for synthetic Responses API replay events.

``_build_synthetic_response_events`` (used by
``CachedResponsesAPIStreamingIterator`` to replay cached streamed
responses) must emit ``*.added`` events with their incrementally
streamed fields empty, matching real OpenAI streams. Carrying the full
payload on ``added`` while also re-streaming it as deltas makes
spec-following clients (which seed from ``added`` and append deltas)
assemble the content twice — for function calls the arguments JSON is
doubled, which is invalid JSON.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

from litellm.responses import streaming_iterator as streaming_module
from litellm.types.llms.openai import ResponsesAPIResponse


def _FakeLoggingObj():
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}
    return logging_obj


def _build_full_fixture_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_contract",
        created_at=int(datetime.now().timestamp()),
        status="completed",
        model="gpt-4.1-mini",
        object="response",
        output=[
            {
                "type": "function_call",
                "id": "fc_contract",
                "call_id": "call_contract",
                "name": "get_weather",
                "arguments": '{"city":"Paris","unit":"celsius"}',
            },
            {
                "type": "message",
                "id": "msg_contract",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello, world!",
                        "annotations": [{"type": "file_citation", "file_id": "file_abc"}],
                    },
                    {
                        "type": "refusal",
                        "refusal": "I cannot do that.",
                    },
                ],
            },
            {
                "type": "reasoning",
                "id": "rs_x",
                "summary": [{"type": "summary_text", "text": "thinking..."}],
            },
        ],
    )


def test_synthetic_output_item_added_has_empty_incremental_fields():
    transformed = _build_full_fixture_response()

    events = streaming_module._build_synthetic_response_events(
        transformed=transformed,
        logging_obj=_FakeLoggingObj(),
        chunk_size=5,
    )

    def event_type_str(e):
        return e.type.value if hasattr(e.type, "value") else str(e.type)

    added_events = [e for e in events if event_type_str(e) == "response.output_item.added"]
    content_part_added_events = [
        e for e in events if event_type_str(e) == "response.content_part.added"
    ]

    fc_added = next(
        e for e in added_events if getattr(e.item, "type", None) == "function_call"
    )
    msg_added = next(
        e for e in added_events if getattr(e.item, "type", None) == "message"
    )
    reasoning_added = next(
        e for e in added_events if getattr(e.item, "type", None) == "reasoning"
    )

    assert getattr(fc_added.item, "arguments", None) == "", (
        f"function_call output_item.added should have arguments='' but got: "
        f"{getattr(fc_added.item, 'arguments', '<missing>')!r}"
    )

    assert getattr(msg_added.item, "content", None) == [], (
        f"message output_item.added should have content=[] but got: "
        f"{getattr(msg_added.item, 'content', '<missing>')!r}"
    )

    assert getattr(reasoning_added.item, "summary", None) == [], (
        f"reasoning output_item.added should have summary=[] but got: "
        f"{getattr(reasoning_added.item, 'summary', '<missing>')!r}"
    )

    output_text_part_added = next(
        e
        for e in content_part_added_events
        if getattr(e.part, "type", None) == "output_text"
    )
    assert getattr(output_text_part_added.part, "text", None) == "", (
        f"output_text content_part.added should have text='' but got: "
        f"{getattr(output_text_part_added.part, 'text', '<missing>')!r}"
    )
    assert getattr(output_text_part_added.part, "annotations", None) == [], (
        f"output_text content_part.added should have annotations=[] but got: "
        f"{getattr(output_text_part_added.part, 'annotations', '<missing>')!r}"
    )

    refusal_part_added = next(
        e
        for e in content_part_added_events
        if getattr(e.part, "type", None) == "refusal"
    )
    assert getattr(refusal_part_added.part, "refusal", None) == "", (
        f"refusal content_part.added should have refusal='' but got: "
        f"{getattr(refusal_part_added.part, 'refusal', '<missing>')!r}"
    )


def test_synthetic_replay_client_assembly_is_not_duplicated():
    fc_args = '{"city":"Paris","unit":"celsius"}'
    original_text = "Hello, world!"
    transformed = _build_full_fixture_response()

    events = streaming_module._build_synthetic_response_events(
        transformed=transformed,
        logging_obj=_FakeLoggingObj(),
        chunk_size=5,
    )

    def event_type_str(e):
        return e.type.value if hasattr(e.type, "value") else str(e.type)

    fc_added_item = next(
        e.item
        for e in events
        if event_type_str(e) == "response.output_item.added"
        and getattr(e, "output_index", None) == 0
    )
    fc_arg_deltas = [
        e
        for e in events
        if event_type_str(e) == "response.function_call_arguments.delta"
        and getattr(e, "output_index", None) == 0
    ]

    assembled_args = getattr(fc_added_item, "arguments", "") + "".join(
        e.delta for e in fc_arg_deltas
    )
    assert assembled_args == fc_args, (
        f"Client assembly of tool arguments doubled: expected {fc_args!r}, got {assembled_args!r}"
    )
    assert json.loads(assembled_args) == {"city": "Paris", "unit": "celsius"}, (
        f"Assembled arguments are not valid JSON: {assembled_args!r}"
    )

    text_part_added = next(
        e
        for e in events
        if event_type_str(e) == "response.content_part.added"
        and getattr(e, "output_index", None) == 1
        and getattr(e.part, "type", None) == "output_text"
    )
    output_text_deltas = [
        e
        for e in events
        if event_type_str(e) == "response.output_text.delta"
        and getattr(e, "output_index", None) == 1
        and getattr(e, "content_index", None) == 0
    ]

    assembled_text = getattr(text_part_added.part, "text", "") + "".join(
        e.delta for e in output_text_deltas
    )
    assert assembled_text == original_text, (
        f"Client assembly of message text doubled: expected {original_text!r}, got {assembled_text!r}"
    )


def test_synthetic_done_events_keep_full_payloads():
    fc_args = '{"city":"Paris","unit":"celsius"}'
    original_text = "Hello, world!"
    transformed = _build_full_fixture_response()

    events = streaming_module._build_synthetic_response_events(
        transformed=transformed,
        logging_obj=_FakeLoggingObj(),
        chunk_size=5,
    )

    def event_type_str(e):
        return e.type.value if hasattr(e.type, "value") else str(e.type)

    fc_args_done = next(
        e for e in events if event_type_str(e) == "response.function_call_arguments.done"
    )
    assert fc_args_done.arguments == fc_args, (
        f"function_call_arguments.done should carry full args, got: {fc_args_done.arguments!r}"
    )

    output_text_done = next(
        e for e in events if event_type_str(e) == "response.output_text.done"
    )
    assert output_text_done.text == original_text, (
        f"output_text.done should carry full text, got: {output_text_done.text!r}"
    )

    fc_item_done = next(
        e
        for e in events
        if event_type_str(e) == "response.output_item.done"
        and getattr(e.item, "type", None) == "function_call"
    )
    assert getattr(fc_item_done.item, "arguments", None) == fc_args, (
        f"output_item.done for function_call should have full arguments, got: "
        f"{getattr(fc_item_done.item, 'arguments', '<missing>')!r}"
    )

    msg_item_done = next(
        e
        for e in events
        if event_type_str(e) == "response.output_item.done"
        and getattr(e.item, "type", None) == "message"
    )
    assert getattr(msg_item_done.item, "content", None), (
        f"output_item.done for message should have non-empty content, got: "
        f"{getattr(msg_item_done.item, 'content', '<missing>')!r}"
    )




def test_synthetic_added_passthrough_for_types_without_incremental_fields():
    """Item types with no delta-streamed fields (e.g. web_search_call)
    must pass through to ``output_item.added`` unchanged."""
    transformed = ResponsesAPIResponse(
        id="resp_passthrough",
        created_at=int(datetime.now().timestamp()),
        status="completed",
        model="gpt-4.1-mini",
        object="response",
        output=[
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
            },
        ],
    )

    events = streaming_module._build_synthetic_response_events(
        transformed=transformed,
        logging_obj=_FakeLoggingObj(),
        chunk_size=5,
    )

    added = next(
        e
        for e in events
        if (e.type.value if hasattr(e.type, "value") else str(e.type))
        == "response.output_item.added"
    )
    assert getattr(added.item, "type", None) == "web_search_call"
    assert getattr(added.item, "id", None) == "ws_1"
    assert getattr(added.item, "status", None) == "completed"
