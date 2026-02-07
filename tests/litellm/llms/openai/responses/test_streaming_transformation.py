from __future__ import annotations

from unittest.mock import MagicMock

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig


def test_response_in_progress_coalesces_missing_response_fields() -> None:
    cfg = OpenAIResponsesAPIConfig()

    parsed_chunk = {
        "type": "response.in_progress",
        "id": "resp_test_in_progress_1",
        "response": {
            "id": "resp_test_in_progress_1",
            # created_at/output intentionally missing
            "status": "in_progress",
        },
    }

    evt = cfg.transform_streaming_response(
        model="gpt-oss-120b",
        parsed_chunk=parsed_chunk,
        logging_obj=MagicMock(),
    )


def test_output_item_added_coalesces_missing_output_index() -> None:
    cfg = OpenAIResponsesAPIConfig()

    parsed_chunk = {
        "type": "response.output_item.added",
        # output_index intentionally missing
        "item": {
            "id": "rs_test_item_1",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        },
        # include response so the event is closer to real SSE payloads
        "response": {
            "id": "resp_test_indices_1",
            "status": "in_progress",
            "created_at": 1,
            "output": [],
        },
    }

    evt = cfg.transform_streaming_response(
        model="gpt-oss-120b",
        parsed_chunk=parsed_chunk,
        logging_obj=MagicMock(),
    )


def test_output_text_delta_coalesces_missing_output_index_and_content_index() -> None:
    cfg = OpenAIResponsesAPIConfig()

    parsed_chunk = {
        "type": "response.output_text.delta",
        # output_index/content_index intentionally missing
        "item_id": "rs_test_item_2",
        "delta": "hi",
        "response": {
            "id": "resp_test_indices_2",
            "status": "in_progress",
            "created_at": 1,
            "output": [],
        },
    }

    evt = cfg.transform_streaming_response(
        model="gpt-oss-120b",
        parsed_chunk=parsed_chunk,
        logging_obj=MagicMock(),
    )


def test_noop_when_required_fields_already_present() -> None:
    cfg = OpenAIResponsesAPIConfig()

    parsed_chunk = {
        "type": "response.created",
        "id": "resp_test_noop_1",
        "response": {
            "id": "resp_test_noop_1",
            "status": "in_progress",
            "created_at": 123,
            "output": [],
        },
    }

    evt = cfg.transform_streaming_response(
        model="gpt-oss-120b",
        parsed_chunk=parsed_chunk,
        logging_obj=MagicMock(),
    )
