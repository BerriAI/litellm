"""
Regression for #25429.

chatgpt.com's Codex backend sends response.completed with an empty output
array. The actual assistant content arrives via preceding
response.output_item.done events. Without accumulation in the streaming
iterator, completed_response.response.output is [] and the
chat-completions bridge raises "Unknown items in responses API response: []".

These tests verify that BaseResponsesAPIStreamingIterator accumulates
output_item.done payloads (via _accumulate_streamed_output_item, called
after post-call hooks in __anext__/__next__) and backfills them into the
response.completed chunk before storing it as completed_response.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx

sys.path.insert(0, os.path.abspath("../../.."))


def _make_iterator():
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.responses import streaming_iterator as _si_mod
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.start_time = None
    logging_obj.completion_start_time = None

    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, text="")

    # Patch get_api_base to avoid triggering chatgpt device-auth during construction.
    with patch.object(_si_mod, "get_api_base", return_value=None):
        return BaseResponsesAPIStreamingIterator(
            response=response,
            model="chatgpt/gpt-5.4",
            responses_api_provider_config=OpenAIResponsesAPIConfig(),
            logging_obj=logging_obj,
            custom_llm_provider="chatgpt",
            request_data={},
        )


_OUTPUT_ITEM = {
    "type": "message",
    "id": "msg_0",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "OK my lord", "annotations": []}],
    "status": "completed",
}

_RESPONSE_BASE = {
    "id": "resp_test",
    "object": "response",
    "created_at": 1700000000,
    "status": "completed",
    "model": "gpt-5.4",
}


def _process_and_accumulate(iterator, raw_json: str) -> None:
    """
    Simulate what __anext__ does: _process_chunk then _accumulate_streamed_output_item.
    Accumulation now happens on the post-hook chunk; in tests there are no hooks,
    so the chunk is unchanged and we call both methods in sequence.
    """
    result = iterator._process_chunk(raw_json)
    if result is not None:
        iterator._accumulate_streamed_output_item(result)


def _text_from_output_item(item) -> str:
    content = item["content"] if isinstance(item, dict) else item.content
    part = content[0]
    return part["text"] if isinstance(part, dict) else part.text


class TestStreamingIteratorOutputRecovery:
    def test_completed_response_output_stays_empty_without_preceding_items(self):
        """
        Baseline: response.completed with output:[] and no preceding
        output_item.done leaves completed_response.response.output empty.
        Confirms the backfill only activates when items were actually streamed.
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            iterator._process_chunk(
                json.dumps({"type": "response.created", "response": {**_RESPONSE_BASE, "output": []}})
            )
            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        assert iterator.completed_response is not None
        output = getattr(iterator.completed_response.response, "output", None)
        assert output == [] or output is None

    def test_completed_response_output_backfilled_from_output_item_done(self):
        """
        Core regression: when response.completed.output is [] but
        response.output_item.done events preceded it, the iterator
        backfills output so completed_response.response.output is non-empty.

        Before the fix this assertion would fail because output stayed [].
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({"type": "response.output_item.done", "output_index": 0, "item": _OUTPUT_ITEM})
            )
            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        assert iterator.completed_response is not None
        output = iterator.completed_response.response.output
        assert len(output) == 1
        assert _text_from_output_item(output[0]) == "OK my lord"
        assert isinstance(output[0], dict), f"expected dict for transformation compat, got {type(output[0])}"
        assert output[0].get("type") == "message"

    def test_authoritative_output_is_not_overwritten(self):
        """
        When the provider sends a non-empty output in response.completed,
        the backfill must not overwrite it.
        """
        authoritative_item = {
            **_OUTPUT_ITEM,
            "id": "msg_auth",
            "content": [{"type": "output_text", "text": "authoritative", "annotations": []}],
        }
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({"type": "response.output_item.done", "output_index": 0, "item": _OUTPUT_ITEM})
            )
            iterator._process_chunk(
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {**_RESPONSE_BASE, "output": [authoritative_item]},
                    }
                )
            )

        output = iterator.completed_response.response.output
        assert len(output) == 1
        assert _text_from_output_item(output[0]) == "authoritative"

    def test_multiple_output_items_ordered_by_index(self):
        """
        Multiple output_item.done events are ordered by output_index,
        not by the order they arrived.
        """
        item_a = {
            **_OUTPUT_ITEM,
            "id": "msg_a",
            "content": [{"type": "output_text", "text": "A", "annotations": []}],
        }
        item_b = {
            **_OUTPUT_ITEM,
            "id": "msg_b",
            "content": [{"type": "output_text", "text": "B", "annotations": []}],
        }
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({"type": "response.output_item.done", "output_index": 1, "item": item_b})
            )
            _process_and_accumulate(iterator, 
                json.dumps({"type": "response.output_item.done", "output_index": 0, "item": item_a})
            )
            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        output = iterator.completed_response.response.output
        assert len(output) == 2
        assert _text_from_output_item(output[0]) == "A"
        assert _text_from_output_item(output[1]) == "B"

    def test_output_text_done_backfilled_when_no_output_item_done(self):
        """
        Fallback for providers that emit OUTPUT_TEXT_DONE without OUTPUT_ITEM_DONE:
        the text-only item is used to backfill output when response.completed.output
        is empty and no OUTPUT_ITEM_DONE events were received.
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_text_only",
                    "text": "text only content",
                })
            )
            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        assert iterator.completed_response is not None
        output = iterator.completed_response.response.output
        assert len(output) == 1
        assert _text_from_output_item(output[0]) == "text only content"

    def test_output_item_done_takes_precedence_over_output_text_done(self):
        """
        When both OUTPUT_TEXT_DONE and OUTPUT_ITEM_DONE arrive for the same
        output_index, the real OUTPUT_ITEM_DONE item must win.
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_text_only",
                    "text": "text only content",
                })
            )
            _process_and_accumulate(iterator, 
                json.dumps({"type": "response.output_item.done", "output_index": 0, "item": _OUTPUT_ITEM})
            )
            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        output = iterator.completed_response.response.output
        assert len(output) == 1
        assert _text_from_output_item(output[0]) == "OK my lord"

    def test_incomplete_response_output_backfilled_from_output_item_done(self):
        """
        response.incomplete (max-tokens truncation) with output:[] must be
        backfilled the same way as response.completed — partial content that
        arrived via output_item.done must not be silently dropped.
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({"type": "response.output_item.done", "output_index": 0, "item": _OUTPUT_ITEM})
            )
            iterator._process_chunk(
                json.dumps({
                    "type": "response.incomplete",
                    "response": {**_RESPONSE_BASE, "status": "incomplete", "output": []},
                })
            )

        assert iterator.completed_response is not None
        output = iterator.completed_response.response.output
        assert len(output) == 1
        assert _text_from_output_item(output[0]) == "OK my lord"

    def test_response_failed_sets_completed_response_without_backfill(self):
        """
        response.failed must set completed_response so logging still fires,
        but must NOT backfill output (no content to recover from a failed response).
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({"type": "response.output_item.done", "output_index": 0, "item": _OUTPUT_ITEM})
            )
            iterator._process_chunk(
                json.dumps({
                    "type": "response.failed",
                    "response": {**_RESPONSE_BASE, "status": "failed", "output": []},
                })
            )

        assert iterator.completed_response is not None
        output = getattr(iterator.completed_response.response, "output", None)
        assert output == [] or output is None

    def test_output_text_done_replace_in_place(self):
        """
        When a second OUTPUT_TEXT_DONE arrives for the same output_index and
        content_index as an existing slot, it must replace that slot in-place
        rather than appending.
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_replace",
                    "text": "first",
                })
            )
            _process_and_accumulate(iterator, 
                json.dumps({
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_replace",
                    "text": "replaced",
                })
            )
            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        output = iterator.completed_response.response.output
        assert len(output) == 1
        content = output[0]["content"]
        assert len(content) == 1
        assert content[0]["text"] == "replaced"

    def test_output_text_done_gap_padding(self):
        """
        When OUTPUT_TEXT_DONE arrives with content_index=2 and no prior slots,
        the iterator must insert two empty placeholder slots before it.
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 2,
                    "item_id": "msg_gap",
                    "text": "late content",
                })
            )
            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        output = iterator.completed_response.response.output
        assert len(output) == 1
        content = output[0]["content"]
        assert len(content) == 3
        assert content[0]["text"] == ""
        assert content[1]["text"] == ""
        assert content[2]["text"] == "late content"

    def test_output_index_absent_uses_sequential_fallback(self):
        """
        When an OUTPUT_ITEM_DONE chunk lacks the output_index field entirely,
        the iterator assigns a synthetic index via max(existing)+1 so items
        are not silently dropped.
        """
        item_a = {**_OUTPUT_ITEM, "id": "msg_a", "content": [{"type": "output_text", "text": "A", "annotations": []}]}
        item_b = {**_OUTPUT_ITEM, "id": "msg_b", "content": [{"type": "output_text", "text": "B", "annotations": []}]}
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, json.dumps({"type": "response.output_item.done", "output_index": 0, "item": item_a}))
            _process_and_accumulate(iterator, json.dumps({"type": "response.output_item.done", "item": item_b}))
            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        output = iterator.completed_response.response.output
        assert len(output) == 2
        assert _text_from_output_item(output[0]) == "A"
        assert _text_from_output_item(output[1]) == "B"

    def test_backfill_exception_is_swallowed(self):
        """
        If model_dump() raises during backfill, the exception must be swallowed
        and logged as a warning rather than crashing the stream. completed_response
        is still set so logging fires.
        """
        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            _process_and_accumulate(iterator, 
                json.dumps({"type": "response.output_item.done", "output_index": 0, "item": _OUTPUT_ITEM})
            )
            with patch.object(
                iterator._streamed_output_items[0],
                "model_dump",
                side_effect=RuntimeError("serialization failure"),
            ):
                iterator._process_chunk(
                    json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
                )

        assert iterator.completed_response is not None

    def test_post_hook_item_is_accumulated_not_pre_hook(self):
        """
        Accumulation must happen after async_post_call_streaming_deployment_hook runs,
        not before. If a hook replaces the item on the chunk, the replaced (post-hook)
        item must be what ends up in completed_response.response.output.
        """
        from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject

        redacted_item = BaseLiteLLMOpenAIResponseObject(
            type="message",
            id="msg_redacted",
            role="assistant",
            status="completed",
            content=[{"type": "output_text", "text": "[REDACTED]", "annotations": []}],
        )

        iterator = _make_iterator()

        with patch.object(iterator, "_handle_logging_completed_response"):
            raw = json.dumps({"type": "response.output_item.done", "output_index": 0, "item": _OUTPUT_ITEM})
            pre_hook_chunk = iterator._process_chunk(raw)
            assert pre_hook_chunk is not None
            # Simulate a hook that replaced the item on the chunk.
            pre_hook_chunk.item = redacted_item
            iterator._accumulate_streamed_output_item(pre_hook_chunk)

            iterator._process_chunk(
                json.dumps({"type": "response.completed", "response": {**_RESPONSE_BASE, "output": []}})
            )

        output = iterator.completed_response.response.output
        assert len(output) == 1
        assert output[0]["content"][0]["text"] == "[REDACTED]"
