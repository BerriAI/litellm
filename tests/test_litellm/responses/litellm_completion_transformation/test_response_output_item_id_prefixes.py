"""
Regression tests for the Chat Completions -> Responses API bridge item IDs.

Bridged output items must carry Responses API ID prefixes (msg_, ig_, rs_) rather
than the upstream chatcmpl-* ID. Native OpenAI Responses rejects a replayed history
whose message item ID does not begin with "msg", and rejects an image generation
call whose ID does not begin with "ig".

Regression test for https://github.com/BerriAI/litellm/issues/27333
"""

from unittest.mock import Mock

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

CHAT_COMPLETION_ID = "chatcmpl-dfa2da3a-1586-4ff7-b64e-f59c692a5d11"


def _make_chat_completion_response(**overrides) -> ModelResponse:
    defaults = dict(
        id=CHAT_COMPLETION_ID,
        created=1717000000,
        model="claude-sonnet-4-5",
        object="chat.completion",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content="apple"),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    defaults.update(overrides)
    return ModelResponse(**defaults)


def _transform(chat_completion_response):
    return LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="Say the single word: apple",
        responses_api_request={},
        chat_completion_response=chat_completion_response,
    )


def _output_items_of_type(response, item_type):
    return [item for item in response.output if getattr(item, "type", None) == item_type]


class TestMessageOutputItemIds:
    def test_message_item_id_uses_msg_prefix(self):
        response = _transform(_make_chat_completion_response())

        message_items = _output_items_of_type(response, "message")
        assert len(message_items) == 1
        assert message_items[0].id.startswith("msg_")

    def test_message_item_id_does_not_leak_chat_completion_id(self):
        response = _transform(_make_chat_completion_response())

        for item in _output_items_of_type(response, "message"):
            assert item.id != CHAT_COMPLETION_ID
            assert not item.id.startswith("chatcmpl-")

    def test_message_item_ids_are_unique_across_responses(self):
        first = _transform(_make_chat_completion_response())
        second = _transform(_make_chat_completion_response())

        first_id = _output_items_of_type(first, "message")[0].id
        second_id = _output_items_of_type(second, "message")[0].id
        assert first_id != second_id


class TestImageGenerationOutputItemIds:
    def _make_choice_with_images(self, count):
        message = Mock(spec=Message)
        message.images = [
            {"image_url": {"url": f"data:image/png;base64,IMG{idx}"}} for idx in range(count)
        ]
        choice = Mock(spec=Choices)
        choice.message = message
        choice.finish_reason = "stop"
        return choice

    def test_image_generation_item_id_uses_ig_prefix(self):
        items = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=_make_chat_completion_response(),
            choice=self._make_choice_with_images(2),
        )

        assert len(items) == 2
        for item in items:
            assert item.id.startswith("ig_")
            assert "chatcmpl-" not in item.id
            assert "_img_" not in item.id

    def test_image_generation_item_ids_are_unique(self):
        items = LiteLLMCompletionResponsesConfig._extract_image_generation_output_items(
            chat_completion_response=_make_chat_completion_response(),
            choice=self._make_choice_with_images(3),
        )

        assert len({item.id for item in items}) == 3


class TestReasoningOutputItemIds:
    def _reasoning_items(self):
        message = Message(role="assistant", content="apple")
        message.reasoning_content = "thinking about fruit"
        choice = Choices(index=0, finish_reason="stop", message=message)
        return LiteLLMCompletionResponsesConfig._extract_reasoning_output_items(
            chat_completion_response=_make_chat_completion_response(),
            choices=[choice],
        )

    def test_reasoning_item_id_uses_rs_prefix(self):
        items = self._reasoning_items()

        assert len(items) == 1
        assert items[0].id.startswith("rs_")

    def test_reasoning_item_id_is_not_a_salted_hash(self):
        item_id = self._reasoning_items()[0].id

        suffix = item_id.removeprefix("rs_")
        assert not suffix.lstrip("-").isdigit()
        assert not suffix.startswith("-")


class TestStreamingItemIdConsistency:
    def _make_iterator(self):
        mock_stream_wrapper = Mock(spec=litellm.CustomStreamWrapper)
        mock_stream_wrapper.logging_obj = Mock()
        return LiteLLMCompletionStreamingIterator(
            model="anthropic/claude-sonnet-4-5",
            litellm_custom_stream_wrapper=mock_stream_wrapper,
            request_input="Say the single word: apple",
            responses_api_request={},
            custom_llm_provider="anthropic",
        )

    def _make_chunk(self, chunk_id, content, finish_reason=None):
        return ModelResponseStream(
            id=chunk_id,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=content, role="assistant"),
                    finish_reason=finish_reason,
                )
            ],
            created=1717000000,
            model="claude-sonnet-4-5",
            object="chat.completion.chunk",
        )

    def test_incremental_item_id_uses_msg_prefix(self):
        iterator = self._make_iterator()

        event = iterator._transform_chat_completion_chunk_to_response_api_chunk(
            self._make_chunk(CHAT_COMPLETION_ID, "apple")
        )

        assert event is not None
        assert event.item_id.startswith("msg_")
        assert event.item_id != CHAT_COMPLETION_ID

    def test_completed_snapshot_reuses_streamed_item_id(self):
        iterator = self._make_iterator()

        streamed_event = iterator._transform_chat_completion_chunk_to_response_api_chunk(
            self._make_chunk(CHAT_COMPLETION_ID, "apple")
        )
        assert streamed_event is not None
        streamed_item_id = streamed_event.item_id

        completed_event = iterator._emit_response_completed_event(
            _make_chat_completion_response()
        )

        assert completed_event is not None
        message_items = _output_items_of_type(completed_event.response, "message")
        assert len(message_items) == 1
        assert message_items[0].id == streamed_item_id

    def test_completed_snapshot_item_id_is_replayable(self):
        iterator = self._make_iterator()
        iterator._transform_chat_completion_chunk_to_response_api_chunk(
            self._make_chunk(CHAT_COMPLETION_ID, "apple")
        )

        completed_event = iterator._emit_response_completed_event(
            _make_chat_completion_response()
        )

        assert completed_event is not None
        for item in _output_items_of_type(completed_event.response, "message"):
            assert item.id.startswith("msg_")
            assert not item.id.startswith("chatcmpl-")
