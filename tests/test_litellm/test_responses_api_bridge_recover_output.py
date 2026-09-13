from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse


class _CompletedEvent:
    def __init__(self, response):
        self.response = response


class _FakeResponsesStream:
    """
    Simulates a Responses API stream whose terminal response.completed event carries
    an empty output array even though output_item.done events streamed the answer
    (see issue #41009).
    """

    def __init__(self, response, streamed_items):
        self._emitted = False
        self._response = response
        self._streamed_items = streamed_items
        self.completed_response = None
        self._hidden_params = {"headers": {"x-test": "1"}}

    def __iter__(self):
        return self

    def __next__(self):
        if not self._emitted:
            self._emitted = True
            self.completed_response = _CompletedEvent(self._response)
            return {"type": "response.completed"}
        raise StopIteration

    def get_streamed_output_items(self):
        return self._streamed_items


def test_collect_response_from_stream_rebuilds_empty_output_from_streamed_items():
    handler = ResponsesToCompletionBridgeHandler()
    response = ResponsesAPIResponse.model_construct(
        id="resp-1",
        created_at=0,
        output=[],
        object="response",
        model="gpt-5.6-sol",
    )
    streamed_items = [
        {
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Hi! How can I help?", "annotations": []}
            ],
        }
    ]
    stream = _FakeResponsesStream(response, streamed_items)

    collected = handler._collect_response_from_stream(stream)

    assert len(collected.output) == 1
    assert "Hi! How can I help?" in str(collected.output[0])


def test_collect_response_from_stream_keeps_nonempty_output_untouched():
    handler = ResponsesToCompletionBridgeHandler()
    output = [
        {
            "id": "msg_0",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "from terminal event", "annotations": []}],
        }
    ]
    response = ResponsesAPIResponse.model_construct(
        id="resp-2",
        created_at=0,
        output=output,
        object="response",
        model="gpt-5.2",
    )
    stream = _FakeResponsesStream(response, [{"id": "msg_1", "type": "message"}])

    collected = handler._collect_response_from_stream(stream)

    assert len(collected.output) == 1
    assert "from terminal event" in str(collected.output[0])
