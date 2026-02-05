from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse


class _CompletedEvent:
    def __init__(self, response):
        self.response = response


class _FakeResponsesStream:
    def __init__(self, response):
        self._emitted = False
        self._response = response
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


def test_should_collect_response_from_stream():
    handler = ResponsesToCompletionBridgeHandler()
    response = ResponsesAPIResponse.model_construct(
        id="resp-1",
        created_at=0,
        output=[],
        object="response",
        model="gpt-5.2",
    )
    stream = _FakeResponsesStream(response)

    collected = handler._collect_response_from_stream(stream)

    assert collected.id == "resp-1"
    assert collected._hidden_params.get("headers") == {"x-test": "1"}
