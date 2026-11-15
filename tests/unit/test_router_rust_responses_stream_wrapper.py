"""The Responses stream wrapper is one class for the process, not one per call (#51936).

A class is its own reference cycle (``__mro__`` holds it), so a class defined inside
``wrap_responses_stream`` kept the stream, its ``logging_obj`` (which holds the messages) and the
upstream ``httpx`` body alive until a generation-2 collection. Measured on the released 1.109.1
wheel, where the chat path had the same per-call pattern: with GC disabled and the stream dropped,
the logging object stayed alive until ``gc.collect()``; with the class hoisted out of the call it
was freed immediately.
"""

from litellm.router_rust.invoke import streaming


class _Attempt:
    """Only ``take_outcome`` is used before the wrapper's stream is iterated."""

    def take_outcome(self) -> None:
        return None


class _Source:
    model = "test-model"
    custom_llm_provider = "openai"
    response = None
    completed_response = None
    request_data = {}
    call_type = "responses"
    litellm_metadata = {}
    start_time = None
    responses_api_provider_config = None

    def __init__(self) -> None:
        self.logging_obj = object()


def test_the_wrapper_class_is_built_once_per_process() -> None:
    first = streaming._fallback_responses_stream_wrapper()
    assert streaming._fallback_responses_stream_wrapper() is first
    assert streaming._FALLBACK_RESPONSES_STREAM_WRAPPER is first


def test_two_streams_share_the_same_wrapper_class() -> None:
    first = streaming.wrap_responses_stream(_Attempt(), _Source())  # type: ignore[arg-type]
    second = streaming.wrap_responses_stream(_Attempt(), _Source())  # type: ignore[arg-type]

    assert type(first) is type(second)
