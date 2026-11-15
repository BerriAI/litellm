"""The Responses stream wrapper is one class for the process, not one per call (#51936).

A class is its own reference cycle (``__mro__`` holds it), so a class defined inside
``wrap_responses_stream`` kept the stream, its ``logging_obj`` (which holds the messages) and the
upstream ``httpx`` body alive until a generation-2 collection. On litellm 1.109.1 the same pattern
in the chat path was measured: after dropping the stream with GC disabled, the logging object stayed
alive; with the class hoisted out of the call it was freed immediately.
"""

from litellm.router_rust.invoke.streaming import wrap_responses_stream


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


def test_responses_stream_wrappers_share_one_class() -> None:
    first = wrap_responses_stream(_Attempt(), _Source())  # type: ignore[arg-type]
    second = wrap_responses_stream(_Attempt(), _Source())  # type: ignore[arg-type]

    assert type(first) is type(second)
    assert "<locals>" not in type(first).__qualname__
