from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from litellm.responses.streaming_iterator import dispatch_stream_success

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


@runtime_checkable
class _HandsOffCompletion(Protocol):
    def hand_completion_to(self, on_complete: Callable[[object], None]) -> None: ...


class HeldStreamCompletion:
    """The executor stream's assembled response, logged by the client-facing Messages stream when it ends."""

    def __init__(self, logging_obj: "LiteLLMLoggingObj") -> None:
        self._logging_obj: Final = logging_obj
        self._response: object | None = None

    def take(self, response: object) -> None:
        self._response = response

    def _release(self) -> object | None:
        response: Final = self._response
        self._response = None
        return response

    def discard(self) -> None:
        self._release()

    def log(self, *, is_async: bool, client_detached: bool = False) -> None:
        response: Final = self._release()
        if response is None:
            return
        dispatch_stream_success(
            self._logging_obj,
            response,
            start_time=None,
            end_time=None,
            cache_hit=None,
            is_async=is_async,
            client_detached=client_detached,
        )


def hold_completion_of(stream: object, logging_obj: "LiteLLMLoggingObj | None") -> HeldStreamCompletion | None:
    if logging_obj is None or not isinstance(stream, _HandsOffCompletion):
        return None
    held: Final = HeldStreamCompletion(logging_obj)
    stream.hand_completion_to(held.take)
    return held
