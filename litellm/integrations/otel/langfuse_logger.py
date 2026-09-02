from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_logger
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.mappers.langfuse import LANGFUSE_OBSERVATION_INPUT, LANGFUSE_OBSERVATION_OUTPUT
from litellm.integrations.otel.model.request_io import request_input, response_output, stream_output
from litellm.integrations.otel.plumbing.context import request_root_span

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import CallTypesLiteral, ModelResponseStream

ROOT_OBSERVATION_IO_CALL_TYPES: Final = frozenset(
    {"completion", "acompletion", "responses", "aresponses", "anthropic_messages", "aanthropic_messages"}
)


class LangfuseOpenTelemetryV2(OpenTelemetryV2):
    """Stamps the request's input and output on the root observation while it is still recording.

    Langfuse shows a trace's input and output from its root observation. The proxy's root span ends
    when the response is sent, before the success callback runs, so the stamps have to come from the
    request-task hooks: input at pre-call, output at post-call success or at the end of the stream.
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: Mapping[str, object],
        call_type: "CallTypesLiteral",
    ) -> None:
        await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)
        if call_type in ROOT_OBSERVATION_IO_CALL_TYPES:
            self._stamp_root(LANGFUSE_OBSERVATION_INPUT, lambda: request_input(data))

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, object],
        user_api_key_dict: "UserAPIKeyAuth",
        response: object,
    ) -> None:
        self._stamp_root(LANGFUSE_OBSERVATION_OUTPUT, lambda: response_output(response))

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: "AsyncIterator[ModelResponseStream]",
        request_data: Mapping[str, object],
    ) -> "AsyncGenerator[ModelResponseStream, None]":
        relayed: Final[list[ModelResponseStream]] = []  # mutable-ok: relayed as they arrive, assembled at end of stream
        async for chunk in response:
            relayed.append(chunk)
            yield chunk
        self._stamp_root(LANGFUSE_OBSERVATION_OUTPUT, lambda: stream_output(tuple(relayed), request_data))

    def _stamp_root(self, key: str, render: Callable[[], str | None]) -> None:
        root: Final = request_root_span()
        if root is None or not root.is_recording():
            return
        try:
            value: Final = render()
        except Exception:  # noqa: BLE001  # telemetry must never fail the request it describes
            verbose_logger.debug("otel v2 langfuse: could not render %s for the root observation", key, exc_info=True)
            return
        if value is not None:
            root.set_attribute(key, value)
