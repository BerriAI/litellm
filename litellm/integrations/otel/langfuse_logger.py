from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_logger
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.mappers.langfuse import LANGFUSE_OBSERVATION_INPUT, LANGFUSE_OBSERVATION_OUTPUT
from litellm.integrations.otel.model.request_io import request_input, response_output, stream_output
from litellm.integrations.otel.plumbing.context import request_root_span

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import ModelResponseStream


class LangfuseOpenTelemetryV2(OpenTelemetryV2):
    """Stamps the request's input and output on the root observation while it is still recording.

    Langfuse shows a trace's input and output from its root observation. The proxy's root span ends
    when the response is sent, before the success callback runs, so both stamps come from the
    post-call hooks in the request task: the request as it stands after the pre-call chain and the
    response as it is returned, for the call types whose response renders as a message.
    """

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, object],
        user_api_key_dict: "UserAPIKeyAuth",
        response: object,
    ) -> None:
        self._stamp_root_io(data, lambda: response_output(response))

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
        self._stamp_root_io(request_data, lambda: stream_output(tuple(relayed), request_data))

    def _stamp_root_io(self, data: Mapping[str, object], render_output: Callable[[], str | None]) -> None:
        root: Final = request_root_span()
        if root is None or not root.is_recording():
            return
        try:
            output: Final = render_output()
            if output is None:
                return
            root.set_attribute(LANGFUSE_OBSERVATION_OUTPUT, output)
            rendered_input: Final = request_input(data)
        except Exception:  # noqa: BLE001  # telemetry must never fail the request it describes
            verbose_logger.debug(
                "otel v2 langfuse: could not render the root observation input or output", exc_info=True
            )
            return
        if rendered_input is not None:
            root.set_attribute(LANGFUSE_OBSERVATION_INPUT, rendered_input)
