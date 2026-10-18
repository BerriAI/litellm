from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Final

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Span

from litellm._logging import verbose_logger
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.mappers.langfuse import (
    LANGFUSE_OBSERVATION_INPUT,
    LANGFUSE_OBSERVATION_OUTPUT,
    LangfuseMapper,
)
from litellm.integrations.otel.model.request_io import request_input, response_output, stream_output
from litellm.integrations.otel.model.trace_controls import caller_trace_controls, request_trace_controls
from litellm.integrations.otel.plumbing.context import request_root_span

if TYPE_CHECKING:
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.types.utils import ModelResponseStream


class LangfuseOpenTelemetryV2(OpenTelemetryV2):
    """Stamps the caller's trace controls (name, user, session, tags) on the request. Langfuse reads them off
    the root observation, and the proxy's root span is still recording when the LLM call starts.

    The proxy end user fills a missing user only on a root this logger's own provider started. With another v2
    callback listed first, that callback's provider starts the root and exports it to its own backend.
    """

    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        root: Final = request_root_span()
        if root is not None and root.is_recording():
            controls: Final = (
                request_trace_controls(kwargs).released(self.config.baggage_metadata_keys)
                if self._started(root)
                else caller_trace_controls(kwargs)
            )
            root.set_attributes(LangfuseMapper.trace_attributes(controls))
        super().log_pre_api_call(model, messages, kwargs)

    def _started(self, span: Span) -> bool:
        return isinstance(span, ReadableSpan) and span.resource is self.tracer_provider.resource


class LangfuseContentOpenTelemetryV2(LangfuseOpenTelemetryV2):
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
        if not self._capture_span_content():
            async for chunk in response:
                yield chunk
            return
        relayed: Final[list[ModelResponseStream]] = []  # mutable-ok: relayed as they arrive, assembled at end of stream
        async for chunk in response:
            relayed.append(chunk)
            yield chunk
        self._stamp_root_io(request_data, lambda: stream_output(tuple(relayed), request_data))

    def _stamp_root_io(self, data: Mapping[str, object], render_output: Callable[[], str | None]) -> None:
        if not self._capture_span_content():
            return
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
