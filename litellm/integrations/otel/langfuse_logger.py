from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import TYPE_CHECKING, Final

from opentelemetry.sdk.trace import ReadableSpan

from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.mappers.base import AttributeMap
from litellm.integrations.otel.mappers.langfuse import (
    LANGFUSE_OBSERVATION_INPUT,
    LANGFUSE_OBSERVATION_OUTPUT,
    LANGFUSE_TRACE_NAME,
    LangfuseMapper,
)
from litellm.integrations.otel.mappers.utils import drop_none_pairs
from litellm.integrations.otel.model.metadata import internal_call_origin
from litellm.integrations.otel.model.trace_controls import caller_trace_controls
from litellm.integrations.otel.model.utils import as_str
from litellm.integrations.otel.plumbing.context import request_root_span
from litellm.integrations.otel.root_io_logger import RootIOOpenTelemetryV2

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.types.utils import ModelResponseStream


class LangfuseOpenTelemetryV2(OpenTelemetryV2):
    """Stamps the caller's trace controls (name, user, session, tags) on the request. Langfuse reads them off
    the root observation, and the proxy's root span is still recording when the LLM call starts."""

    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        root: Final = request_root_span()
        if root is not None and root.is_recording():
            root.set_attributes(
                LangfuseMapper.trace_attributes(
                    caller_trace_controls(kwargs),
                    as_str(kwargs.get("call_type")) if self._derives_trace_name(root, kwargs) else None,
                )
            )
        super().log_pre_api_call(model, messages, kwargs)

    def _derives_trace_name(self, root: "Span", kwargs: Mapping[str, object]) -> bool:
        """Whether this call may name the request root after its call type: the call is not one litellm made for
        its own features, this logger's provider started the root with no parent, and neither the caller nor an
        earlier call in the request has named it."""
        return (
            internal_call_origin(None, kwargs) is None
            and isinstance(root, ReadableSpan)
            and root.parent is None
            and root.resource is self.tracer_provider.resource
            and LANGFUSE_TRACE_NAME not in (root.attributes or {})
        )


class LangfuseContentOpenTelemetryV2(RootIOOpenTelemetryV2, LangfuseOpenTelemetryV2):
    """Stamps the request's input and output on the root observation while it is still recording, where Langfuse
    shows a trace's input and output."""

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: "AsyncIterator[ModelResponseStream]",
        request_data: Mapping[str, object],
    ) -> "AsyncGenerator[ModelResponseStream, None]":
        async for chunk in self._relay_stream(response, request_data):
            yield chunk

    def _root_io_attributes(self, rendered_input: str | None, output: str) -> AttributeMap:
        return drop_none_pairs(((LANGFUSE_OBSERVATION_OUTPUT, output), (LANGFUSE_OBSERVATION_INPUT, rendered_input)))
